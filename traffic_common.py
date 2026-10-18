"""Shared signal scheduler and episode measurements for all four environments."""
from __future__ import annotations

import tempfile
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import traci

from traffic_demand import materialize_routes, scheduled_departures

SCHEMA_VERSION = 'corrected-v1'
DIRECTIONS = ('north', 'south', 'east', 'west')
EDGES_8 = {
    'tls_1': ('north_to_i1', 'i3_to_i1', 'i2_to_i1', 'west_to_i1'),
    'tls_2': ('north_to_i2', 'i4_to_i2', 'i5_to_i2', 'i1_to_i2'),
    'tls_3': ('i1_to_i3', 'south_to_i3', 'i4_to_i3', 'west_to_i3'),
    'tls_4': ('i2_to_i4', 'south_to_i4', 'i7_to_i4', 'i3_to_i4'),
    'tls_5': ('north_to_i5', 'i7_to_i5', 'i6_to_i5', 'i2_to_i5'),
    'tls_6': ('north_to_i6', 'i8_to_i6', 'east_to_i6', 'i5_to_i6'),
    'tls_7': ('i5_to_i7', 'south_to_i7', 'i8_to_i7', 'i4_to_i7'),
    'tls_8': ('i6_to_i8', 'south_to_i8', 'east_to_i8', 'i7_to_i8'),
}
EDGES_4 = {k: tuple(v) for k, v in list(EDGES_8.items())[:4]}
EDGES_4['tls_2'] = ('north_to_i2', 'i4_to_i2', 'east_to_i2', 'i1_to_i2')
EDGES_4['tls_4'] = ('i2_to_i4', 'south_to_i4', 'east_to_i4', 'i3_to_i4')
NEIGHBORS = {'tls_1': ['tls_2', 'tls_3'], 'tls_2': ['tls_1', 'tls_4'],
             'tls_3': ['tls_1', 'tls_4'], 'tls_4': ['tls_2', 'tls_3'],
             'tls_5': ['tls_6', 'tls_7'], 'tls_6': ['tls_5', 'tls_8'],
             'tls_7': ['tls_5', 'tls_8'], 'tls_8': ['tls_6', 'tls_7']}


class SignalScheduler:
    """Keep/switch with 0/2 greens and 1/3 yellows; time is SUMO time."""
    def __init__(self, conn, tls_ids, min_green=5, yellow=3):
        self.conn, self.min_green, self.yellow = conn, min_green, yellow
        self.logical = dict.fromkeys(tls_ids, 0)
        self.green_since = dict.fromkeys(tls_ids, 0.)
        self.yellow_until = dict.fromkeys(tls_ids, None)
        for tls in tls_ids:
            conn.trafficlight.setPhase(tls, 0)
            conn.trafficlight.setPhaseDuration(tls, 1e9)

    def elapsed(self, tls, now):
        return 0. if self.yellow_until[tls] is not None else now - self.green_since[tls]

    def request(self, tls, action, now):
        if action not in (0, 1):
            raise ValueError('Actions must be 0 (hold) or 1 (switch)')
        if not action:
            return False
        if self.yellow_until[tls] is not None or self.elapsed(tls, now) < self.min_green:
            return True
        self.conn.trafficlight.setPhase(tls, 1 if self.logical[tls] == 0 else 3)
        self.conn.trafficlight.setPhaseDuration(tls, self.yellow)
        self.logical[tls] = 1 - self.logical[tls]
        self.yellow_until[tls] = now + self.yellow
        return False

    def advance(self, now):
        for tls, until in self.yellow_until.items():
            if until is not None and now >= until:
                self.conn.trafficlight.setPhase(tls, 2 * self.logical[tls])
                self.conn.trafficlight.setPhaseDuration(tls, 1e9)
                self.green_since[tls] = until
                self.yellow_until[tls] = None


class EpisodeMetrics:
    def __init__(self, scheduled, tls_ids):
        self.scheduled = scheduled
        self.departed, self.arrived = set(), {}
        self.removed = set()
        self.queue_integrals = dict.fromkeys(tls_ids, 0.)
        self.backlogs = []
        self.collisions = 0
        self.teleports = 0
        self.result = None

    def tick(self, conn, queues):
        now = float(conn.simulation.getTime())
        self.departed.update(conn.simulation.getDepartedIDList())
        for vid in conn.simulation.getArrivedIDList():
            self.arrived[vid] = now
        active = set(conn.vehicle.getIDList())
        self.removed.update(self.departed - set(self.arrived) - active)
        # Teleporting vehicles may temporarily disappear but later reappear.
        self.removed.difference_update(active | set(self.arrived))
        self.collisions += conn.simulation.getCollidingVehiclesNumber()
        self.teleports += conn.simulation.getStartingTeleportNumber()
        for tls, queue in queues.items():
            self.queue_integrals[tls] += queue
        offered_now = sum(t < now for t in self.scheduled.values())
        self.backlogs.append(max(0, offered_now - len(self.departed)))

    def finish(self, horizon, tripinfo):
        if self.result is not None:
            return self.result
        trips = ET.parse(tripinfo).getroot().findall('tripinfo')
        complete, incomplete = [], []
        for trip in trips:
            attrs = trip.attrib
            vid = attrs['id']
            self.departed.add(vid)
            arrival = float(attrs.get('arrival', '-1'))
            if arrival >= 0:
                self.arrived[vid] = arrival
                complete.append(float(attrs['waitingTime']))
            elif attrs.get('vaporized', ''):
                self.removed.add(vid)
            else:
                incomplete.append(float(attrs['waitingTime']))
        scheduled = {v: t for v, t in self.scheduled.items() if t < horizon}
        offered = len(scheduled)
        pending = len(set(scheduled) - self.departed)
        arrived = len(set(scheduled) & set(self.arrived))
        departed = len(set(scheduled) & self.departed)
        removed = len(self.removed & set(scheduled) - set(self.arrived))
        unfinished = departed - arrived - removed
        # Removed vehicles receive the remaining observation horizon as cost;
        # disappearing traffic must never improve an algorithm's score.
        system_time = sum(max(0., min(horizon, self.arrived.get(v, horizon)) - t)
                          for v, t in scheduled.items())
        means = {tls: q / horizon if horizon else None for tls, q in self.queue_integrals.items()}
        n = len(self.backlogs)
        q3 = self.backlogs[n // 2:3 * n // 4]
        q4 = self.backlogs[3 * n // 4:]
        self.result = {
            'schema_version': SCHEMA_VERSION, 'duration_s': horizon,
            'offered': offered, 'departed': departed, 'arrived': arrived,
            'unfinished': unfinished, 'waiting_to_enter': pending, 'removed': removed,
            'throughput_vehicles': arrived,
            'throughput_vehicles_per_hour': arrived * 3600 / horizon if horizon else None,
            'completion_rate': arrived / offered if offered else None,
            'total_system_time_s': system_time,
            'system_time_per_offered_vehicle_s': system_time / offered if offered else None,
            'completed_trip_wait_mean_s': float(np.mean(complete)) if complete else None,
            'unfinished_wait_mean_s': float(np.mean(incomplete)) if incomplete else None,
            'mean_queue_by_intersection': means,
            'mean_queue_per_intersection': float(np.mean(list(means.values()))) if horizon else None,
            'final_insertion_backlog_fraction': pending / offered if offered else None,
            'backlog_q3_mean': float(np.mean(q3)) if q3 else None,
            'backlog_q4_mean': float(np.mean(q4)) if q4 else None,
            'collisions': self.collisions, 'teleports': self.teleports,
            'accounting_valid': offered == pending + departed and departed == arrived + unfinished + removed,
        }
        return self.result


class ControlledEnvironment:
    """Common implementation; wrappers only select topology and cooperation."""
    def __init__(self, net_file, route_file, use_gui=False, num_seconds=3600,
                 delta_time=5, topology='single', cooperative=False, seed=42,
                 output_dir=None):
        self.net_file, self.route_file = str(Path(net_file).resolve()), str(Path(route_file).resolve())
        self.use_gui, self.num_seconds, self.delta_time = use_gui, int(num_seconds), int(delta_time)
        if self.num_seconds <= 0 or self.delta_time <= 0:
            raise ValueError('Duration and decision interval must be positive integers')
        self.topology, self.cooperative, self.seed = topology, cooperative, seed
        self.single = topology == 'single'
        tuples = {'center': ('north_in', 'south_in', 'east_in', 'west_in')} if self.single else (EDGES_4 if topology == 'four' else EDGES_8)
        self._edges = {tls: dict(zip(DIRECTIONS, edges)) for tls, edges in tuples.items()}
        self.tls_ids = list(self._edges)
        self.tls_id = self.tls_ids[0]
        self.edges = self._edges[self.tls_id] if self.single else self._edges
        self.neighbors = NEIGHBORS
        self.group_a, self.group_b = self.tls_ids[:4], self.tls_ids[4:]
        self.groups = {'group_a': self.group_a}
        if self.group_b:
            self.groups['group_b'] = self.group_b
        self.tls_group = {tls: g for g, members in self.groups.items() for tls in members}
        self.phases = {0: 0, 1: 2}
        self.output_dir = Path(output_dir) if output_dir else None
        self.sumo_running = False
        self.simulation_step = 0
        self.episode_metrics = None
        self._temp = None

    def reset(self, seed=None, route_file=None):
        self.close()
        if self._temp is not None:
            self._temp.cleanup()
        if seed is not None:
            self.seed = int(seed)
        if route_file is not None:
            self.route_file = str(Path(route_file).resolve())
        self._temp = tempfile.TemporaryDirectory(prefix='traffic_rl_')
        work = Path(self._temp.name)
        route_root = ET.parse(self.route_file).getroot()
        schedule_file = self.route_file
        if route_root.findall('flow'):
            schedule_file = work / 'vehicles.rou.xml'
            materialize_routes(self.net_file, self.route_file, schedule_file, self.seed, horizon=self.num_seconds)
        else:
            # Validation is shared even for explicit schedules.
            from traffic_demand import route_definitions
            root, edges_for = route_definitions(self.net_file, schedule_file)
            for item in root.findall('vehicle'):
                edges_for(item)
        self.scheduled = scheduled_departures(schedule_file, self.num_seconds)
        self.tripinfo = work / 'tripinfo.xml'
        self.sumo_cmd = ['sumo-gui' if self.use_gui else 'sumo', '--net-file', self.net_file,
                         '--route-files', str(schedule_file), '--step-length', '1', '--begin', '0',
                         '--seed', str(self.seed), '--no-step-log', 'true', '--duration-log.disable', 'true',
                         '--waiting-time-memory', '10000', '--time-to-teleport', '-1',
                         '--tripinfo-output', str(self.tripinfo), '--tripinfo-output.write-unfinished', 'true']
        traci.start(self.sumo_cmd, verbose=False)
        self.conn = traci.getConnection('default')
        self.sumo_running = True
        self.scheduler = SignalScheduler(self.conn, self.tls_ids)
        self.metrics = EpisodeMetrics(self.scheduled, self.tls_ids)
        self.episode_metrics = None
        self.simulation_step = 0
        self.invalid_switch = dict.fromkeys(self.tls_ids, False)
        self.total_waiting_time = 0. if self.single else dict.fromkeys(self.tls_ids, 0.)
        self.vehicles_passed = 0
        self._sync_timers()
        return self._get_state() if self.single else self._get_all_states()

    def start_sumo(self):
        return self.reset()

    def _sync_timers(self):
        self.current_phases = dict(self.scheduler.logical)
        timers = {tls: self.scheduler.elapsed(tls, self.simulation_step) for tls in self.tls_ids}
        self.current_phase = self.current_phases[self.tls_id]
        self.time_since_last_change = timers[self.tls_id] if self.single else timers

    def _queues(self):
        return {tls: sum(self.conn.edge.getLastStepHaltingNumber(e) for e in edges.values())
                for tls, edges in self._edges.items()}

    def _wait(self, tls):
        return sum(self.conn.vehicle.getWaitingTime(v) for e in self._edges[tls].values()
                   for v in self.conn.edge.getLastStepVehicleIDs(e))

    def _get_state(self, tls_id=None):
        tls = tls_id or self.tls_id
        state = [min(20, self.conn.edge.getLastStepHaltingNumber(e)) for e in self._edges[tls].values()]
        state += [self.scheduler.logical[tls], min(60, self.scheduler.elapsed(tls, self.simulation_step))]
        if self.cooperative:
            queues = self._queues()
            state += [min(40, queues[n]) for n in self.neighbors[tls]]
        return np.asarray(state, dtype=np.float32)

    _get_local_state = _get_state

    def _get_all_states(self):
        return {tls: self._get_state(tls) for tls in self.tls_ids}

    _get_all_local_states = _get_all_states

    def step(self, actions):
        if not self.sumo_running or self.simulation_step >= self.num_seconds:
            raise RuntimeError('Call reset before stepping a completed environment')
        actions = {self.tls_id: int(actions)} if self.single else actions
        if set(actions) != set(self.tls_ids):
            raise ValueError('An action for every intersection is required')
        self.invalid_switch = {tls: self.scheduler.request(tls, a, self.simulation_step) for tls, a in actions.items()}
        for _ in range(min(self.delta_time, self.num_seconds - self.simulation_step)):
            self.conn.simulationStep()
            self.simulation_step += 1
            self.scheduler.advance(self.simulation_step)
            self.metrics.tick(self.conn, self._queues())
        self._sync_timers()
        states = self._get_all_states()
        queues = self._queues()
        individual = {}
        for tls in self.tls_ids:
            waiting = self._wait(tls)
            individual[tls] = -queues[tls] - .5 * waiting - 10 * self.invalid_switch[tls]
            if self.single:
                self.total_waiting_time += waiting
            else:
                self.total_waiting_time[tls] += waiting
        rewards = dict(individual)
        if self.cooperative:
            for members in self.groups.values():
                mean = float(np.mean([individual[tls] for tls in members]))
                rewards.update(dict.fromkeys(members, mean))
        self.vehicles_passed = len(self.metrics.arrived)
        done = self.simulation_step >= self.num_seconds
        info = self._get_metrics()
        info['individual_rewards'] = individual
        info['invalid_switches'] = dict(self.invalid_switch)
        if done:
            # States and reward are captured before closing the connection.
            info['episode_metrics'] = self.finish_episode()
        return (states[self.tls_id], rewards[self.tls_id], done, info) if self.single else (states, rewards, done, info)

    def _get_metrics(self):
        ids = self.conn.vehicle.getIDList()
        wait = sum(self.conn.vehicle.getWaitingTime(v) for v in ids)
        queues = self._queues()
        metrics = {'total_vehicles': len(ids), 'network_waiting_time': wait,
                   'total_waiting_time': wait, 'avg_waiting_time': wait / len(ids) if ids else None,
                   'total_queue': sum(queues.values()), 'vehicles_passed': len(self.metrics.arrived),
                   'throughput_vehicles': len(self.metrics.arrived),
                   'average_speed': float(np.mean([self.conn.vehicle.getSpeed(v) for v in ids])) if ids else None,
                   'per_intersection': {tls: {'queue': queues[tls], 'phase': self.current_phases[tls],
                         'time_since_change': self.scheduler.elapsed(tls, self.simulation_step)} for tls in self.tls_ids},
                   'per_group': {g: {'total_queue': sum(queues[tls] for tls in members),
                                    'avg_queue': float(np.mean([queues[tls] for tls in members]))}
                                 for g, members in self.groups.items()}}
        return metrics

    def get_traffic_metrics(self):
        return self.episode_metrics if not self.sumo_running else self._get_metrics()

    def finish_episode(self):
        if self.episode_metrics is not None:
            return self.episode_metrics
        if not self.sumo_running:
            return None
        self.conn.close()
        self.sumo_running = False
        self.episode_metrics = self.metrics.finish(self.simulation_step, self.tripinfo)
        return self.episode_metrics

    def close(self):
        if self.sumo_running:
            self.finish_episode()

    def get_state_dim(self):
        return 7 if self.topology == 'supervisor' else (8 if self.cooperative else 6)

    def get_local_state_dim(self):
        return 6

    def get_action_dim(self):
        return 2

    def build_enhanced_states(self, local_states, signals_a, signals_b):
        signals = {**signals_a, **signals_b}
        return {tls: np.append(local_states[tls], signals[tls]).astype(np.float32) for tls in self.tls_ids}

    def get_group_summary(self, local_states_dict, group_tls_ids, boundary_tls_ids):
        queues = [float(local_states_dict[tls][:4].sum()) for tls in group_tls_ids]
        ids = [v for tls in group_tls_ids for e in self._edges[tls].values()
               for v in self.conn.edge.getLastStepVehicleIDs(e)] if self.sumo_running else []
        avg_wait = sum(self.conn.vehicle.getWaitingTime(v) for v in ids) / len(ids) if ids else 0.
        return np.array([np.mean(queues), max(queues), avg_wait,
                         sum(local_states_dict[tls][:4].sum() for tls in boundary_tls_ids)], dtype=np.float32)

    def get_group_avg_reward(self, rewards, group):
        return float(np.mean([rewards[tls] for tls in group]))
