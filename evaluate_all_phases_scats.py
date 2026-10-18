"""
Master 7-Phase SCATS Real-World Evaluation Benchmark
======================================================
Evaluates Victoria SCATS empirical demand across all 7 architectural phases:
  1. Webster Fixed-Time Baseline (1 Intersection)
  2. Phase 0: Single Agent DDQN (1 Intersection)
  3. Phase 1: 4-Intersection Independent DDQN
  4. Phase 1: 4-Intersection Cooperative DDQN
  5. Phase 2: 8-Intersection Independent DDQN
  6. Phase 2: 8-Intersection Supervisor (No Communication)
  7. Phase 2: 8-Intersection Global Supervisor (With Communication)
"""

import os
import sys
import codecs
sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')

import numpy as np
import torch
import traci

# Environments
from sumo_environment import SumoEnvironment
from sumo_environment_multiagent import MultiAgentSumoEnvironment
from sumo_environment_8intersection import EightIntersectionEnv
from sumo_environment_supervisor import SupervisorSumoEnvironment

# Agents
from agent import DDQNAgent
from supervisor_agent import SupervisorAgent

def evaluate_episode_loop(env, agents=None, env_type="single", coop_type="none", supervisors=None):
    """Generic evaluation step loop for one episode"""
    state = env.reset()
    done = False
    step = 0
    episode_arrived = 0
    
    num_intersections = 1 if env_type == "single" else (4 if env_type == "multiagent" else 8)
    step_queue_list = []
    step_wait_list = []
    
    while not done:
        if env_type == "single":
            if agents is None:
                # Fixed time 30s
                action = 1 if step % (30 // env.delta_time) == 0 else 0
            else:
                action = agents.select_action(state, training=False)
            next_state, reward, done, info = env.step(action)
            state = next_state
            
        elif env_type == "multiagent":
            actions = {tls: agents[tls].select_action(state[tls], training=False) for tls in env.tls_ids}
            next_state, reward, done, info = env.step(actions)
            state = next_state
            
        elif env_type == "8inter":
            actions = {tls: agents[tls].select_action(state[tls], training=False) for tls in env.tls_ids}
            next_state, reward, done, info = env.step(actions)
            state = next_state
            
        elif env_type == "supervisor":
            sup_a, sup_b = supervisors
            if coop_type == "no_comm":
                sig_a = sup_a.get_signals({t: state[t] for t in env.group_a})
                sig_b = sup_b.get_signals({t: state[t] for t in env.group_b})
            else: # with comm
                sum_a = env.get_group_summary(state, env.group_a, boundary_tls_ids=['tls_2', 'tls_4'])
                sum_b = env.get_group_summary(state, env.group_b, boundary_tls_ids=['tls_5', 'tls_7'])
                sig_a = sup_a.get_signals_with_global({t: state[t] for t in env.group_a}, other_summary=sum_b)
                sig_b = sup_b.get_signals_with_global({t: state[t] for t in env.group_b}, other_summary=sum_a)
                
            enhanced = env.build_enhanced_states(state, sig_a, sig_b)
            actions = {tls: agents[tls].select_action(enhanced[tls], training=False) for tls in env.tls_ids}
            next_state, reward, done, info = env.step(actions)
            state = next_state
            
        step += 1
        try:
            episode_arrived += traci.simulation.getArrivedNumber()
        except:
            pass
            
        # Accumulate true episode-wide metrics at every step
        vehicle_ids = traci.vehicle.getIDList()
        if len(vehicle_ids) > 0:
            step_wait_list.append(sum(traci.vehicle.getWaitingTime(v) for v in vehicle_ids) / len(vehicle_ids))
        else:
            step_wait_list.append(0.0)
            
        total_queue = 0
        if env_type == "single":
            for edge in env.edges.values():
                total_queue += traci.edge.getLastStepHaltingNumber(edge)
        else:
            for tls in env.tls_ids:
                for d in ['north', 'south', 'east', 'west']:
                    try:
                        total_queue += traci.edge.getLastStepHaltingNumber(env.edges[tls][d])
                    except:
                        pass
        step_queue_list.append(total_queue / num_intersections)
            
    return {
        'active_cars': len(traci.vehicle.getIDList()),
        'completed_trips': episode_arrived,
        'queue_per_inter': float(np.mean(step_queue_list)),
        'avg_wait_time': float(np.mean(step_wait_list))
    }


def run_benchmark_for_config(name, env, agents, env_type="single", coop_type="none", supervisors=None, num_episodes=3):
    print(f"\n🚀 Running Benchmark: [{name}] ({num_episodes} episodes)...")
    results = {'active_cars': [], 'completed_trips': [], 'queue_per_inter': [], 'avg_wait_time': []}
    
    for ep in range(num_episodes):
        ep_res = evaluate_episode_loop(env, agents, env_type, coop_type, supervisors)
        for k in results:
            results[k].append(ep_res[k])
        print(f"   Ep {ep+1}: Wait Time = {ep_res['avg_wait_time']:.2f}s | Queue/Inter = {ep_res['queue_per_inter']:.2f} | Completed = {ep_res['completed_trips']}")
        
    summary = {}
    for k, vals in results.items():
        summary[k] = (np.mean(vals), np.std(vals))
    return summary


def main():
    print("="*80)
    print(" VICTORIA SCATS REAL-WORLD DATASET MASTER 7-PHASE BENCHMARK")
    print("="*80)
    
    all_results = {}
    num_eps = 3
    
    # ---------------------------------------------------------
    # 1. & 2. Single Intersection (Fixed Time & Phase 0 DDQN)
    # ---------------------------------------------------------
    env_single = SumoEnvironment(
        net_file='sumo_files/intersection.net.xml',
        route_file='sumo_files/routes_scats_empirical.rou.xml',
        use_gui=False, num_seconds=3600, delta_time=5
    )
    all_results['1. Fixed-Time Baseline'] = run_benchmark_for_config(
        "1. Fixed-Time Baseline", env_single, None, "single", num_episodes=num_eps
    )
    
    agent_single = DDQNAgent(state_dim=6, action_dim=2, hidden_dim=128)
    if os.path.exists('models/ddqn_traffic_final.pth'):
        agent_single.load('models/ddqn_traffic_final.pth')
    all_results['2. Single Agent DDQN'] = run_benchmark_for_config(
        "2. Single Agent DDQN", env_single, agent_single, "single", num_episodes=num_eps
    )
    env_single.close()
    
    # ---------------------------------------------------------
    # 3. & 4. 4-Intersection Grid (Independent & Cooperative)
    # ---------------------------------------------------------
    env_4ind = MultiAgentSumoEnvironment(
        net_file='sumo_files_multiagent/multiagent.net.xml',
        route_file='sumo_files_multiagent/routes_scats_empirical.rou.xml',
        use_gui=False, num_seconds=3600, delta_time=5, cooperative=False
    )
    agents_4ind = {}
    for i, tls in enumerate(env_4ind.tls_ids, 1):
        ag = DDQNAgent(state_dim=6, action_dim=2, hidden_dim=128)
        ckpt = f'checkpoints_multiagent/tls_{i}_final.pth'
        if os.path.exists(ckpt): ag.load(ckpt)
        agents_4ind[tls] = ag
    all_results['3. 4-Inter Independent DDQN'] = run_benchmark_for_config(
        "3. 4-Inter Independent DDQN", env_4ind, agents_4ind, "multiagent", num_episodes=num_eps
    )
    env_4ind.close()
    
    env_4coop = MultiAgentSumoEnvironment(
        net_file='sumo_files_multiagent/multiagent.net.xml',
        route_file='sumo_files_multiagent/routes_scats_empirical.rou.xml',
        use_gui=False, num_seconds=3600, delta_time=5, cooperative=True
    )
    agents_4coop = {}
    for i, tls in enumerate(env_4coop.tls_ids, 1):
        ag = DDQNAgent(state_dim=8, action_dim=2, hidden_dim=128) # 8-dim for cooperative
        ckpt = f'checkpoints_cooperative/tls_{i}_final.pth'
        if os.path.exists(ckpt): ag.load(ckpt)
        agents_4coop[tls] = ag
    all_results['4. 4-Inter Cooperative DDQN'] = run_benchmark_for_config(
        "4. 4-Inter Cooperative DDQN", env_4coop, agents_4coop, "multiagent", num_episodes=num_eps
    )
    env_4coop.close()
    
    # ---------------------------------------------------------
    # 5. 8-Intersection Independent DDQN
    # ---------------------------------------------------------
    env_8ind = EightIntersectionEnv(
        use_gui=False, num_seconds=3600, delta_time=5
    )
    # Override route file to SCATS empirical
    env_8ind.route_file = 'sumo_files_8intersection/routes_scats_empirical.rou.xml'
    agents_8ind = {}
    for i, tls in enumerate(env_8ind.tls_ids, 1):
        ag = DDQNAgent(state_dim=8, action_dim=2, hidden_dim=128)
        ckpt = f'checkpoints_8intersection/tls_{i}_final.pth'
        if os.path.exists(ckpt): ag.load(ckpt)
        agents_8ind[tls] = ag
    all_results['5. 8-Inter Independent DDQN'] = run_benchmark_for_config(
        "5. 8-Inter Independent DDQN", env_8ind, agents_8ind, "8inter", num_episodes=num_eps
    )
    env_8ind.close()
    
    # ---------------------------------------------------------
    # 6. 8-Intersection Supervisor (No Communication)
    # ---------------------------------------------------------
    env_sup = SupervisorSumoEnvironment(
        use_gui=False, num_seconds=3600, delta_time=5
    )
    env_sup.route_file = 'sumo_files_8intersection/routes_scats_empirical.rou.xml'
    agents_sup = {}
    for tls in env_sup.tls_ids:
        ag = DDQNAgent(state_dim=7, action_dim=2, hidden_dim=128)
        ckpt = f'checkpoints_supervisor/agents/{tls}_final.pth'
        if os.path.exists(ckpt): ag.load(ckpt)
        agents_sup[tls] = ag
    sup_a_nocomm = SupervisorAgent(env_sup.group_a, 6, 64, global_summary_dim=0)
    sup_b_nocomm = SupervisorAgent(env_sup.group_b, 6, 64, global_summary_dim=0)
    if os.path.exists('checkpoints_supervisor/supervisors/supervisor_a_final.pth'):
        sup_a_nocomm.load('checkpoints_supervisor/supervisors/supervisor_a_final.pth')
    if os.path.exists('checkpoints_supervisor/supervisors/supervisor_b_final.pth'):
        sup_b_nocomm.load('checkpoints_supervisor/supervisors/supervisor_b_final.pth')
    all_results['6. 8-Inter Supervisor (No Comm)'] = run_benchmark_for_config(
        "6. 8-Inter Supervisor (No Comm)", env_sup, agents_sup, "supervisor", "no_comm", (sup_a_nocomm, sup_b_nocomm), num_episodes=num_eps
    )
    env_sup.close()
    
    # ---------------------------------------------------------
    # 7. 8-Intersection Global Supervisor (With Communication)
    # ---------------------------------------------------------
    env_gsup = SupervisorSumoEnvironment(
        use_gui=False, num_seconds=3600, delta_time=5
    )
    env_gsup.route_file = 'sumo_files_8intersection/routes_scats_empirical.rou.xml'
    agents_gsup = {}
    for tls in env_gsup.tls_ids:
        ag = DDQNAgent(state_dim=7, action_dim=2, hidden_dim=128)
        ckpt = f'checkpoints_global_supervisor/agents/{tls}_final.pth'
        if os.path.exists(ckpt): ag.load(ckpt)
        agents_gsup[tls] = ag
    sup_a_comm = SupervisorAgent(env_gsup.group_a, 6, 64, global_summary_dim=4)
    sup_b_comm = SupervisorAgent(env_gsup.group_b, 6, 64, global_summary_dim=4)
    if os.path.exists('checkpoints_global_supervisor/supervisors/supervisor_a_final.pth'):
        sup_a_comm.load('checkpoints_global_supervisor/supervisors/supervisor_a_final.pth')
    if os.path.exists('checkpoints_global_supervisor/supervisors/supervisor_b_final.pth'):
        sup_b_comm.load('checkpoints_global_supervisor/supervisors/supervisor_b_final.pth')
    all_results['7. 8-Inter Supervisor (With Comm)'] = run_benchmark_for_config(
        "7. 8-Inter Supervisor (With Comm)", env_gsup, agents_gsup, "supervisor", "comm", (sup_a_comm, sup_b_comm), num_episodes=num_eps
    )
    env_gsup.close()
    
    # ---------------------------------------------------------
    # Print Final Consolidated Table
    # ---------------------------------------------------------
    print("\n" + "="*90)
    print(" 🏆 VICTORIA SCATS REAL-WORLD BENCHMARK ACROSS ALL 7 PHASES (AVG ± STD)")
    print("="*90)
    print(f"{'Phase / Architecture':<33} | {'Avg Wait Time (s)':<18} | {'Queue / Inter':<14} | {'Throughput (Trips)':<18}")
    print("-" * 90)
    for name, res in all_results.items():
        wait_m, wait_s = res['avg_wait_time']
        q_m, q_s = res['queue_per_inter']
        t_m, t_s = res['completed_trips']
        print(f"{name:<33} | {wait_m:>7.2f} ± {wait_s:<6.2f} | {q_m:>5.2f} ± {q_s:<4.2f} | {t_m:>7.2f} ± {t_s:<6.2f}")
    print("="*90)
    print("\n✅ Step 2 (Master Benchmark Execution) Completed Successfully!")

if __name__ == '__main__':
    main()
