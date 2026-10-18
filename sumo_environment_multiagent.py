"""Four topology using the shared corrected SUMO runtime."""
from traffic_common import ControlledEnvironment


class MultiAgentSumoEnvironment(ControlledEnvironment):
    def __init__(self, net_file='sumo_files_multiagent/multiagent.net.xml', route_file='sumo_files_multiagent/multiagent.rou.xml', use_gui=False,
                 num_seconds=3600, delta_time=5, cooperative=False, **kwargs):
        super().__init__(net_file, route_file, use_gui, num_seconds, delta_time,
                         topology='four', cooperative=cooperative, **kwargs)
