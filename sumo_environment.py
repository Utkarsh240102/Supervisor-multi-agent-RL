"""Single topology using the shared corrected SUMO runtime."""
from traffic_common import ControlledEnvironment


class SumoEnvironment(ControlledEnvironment):
    def __init__(self, net_file='sumo_files/intersection.net.xml', route_file='sumo_files/routes.rou.xml', use_gui=False,
                 num_seconds=3600, delta_time=5, **kwargs):
        super().__init__(net_file, route_file, use_gui, num_seconds, delta_time,
                         topology='single', cooperative=False, **kwargs)
