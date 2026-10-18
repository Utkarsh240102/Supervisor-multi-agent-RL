"""
Master SCATS Demand Generator for All Grid Sizes (1, 4, and 8 Intersections)
=============================================================================
Generates calibrated empirical Victoria SCATS rush-hour demand curves across
all three project topologies:
  1. Single Intersection (sumo_files/routes_scats_empirical.rou.xml)
  2. 4-Intersection Grid (sumo_files_multiagent/routes_scats_empirical.rou.xml)
  3. 8-Intersection Grid (sumo_files_8intersection/routes_scats_empirical.rou.xml)
"""

import os
import sys
import codecs
sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')

def generate_1_intersection_scats():
    routes_xml = """<?xml version="1.0" encoding="UTF-8"?>
<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">
    <vType id="car" accel="2.6" decel="4.5" sigma="0.5" length="5.0" minGap="2.5" maxSpeed="50.0" guiShape="passenger"/>

    <!-- Tier 1: 0-15 mins (Pre-Peak ~360 veh/hr) -->
    <flow id="ns_t1" type="car" begin="0" end="900" probability="0.08" from="north_in" to="south_out"/>
    <flow id="sn_t1" type="car" begin="0" end="900" probability="0.08" from="south_in" to="north_out"/>
    <flow id="ew_t1" type="car" begin="0" end="900" probability="0.07" from="east_in" to="west_out"/>
    <flow id="we_t1" type="car" begin="0" end="900" probability="0.07" from="west_in" to="east_out"/>

    <!-- Tier 2: 15-30 mins (Rising Peak ~720 veh/hr) -->
    <flow id="ns_t2" type="car" begin="900" end="1800" probability="0.18" from="north_in" to="south_out"/>
    <flow id="sn_t2" type="car" begin="900" end="1800" probability="0.18" from="south_in" to="north_out"/>
    <flow id="ew_t2" type="car" begin="900" end="1800" probability="0.15" from="east_in" to="west_out"/>
    <flow id="we_t2" type="car" begin="900" end="1800" probability="0.15" from="west_in" to="east_out"/>

    <!-- Tier 3: 30-45 mins (Peak Bottleneck ~1260 veh/hr) -->
    <flow id="ns_t3" type="car" begin="1800" end="2700" probability="0.32" from="north_in" to="south_out"/>
    <flow id="sn_t3" type="car" begin="1800" end="2700" probability="0.32" from="south_in" to="north_out"/>
    <flow id="ew_t3" type="car" begin="1800" end="2700" probability="0.28" from="east_in" to="west_out"/>
    <flow id="we_t3" type="car" begin="1800" end="2700" probability="0.28" from="west_in" to="east_out"/>

    <!-- Tier 4: 45-60 mins (Post-Peak Taper ~540 veh/hr) -->
    <flow id="ns_t4" type="car" begin="2700" end="3600" probability="0.14" from="north_in" to="south_out"/>
    <flow id="sn_t4" type="car" begin="2700" end="3600" probability="0.14" from="south_in" to="north_out"/>
    <flow id="ew_t4" type="car" begin="2700" end="3600" probability="0.12" from="east_in" to="west_out"/>
    <flow id="we_t4" type="car" begin="2700" end="3600" probability="0.12" from="west_in" to="east_out"/>
</routes>
"""
    os.makedirs('sumo_files', exist_ok=True)
    with open('sumo_files/routes_scats_empirical.rou.xml', 'w') as f:
        f.write(routes_xml)
    print("✅ [1-Intersection] Generated: sumo_files/routes_scats_empirical.rou.xml")


def generate_4_intersection_scats():
    routes_xml = """<?xml version="1.0" encoding="UTF-8"?>
<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">
    <vType id="car" accel="2.6" decel="4.5" sigma="0.5" length="5" minGap="2.5" maxSpeed="50" guiShape="passenger"/>

    <!-- Routes -->
    <route id="n1_to_s1" edges="north_to_i1 i1_to_i3 i3_to_south"/>
    <route id="n2_to_s2" edges="north_to_i2 i2_to_i4 i4_to_south"/>
    <route id="s1_to_n1" edges="south_to_i3 i3_to_i1 i1_to_north"/>
    <route id="s2_to_n2" edges="south_to_i4 i4_to_i2 i2_to_north"/>
    <route id="e1_to_w1" edges="east_to_i2 i2_to_i1 i1_to_west"/>
    <route id="e2_to_w2" edges="east_to_i4 i4_to_i3 i3_to_west"/>
    <route id="w1_to_e1" edges="west_to_i1 i1_to_i2 i2_to_east"/>
    <route id="w2_to_e2" edges="west_to_i3 i3_to_i4 i4_to_east"/>
    <route id="nw_to_se" edges="north_to_i1 i1_to_i2 i2_to_i4 i4_to_south"/>
    <route id="ne_to_sw" edges="north_to_i2 i2_to_i1 i1_to_i3 i3_to_south"/>

    <!-- Tier 1: 0-900s (Pre-Peak) -->
    <flow id="f_n1_s1_t1" type="car" route="n1_to_s1" begin="0" end="900" probability="0.03"/>
    <flow id="f_n2_s2_t1" type="car" route="n2_to_s2" begin="0" end="900" probability="0.03"/>
    <flow id="f_s1_n1_t1" type="car" route="s1_to_n1" begin="0" end="900" probability="0.03"/>
    <flow id="f_s2_n2_t1" type="car" route="s2_to_n2" begin="0" end="900" probability="0.03"/>
    <flow id="f_e1_w1_t1" type="car" route="e1_to_w1" begin="0" end="900" probability="0.03"/>
    <flow id="f_e2_w2_t1" type="car" route="e2_to_w2" begin="0" end="900" probability="0.03"/>
    <flow id="f_w1_e1_t1" type="car" route="w1_to_e1" begin="0" end="900" probability="0.03"/>
    <flow id="f_w2_e2_t1" type="car" route="w2_to_e2" begin="0" end="900" probability="0.03"/>

    <!-- Tier 2: 900-1800s (Rising Peak) -->
    <flow id="f_n1_s1_t2" type="car" route="n1_to_s1" begin="900" end="1800" probability="0.06"/>
    <flow id="f_n2_s2_t2" type="car" route="n2_to_s2" begin="900" end="1800" probability="0.06"/>
    <flow id="f_s1_n1_t2" type="car" route="s1_to_n1" begin="900" end="1800" probability="0.06"/>
    <flow id="f_s2_n2_t2" type="car" route="s2_to_n2" begin="900" end="1800" probability="0.06"/>
    <flow id="f_e1_w1_t2" type="car" route="e1_to_w1" begin="900" end="1800" probability="0.06"/>
    <flow id="f_e2_w2_t2" type="car" route="e2_to_w2" begin="900" end="1800" probability="0.06"/>
    <flow id="f_w1_e1_t2" type="car" route="w1_to_e1" begin="900" end="1800" probability="0.06"/>
    <flow id="f_w2_e2_t2" type="car" route="w2_to_e2" begin="900" end="1800" probability="0.06"/>

    <!-- Tier 3: 1800-2700s (Peak SCATS Bottleneck) -->
    <flow id="f_n1_s1_t3" type="car" route="n1_to_s1" begin="1800" end="2700" probability="0.11"/>
    <flow id="f_n2_s2_t3" type="car" route="n2_to_s2" begin="1800" end="2700" probability="0.11"/>
    <flow id="f_s1_n1_t3" type="car" route="s1_to_n1" begin="1800" end="2700" probability="0.11"/>
    <flow id="f_s2_n2_t3" type="car" route="s2_to_n2" begin="1800" end="2700" probability="0.11"/>
    <flow id="f_e1_w1_t3" type="car" route="e1_to_w1" begin="1800" end="2700" probability="0.11"/>
    <flow id="f_e2_w2_t3" type="car" route="e2_to_w2" begin="1800" end="2700" probability="0.11"/>
    <flow id="f_w1_e1_t3" type="car" route="w1_to_e1" begin="1800" end="2700" probability="0.11"/>
    <flow id="f_w2_e2_t3" type="car" route="w2_to_e2" begin="1800" end="2700" probability="0.11"/>

    <!-- Tier 4: 2700-3600s (Post-Peak Taper) -->
    <flow id="f_n1_s1_t4" type="car" route="n1_to_s1" begin="2700" end="3600" probability="0.04"/>
    <flow id="f_n2_s2_t4" type="car" route="n2_to_s2" begin="2700" end="3600" probability="0.04"/>
    <flow id="f_s1_n1_t4" type="car" route="s1_to_n1" begin="2700" end="3600" probability="0.04"/>
    <flow id="f_s2_n2_t4" type="car" route="s2_to_n2" begin="2700" end="3600" probability="0.04"/>
    <flow id="f_e1_w1_t4" type="car" route="e1_to_w1" begin="2700" end="3600" probability="0.04"/>
    <flow id="f_e2_w2_t4" type="car" route="e2_to_w2" begin="2700" end="3600" probability="0.04"/>
    <flow id="f_w1_e1_t4" type="car" route="w1_to_e1" begin="2700" end="3600" probability="0.04"/>
    <flow id="f_w2_e2_t4" type="car" route="w2_to_e2" begin="2700" end="3600" probability="0.04"/>
</routes>
"""
    os.makedirs('sumo_files_multiagent', exist_ok=True)
    with open('sumo_files_multiagent/routes_scats_empirical.rou.xml', 'w') as f:
        f.write(routes_xml)
    print("✅ [4-Intersection] Generated: sumo_files_multiagent/routes_scats_empirical.rou.xml")


def generate_8_intersection_scats():
    routes_xml = """<?xml version="1.0" encoding="UTF-8"?>
<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">
    <vType id="car" accel="2.6" decel="4.5" sigma="0.5" length="5.0" minGap="2.5" maxSpeed="50.0" guiShape="passenger"/>

    <!-- Routes definition -->
    <route id="n1_s1" edges="n_ext1_to_1 1_to_3 3_to_s_ext1"/>
    <route id="n2_s2" edges="n_ext2_to_2 2_to_4 4_to_s_ext2"/>
    <route id="n3_s3" edges="n_ext3_to_5 5_to_7 7_to_s_ext3"/>
    <route id="n4_s4" edges="n_ext4_to_6 6_to_8 8_to_s_ext4"/>
    <route id="s1_n1" edges="s_ext1_to_3 3_to_1 1_to_n_ext1"/>
    <route id="s2_n2" edges="s_ext2_to_4 4_to_2 2_to_n_ext2"/>
    <route id="s3_n3" edges="s_ext3_to_7 7_to_5 5_to_n_ext3"/>
    <route id="s4_n4" edges="s_ext4_to_8 8_to_6 6_to_n_ext4"/>
    <route id="w1_e1" edges="w_ext1_to_1 1_to_2 2_to_5 5_to_6 6_to_e_ext1"/>
    <route id="w2_e2" edges="w_ext2_to_3 3_to_4 4_to_7 7_to_8 8_to_e_ext2"/>
    <route id="e1_w1" edges="e_ext1_to_6 6_to_5 5_to_2 2_to_1 1_to_w_ext1"/>
    <route id="e2_w2" edges="e_ext2_to_8 8_to_7 7_to_4 4_to_3 3_to_w_ext2"/>

    <!-- Tier 1: 0-900s (Pre-Peak) -->
    <flow id="f_n1_s1_t1" type="car" route="n1_s1" begin="0" end="900" probability="0.02"/>
    <flow id="f_n2_s2_t1" type="car" route="n2_s2" begin="0" end="900" probability="0.02"/>
    <flow id="f_n3_s3_t1" type="car" route="n3_s3" begin="0" end="900" probability="0.02"/>
    <flow id="f_n4_s4_t1" type="car" route="n4_s4" begin="0" end="900" probability="0.02"/>
    <flow id="f_w1_e1_t1" type="car" route="w1_e1" begin="0" end="900" probability="0.02"/>
    <flow id="f_w2_e2_t1" type="car" route="w2_e2" begin="0" end="900" probability="0.02"/>
    <flow id="f_e1_w1_t1" type="car" route="e1_w1" begin="0" end="900" probability="0.02"/>
    <flow id="f_e2_w2_t1" type="car" route="e2_w2" begin="0" end="900" probability="0.02"/>

    <!-- Tier 2: 900-1800s (Rising Peak) -->
    <flow id="f_n1_s1_t2" type="car" route="n1_s1" begin="900" end="1800" probability="0.04"/>
    <flow id="f_n2_s2_t2" type="car" route="n2_s2" begin="900" end="1800" probability="0.04"/>
    <flow id="f_n3_s3_t2" type="car" route="n3_s3" begin="900" end="1800" probability="0.04"/>
    <flow id="f_n4_s4_t2" type="car" route="n4_s4" begin="900" end="1800" probability="0.04"/>
    <flow id="f_w1_e1_t2" type="car" route="w1_e1" begin="900" end="1800" probability="0.04"/>
    <flow id="f_w2_e2_t2" type="car" route="w2_e2" begin="900" end="1800" probability="0.04"/>
    <flow id="f_e1_w1_t2" type="car" route="e1_w1" begin="900" end="1800" probability="0.04"/>
    <flow id="f_e2_w2_t2" type="car" route="e2_w2" begin="900" end="1800" probability="0.04"/>

    <!-- Tier 3: 1800-2700s (Peak SCATS Bottleneck) -->
    <flow id="f_n1_s1_t3" type="car" route="n1_s1" begin="1800" end="2700" probability="0.07"/>
    <flow id="f_n2_s2_t3" type="car" route="n2_s2" begin="1800" end="2700" probability="0.07"/>
    <flow id="f_n3_s3_t3" type="car" route="n3_s3" begin="1800" end="2700" probability="0.07"/>
    <flow id="f_n4_s4_t3" type="car" route="n4_s4" begin="1800" end="2700" probability="0.07"/>
    <flow id="f_w1_e1_t3" type="car" route="w1_e1" begin="1800" end="2700" probability="0.07"/>
    <flow id="f_w2_e2_t3" type="car" route="w2_e2" begin="1800" end="2700" probability="0.07"/>
    <flow id="f_e1_w1_t3" type="car" route="e1_w1" begin="1800" end="2700" probability="0.07"/>
    <flow id="f_e2_w2_t3" type="car" route="e2_w2" begin="1800" end="2700" probability="0.07"/>

    <!-- Tier 4: 2700-3600s (Post-Peak Taper) -->
    <flow id="f_n1_s1_t4" type="car" route="n1_s1" begin="2700" end="3600" probability="0.025"/>
    <flow id="f_n2_s2_t4" type="car" route="n2_s2" begin="2700" end="3600" probability="0.025"/>
    <flow id="f_n3_s3_t4" type="car" route="n3_s3" begin="2700" end="3600" probability="0.025"/>
    <flow id="f_n4_s4_t4" type="car" route="n4_s4" begin="2700" end="3600" probability="0.025"/>
    <flow id="f_w1_e1_t4" type="car" route="w1_e1" begin="2700" end="3600" probability="0.025"/>
    <flow id="f_w2_e2_t4" type="car" route="w2_e2" begin="2700" end="3600" probability="0.025"/>
    <flow id="f_e1_w1_t4" type="car" route="e1_w1" begin="2700" end="3600" probability="0.025"/>
    <flow id="f_e2_w2_t4" type="car" route="e2_w2" begin="2700" end="3600" probability="0.025"/>
</routes>
"""
    os.makedirs('sumo_files_8intersection', exist_ok=True)
    with open('sumo_files_8intersection/routes_scats_empirical.rou.xml', 'w') as f:
        f.write(routes_xml)
    print("✅ [8-Intersection] Generated: sumo_files_8intersection/routes_scats_empirical.rou.xml")


def main():
    print("="*70)
    print("GENERATING EMPIRICAL SCATS DEMAND CURVES ACROSS ALL GRID SIZES")
    print("="*70)
    generate_1_intersection_scats()
    generate_4_intersection_scats()
    generate_8_intersection_scats()
    print("="*70)
    print("✅ Step 1 Completed Successfully!")

if __name__ == '__main__':
    main()
