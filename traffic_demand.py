"""Validated, reproducible SUMO demand; no controller-dependent spawning."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import sumolib


def route_definitions(net_file, route_file):
    net = sumolib.net.readNet(str(net_file))
    root = ET.parse(route_file).getroot()
    routes = {r.attrib['id']: r.attrib['edges'].split() for r in root.findall('route')}

    def edges_for(item):
        if item.get('route'):
            if item.get('route') not in routes:
                raise ValueError(f"Unknown route {item.get('route')}")
            edges = routes[item.get('route')]
        elif item.find('route') is not None:
            edges = item.find('route').attrib['edges'].split()
        elif item.get('from') and item.get('to'):
            path, _ = net.getShortestPath(net.getEdge(item.get('from')), net.getEdge(item.get('to')))
            if path is None:
                raise ValueError(f"No route for {item.attrib}")
            edges = [e.getID() for e in path]
        else:
            raise ValueError(f"An explicit route or from/to is required: {item.attrib}")
        for edge in edges:
            if not net.hasEdge(edge):
                raise ValueError(f"Unknown network edge {edge}")
        for a, b in zip(edges, edges[1:]):
            if net.getEdge(b) not in net.getEdge(a).getOutgoing():
                raise ValueError(f"Disconnected route: {a} -> {b}")
        return edges

    # Validate unused named routes too: invalid input must not silently pass.
    for route in root.findall('route'):
        edges_for(ET.Element('vehicle', route=route.get('id')))
    return root, edges_for


def materialize_routes(net_file, route_file, output_file, seed=42, multiplier=1.0, horizon=3600):
    """Expand Bernoulli flows into sorted vehicles with inline routes.

    The same seed uses common random numbers across demand multipliers. Explicit
    schedules are preserved, including their vehicle IDs. No SUMO --random flag.
    """
    root, edges_for = route_definitions(net_file, route_file)
    rng = np.random.default_rng(seed)
    result = ET.Element('routes')
    for typ in root.findall('vType'):
        result.append(copy.deepcopy(typ))
    vehicles = []
    for item in root:
        if item.tag == 'vehicle':
            depart = float(item.get('depart'))
            if 0 <= depart < horizon:
                attrs = {k: v for k, v in item.attrib.items() if k not in {'route', 'from', 'to'}}
                v = ET.Element('vehicle', attrs)
                ET.SubElement(v, 'route', edges=' '.join(edges_for(item)))
                vehicles.append((depart, v))
        elif item.tag == 'flow':
            if 'probability' not in item.attrib:
                raise ValueError('Only probability flows and explicit vehicles are supported')
            p = float(item.get('probability')) * multiplier
            if not 0 <= p <= 1:
                raise ValueError(f"Invalid scaled probability {p}: {item.get('id')}")
            edges = edges_for(item)
            begin = max(0, int(float(item.get('begin', '0'))))
            end = min(int(float(item.get('end', str(horizon)))), horizon)
            for t in range(begin, end):
                if rng.random() < p:
                    attrs = {'id': f"{item.get('id')}_{t}", 'depart': str(t),
                             'departLane': 'best', 'departSpeed': 'max'}
                    if item.get('type'):
                        attrs['type'] = item.get('type')
                    v = ET.Element('vehicle', attrs)
                    ET.SubElement(v, 'route', edges=' '.join(edges))
                    vehicles.append((float(t), v))
    vehicles.sort(key=lambda p: (p[0], p[1].get('id')))
    ids = [v.get('id') for _, v in vehicles]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate vehicle IDs')
    result.extend(v for _, v in vehicles)
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(result)
    ET.ElementTree(result).write(output_file, encoding='utf-8', xml_declaration=True)
    return {v.get('id'): t for t, v in vehicles}


def scheduled_departures(route_file, horizon):
    root = ET.parse(route_file).getroot()
    if root.findall('flow') or root.findall('trip'):
        raise ValueError('Metrics require a materialized vehicle schedule')
    return {v.get('id'): float(v.get('depart')) for v in root.findall('vehicle')
            if 0 <= float(v.get('depart')) < horizon}


def demand_capacity(net_file, route_file, multiplier=1.0, horizon=3600):
    """Mean hourly demand / nominal green-adjusted lane capacity.

    Count each traversed incoming lane movement, including downstream crossings.
    Split demand over all compatible lanes; shared lanes get summed movement
    demand. The critical lane at each signal determines that signal's ratio.
    This is nominal capacity, not an empirical saturation-flow measurement.
    """
    root, edges_for = route_definitions(net_file, route_file)
    net = sumolib.net.readNet(str(net_file))
    demand = {}
    for item in list(root.findall('flow')) + list(root.findall('vehicle')):
        if item.tag == 'flow':
            duration = max(0., min(horizon, float(item.get('end', horizon))) - max(0., float(item.get('begin', '0'))))
            prob = float(item.get('probability')) * multiplier
            if not 0 <= prob <= 1:
                raise ValueError('Scaled probability outside [0, 1]')
            rate = prob * duration * 3600 / horizon
        else:
            rate = 3600 / horizon if 0 <= float(item.get('depart')) < horizon else 0
        edges = edges_for(item)
        for a, b in zip(edges, edges[1:]):
            connections = net.getEdge(a).getConnections(net.getEdge(b))
            lanes = {c.getFromLane().getID() for c in connections if c.getTLSID()}
            for lane in lanes:
                demand[lane] = demand.get(lane, 0.) + rate / len(lanes)
    ratios = {}
    for tls in net.getTrafficLights():
        lanes = {c[0].getID() for c in tls.getConnections()}
        ratios[tls.getID()] = max((demand.get(lane, 0.) for lane in lanes), default=0.) / (1800 * 30 / 66)
    return {'per_intersection_vc': ratios, 'mean_critical_vc': float(np.mean(list(ratios.values()))),
            'max_critical_vc': max(ratios.values()), 'nominal_saturation_headway_s': 2.0,
            'reference_green_s': 30, 'reference_cycle_s': 66}


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()
