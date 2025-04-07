def main():

    for _ in range(NUM_NIDI):

        # Generate nidus architecture
        compartments = random_normal_int(3, 6)
        columns = random_normal_int(3, 7)
        nidus = create_nidus(compartments, columns)

        # Define feeders and drainers
        feeders = get_feeders(nidus)

        # For each occlusion scenario
        for occluded_feeder in [None] + feeders:

            nidus = apply_occlusion(nidus, occluded_feeder)

            # Simulate all injection scenarios
            for scenario in INJECTION_SCENARIOS:

                flows, pressures, graph = simulate(nidus, scenario)

                # Evaluate physiological outcomes
                stats = get_stats(graph)
                export(stats)


def create_nidus(compartments: int, columns: int) -> Graph:
    """Generates a synthetic AVM nidus as a vascular graph.

    The generated architecture consists of interconnected compartments and columns
    of vessels, including plexiform and a central fistulous path. Intercompartmental
    vessels are also added to mimic anatomical variability.

    Args:
        compartments: Number of horizontal vascular compartments in the nidus.
        columns: Number of vertical columns (proxy for spatial progression from feeder to drainer).

    Returns:
        An undirected graph representing the AVM nidus architecture.
    """
    # Initialize an empty undirected graph with predefined feeders and drainers
    nidus = initialize_graph()

    # Create nodes in a grid of [columns × compartments]
    layout = create_node_layout(compartments, columns)

    # Add intranidal plexiform vessels between adjacent columns within each compartment
    for c in range(columns - 1):
        for k in range(compartments):
            A = layout[c][k]  # Nodes in compartment k, column c
            B = layout[c + 1][k]  # Nodes in same compartment, next column
            randomly_pair_nodes(nidus, A, B, vessel_type="plexiform")

    # Connect each leftmost node to its nearest arterial feeder (by Euclidean distance)
    for node in get_leftmost_nodes(layout):
        feeder = get_closest_feeder(node)
        add_edge(nidus, feeder, node, vessel_type="plexiform")

    # Connect each rightmost node to its nearest draining vein
    for node in get_rightmost_nodes(layout):
        drainer = get_closest_drainer(node)
        add_edge(nidus, node, drainer, vessel_type="plexiform")

    # Add any unconnected feeder or drainer to its closest nidus node
    for feeder in remaining_unconnected_feeders():
        nearest = get_closest_nidus_node(feeder)
        add_edge(nidus, feeder, nearest, vessel_type="plexiform")

    for drainer in remaining_unconnected_drainers():
        nearest = get_closest_nidus_node(drainer)
        add_edge(nidus, nearest, drainer, vessel_type="plexiform")

    # Add intercompartmental vessels
    for _ in range(2 * compartments * columns):
        src_col, src_comp = random_column_and_compartment()
        target_col = sample_nearby_column(src_col)
        target_comp = choose_different_compartment(src_comp)
        src_node = random.choice(layout[src_col][src_comp])
        target_node = random.choice(layout[target_col][target_comp])
        add_edge(nidus, src_node, target_node, vessel_type="plexiform")

    # Add a continuous intranidal fistulous path from AF2 to DV2 through the center compartment
    mid_comp = compartments // 2
    path_nodes = [choose_node(layout[c][mid_comp]) for c in range(columns)]
    add_edge(nidus, "AF2", path_nodes[0], vessel_type="fistulous")
    for i in range(len(path_nodes) - 1):
        add_edge(nidus, path_nodes[i], path_nodes[i + 1], vessel_type="fistulous")
    add_edge(nidus, path_nodes[-1], "DV2", vessel_type="fistulous")

    return nidus


def simulate(nidus: Graph, scenario: Scenario) -> tuple[list[float], list[float], Graph]:
    """Solves for vessel flow and pressure under a given set of EMF inputs.

    Builds a system of linear equations based on Kirchhoff’s laws to model steady-state
    blood flow through the vascular network, then computes pressure gradients using
    Hagen-Poiseuille’s law. Returns a directed graph with physiological flow directions.

    Args:
        nidus: The vascular network to simulate, represented as a graph.
        scenario: A physiological condition defining EMFs and boundary pressures.

    Returns:
        flows: A list of computed flow values for each vessel (mL/min).
        pressures: Corresponding pressure drops across each vessel (mmHg).
        graph: A directed version of the input graph with flow and pressure attributes.
    """
    # Get list of all vessels as directed edges
    vessels = get_all_vessels(nidus)

    # Get external pressure sources (EMFs) for this scenario
    pressure_inputs = scenario.get_external_pressures()

    # Initialize a system of linear equations to solve for flow in each vessel
    equations = []

    # Add one equation per node - conservation of flow
    # ∑ Q_in - ∑ Q_out = 0
    for node in nidus.nodes:
        equations.append(flow_conservation_equation(node, vessels))

    # Add one equation per loop (cycle) - conservation of pressure
    # ∑ R_j Q_j = ∑ EMF_i
    for cycle in get_cycle_basis(nidus):
        equations.append(pressure_loop_equation(cycle, pressure_inputs))

    # Solve the linear system for flows
    flows = solve_linear_system(equations)

    # Compute pressure drop in each vessel: ΔP = R × Q
    pressures = []
    for vessel, flow in zip(vessels, flows):
        resistance = nidus.get_resistance(vessel)
        pressures.append(resistance * flow)

    # Build a directed graph with flow and pressure attributes
    graph = create_flow_graph(nidus, vessels, flows, pressures)

    return flows, pressures, graph


def get_stats(graph: Graph) -> dict:
    """Extracts physiological and structural metrics from a flow simulation.

    Aggregates flow, pressure, and anatomical statistics across different vessel types.
    If an injection was performed, calculates the extent of intranidal filling.

    Args:
        graph: Directed graph output from `simulate()` with flow and pressure attributes.

    Returns:
        A dictionary of physiological and architectural metrics.
    """

    stats = dict()

    # Tally flow and pressure stats for each edge
    for edge in graph.edges():
        update_stats(stats, edge)

    # Estimate rupture risk using log-pressure scaling
    # risk = log(P / Pmin) / log(Pmax / Pmin)
    add_rupture_risk_to_stats(stats, graph)

    # Forward BFS from injection location within nidus
    reached_nodes = bfs(graph, start_from=graph.get_injection_location())

    # Post-injection analysis using flow directions from baseline graph
    reached_nodes = bfs(graph.without_injection(), start_from=reached_nodes)

    stats["filling"] = reached_nodes / graph.get_num_intranidal_vessels()

    return stats