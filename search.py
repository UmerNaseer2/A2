"""
COS30019 Assignment 2A - Route Finding Search Algorithms
==========================================================
This file serves two purposes:
  1. AIMA-compatible Problem/Node classes (importable by search.ipynb)
  2. CLI entry point: python search.py <filename> <method>

Algorithms: DFS, BFS, GBFS, AS, CUS1 (IDDFS), CUS2 (Weighted A*)
"""

import sys
import math
import heapq
from dataclasses import dataclass
from collections import deque


# ══════════════════════════════════════════════════════════════════
#  AIMA-COMPATIBLE BASE CLASSES
# ══════════════════════════════════════════════════════════════════

class Problem:
    """Abstract base class for a search problem (AIMA-compatible)."""

    def __init__(self, initial, goal=None):
        self.initial = initial
        self.goal = goal

    def actions(self, state):
        raise NotImplementedError

    def result(self, state, action):
        raise NotImplementedError

    def goal_test(self, state):
        if isinstance(self.goal, list):
            return state in self.goal
        return state == self.goal

    def path_cost(self, c, state1, action, state2):
        return c + 1

    def h(self, node):
        return 0


class Node:
    """A node in a search tree (AIMA-compatible)."""

    def __init__(self, state, parent=None, action=None, path_cost=0):
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost
        self.depth = 0 if parent is None else parent.depth + 1

    def __repr__(self):
        return f"<Node {self.state}>"

    def __lt__(self, other):
        return self.state < other.state

    def __eq__(self, other):
        return isinstance(other, Node) and self.state == other.state

    def __hash__(self):
        return hash(self.state)

    def expand(self, problem):
        return [self.child_node(problem, action)
                for action in problem.actions(self.state)]

    def child_node(self, problem, action):
        next_state = problem.result(self.state, action)
        return Node(next_state, self, action,
                    problem.path_cost(self.path_cost, self.state, action, next_state))

    def solution(self):
        """Return list of actions from root to this node."""
        return [node.action for node in self.path()[1:]]

    def path(self):
        """Return list of nodes from root to this node."""
        node, path_back = self, []
        while node:
            path_back.append(node)
            node = node.parent
        return list(reversed(path_back))


# ══════════════════════════════════════════════════════════════════
#  SEARCH BOOKKEEPING
# ══════════════════════════════════════════════════════════════════

@dataclass
class SearchStats:
    """Lightweight metrics attached to the problem and returned goal node."""

    generated: int = 0
    expanded: int = 0


def _finish_search(problem, node, stats):
    """
    Attach search metrics to both the problem and the returned node.
    This keeps the public API compatible while giving the CLI exact counts.
    """
    problem.search_stats = stats
    if node is not None:
        node.search_stats = stats
    return node


# ══════════════════════════════════════════════════════════════════
#  FILE PARSER
# ══════════════════════════════════════════════════════════════════

def parse_problem(filename):
    """Parse a problem file -> (nodes_coords, edges, origin, destinations)."""
    nodes_coords = {}
    edges = {}
    origin = None
    destinations = []
    section = None

    with open(filename, 'r') as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            lower = line.lower()
            if lower.startswith('nodes:'):
                section = 'nodes'
                continue
            elif lower.startswith('edges:'):
                section = 'edges'
                continue
            elif lower.startswith('origin:'):
                rest = line[7:].strip()
                if rest:
                    origin = int(rest)
                    section = None
                else:
                    section = 'origin'
                continue
            elif lower.startswith('destinations:'):
                rest = line[13:].strip()
                if rest:
                    destinations = [int(d.strip()) for d in rest.split(';') if d.strip()]
                    section = None
                else:
                    section = 'destinations'
                continue

            if section == 'nodes':
                parts = line.split(':')
                node_id = int(parts[0].strip())
                coord_str = parts[1].strip().strip('()')
                x, y = [int(v.strip()) for v in coord_str.split(',')]
                nodes_coords[node_id] = (x, y)
                edges.setdefault(node_id, [])

            elif section == 'edges':
                paren_end = line.index(')')
                from_node, to_node = [int(v.strip()) for v in line[1:paren_end].split(',')]
                cost = int(line[paren_end + 1:].strip().lstrip(':').strip())
                edges.setdefault(from_node, [])
                edges[from_node].append((to_node, cost))

            elif section == 'origin':
                origin = int(line)
                section = None

            elif section == 'destinations':
                destinations = [int(d.strip()) for d in line.split(';') if d.strip()]
                section = None

    for node_id in edges:
        edges[node_id].sort(key=lambda x: x[0])

    return nodes_coords, edges, origin, destinations


# ══════════════════════════════════════════════════════════════════
#  ROUTE FINDING PROBLEM
# ══════════════════════════════════════════════════════════════════

class RouteProblem(Problem):
    """
    Route finding on a weighted directed graph.
    State: node ID (int). Action: neighbour node ID (int).
    """

    def __init__(self, initial, goals, nodes_coords, edges):
        super().__init__(initial, goals)
        self.nodes_coords = nodes_coords
        self.edges = edges
        self.goals = goals
        self.heuristic_scale = self._compute_heuristic_scale()

    def actions(self, state):
        return [n for n, _ in sorted(self.edges.get(state, []), key=lambda x: x[0])]

    def result(self, state, action):
        return action

    def goal_test(self, state):
        return state in self.goals

    def path_cost(self, c, state1, action, state2):
        for neighbour, cost in self.edges.get(state1, []):
            if neighbour == state2:
                return c + cost
        return c + math.inf

    def _compute_heuristic_scale(self):
        """
        Scale straight-line distance into a safe lower bound on path cost.

        For every edge, cost >= scale * EuclideanDistance(edge).
        Using the minimum such scale keeps A* admissible even when edge costs
        are not identical to geometric distance.
        """
        ratios = []
        for from_node, neighbours in self.edges.items():
            x1, y1 = self.nodes_coords[from_node]
            for to_node, cost in neighbours:
                x2, y2 = self.nodes_coords[to_node]
                distance = math.dist((x1, y1), (x2, y2))
                if distance > 0:
                    ratios.append(cost / distance)
        return min(ratios, default=0)

    def h(self, node):
        x1, y1 = self.nodes_coords[node.state]
        straight_line = min(
            math.dist((x1, y1), self.nodes_coords[d])
            for d in self.goals
        )
        return self.heuristic_scale * straight_line


# ══════════════════════════════════════════════════════════════════
#  SEARCH ALGORITHMS
#  All return a goal Node (AIMA-compatible) or None
# ══════════════════════════════════════════════════════════════════

def best_first_graph_search(problem, evaluation_fn):
    """
    Shared engine for informed graph search.

    The frontier is ordered by `evaluation_fn(node)`.
    Duplicate states are only reopened when a cheaper path is found.
    """
    counter = 0
    start = Node(problem.initial)
    stats = SearchStats(generated=1)
    frontier = [(evaluation_fn(start), start.state, counter, start)]
    best_g = {start.state: 0}

    while frontier:
        _, _, _, node = heapq.heappop(frontier)

        if node.path_cost > best_g.get(node.state, math.inf):
            continue

        if problem.goal_test(node.state):
            return _finish_search(problem, node, stats)

        stats.expanded += 1

        for child in node.expand(problem):
            if child.path_cost >= best_g.get(child.state, math.inf):
                continue
            best_g[child.state] = child.path_cost
            counter += 1
            stats.generated += 1
            heapq.heappush(
                frontier,
                (evaluation_fn(child), child.state, counter, child),
            )

    return _finish_search(problem, None, stats)


def depth_first_graph_search(problem):
    """DFS with cycle detection. Tie-break: smaller node IDs first."""
    node = Node(problem.initial)
    stack = [node]
    explored = set()
    while stack:
        node = stack.pop()
        if node.state in explored:
            continue
        explored.add(node.state)
        if problem.goal_test(node.state):
            return node
        for child in sorted(node.expand(problem), key=lambda n: n.state, reverse=True):
            if child.state not in explored:
                stack.append(child)
    return None


def breadth_first_graph_search(problem):
    """BFS with cycle detection. Tie-break: ascending node ID."""
    node = Node(problem.initial)
    if problem.goal_test(node.state):
        return node
    frontier = deque([node])
    explored = set()
    while frontier:
        node = frontier.popleft()
        explored.add(node.state)
        for child in node.expand(problem):
            if child.state not in explored and child not in frontier:
                if problem.goal_test(child.state):
                    return child
                frontier.append(child)
    return None


def greedy_best_first_graph_search(problem):
    """GBFS using the shared best-first engine with f(n) = h(n)."""
    return best_first_graph_search(problem, lambda node: problem.h(node))


def astar_search(problem):
    """A* using the shared best-first engine with f(n) = g(n) + h(n)."""
    return best_first_graph_search(
        problem,
        lambda node: node.path_cost + problem.h(node),
    )


def core_search_engine(problem):
    """
    Preferred route-finding engine for this assignment.

    We use A* as the shared "brain" because it combines:
      - actual travel cost so far, g(n)
      - an admissible straight-line lower bound, h(n)
    This makes it the best default choice for weighted route finding.
    """
    return astar_search(problem)


def iterative_deepening_search(problem):
    """
    CUS1: Iterative Deepening DFS (IDDFS).
    Uninformed — combines DFS space efficiency with BFS completeness.
    Increments depth limit from 0 upwards until goal found.
    """
    def dls(node, limit):
        if problem.goal_test(node.state):
            return node
        if limit == 0:
            return 'cutoff'
        cutoff_occurred = False
        for child in node.expand(problem):
            result = dls(child, limit - 1)
            if result == 'cutoff':
                cutoff_occurred = True
            elif result is not None:
                return result
        return 'cutoff' if cutoff_occurred else None

    for depth in range(sys.maxsize):
        result = dls(Node(problem.initial), depth)
        if result != 'cutoff':
            return result
    return None


def weighted_astar_search(problem, weight=1.5):
    """
    CUS2: Weighted A* — f(n) = g(n) + w*h(n), w > 1.
    Informed — biases toward the goal more aggressively than A*.
    w=1 reduces to standard A*; higher w = faster but less optimal.
    """
    return best_first_graph_search(
        problem,
        lambda node: node.path_cost + weight * problem.h(node),
    )


# ══════════════════════════════════════════════════════════════════
#  RESEARCH: VISIT ALL DESTINATIONS (shortest combined path)
# ══════════════════════════════════════════════════════════════════

def visit_all_destinations(nodes_coords, edges, origin, destinations):
    """
    Finds the shortest path from origin that visits ALL destination nodes.

    Strategy: try all permutations of destinations, stitch segments
    together using A*, return the permutation with the lowest total cost.
    Works well for small destination sets (2-5 nodes).

    Returns (total_cost, full_path_list) or None if unreachable.
    """
    import itertools

    def astar_segment(start, end):
        prob = RouteProblem(start, [end], nodes_coords, edges)
        node = astar_search(prob)
        if node is None:
            return None
        return node.path_cost, [n.state for n in node.path()]

    best_cost = math.inf
    best_path = None

    for perm in itertools.permutations(destinations):
        waypoints = [origin] + list(perm)
        total_cost = 0
        full_path = [origin]
        valid = True

        for i in range(len(waypoints) - 1):
            result = astar_segment(waypoints[i], waypoints[i + 1])
            if result is None:
                valid = False
                break
            cost, segment = result
            total_cost += cost
            full_path += segment[1:]

        if valid and total_cost < best_cost:
            best_cost = total_cost
            best_path = full_path

    return (best_cost, best_path) if best_path else None


# ══════════════════════════════════════════════════════════════════
#  NODE COUNTER WRAPPER (for CLI compatibility)
# ══════════════════════════════════════════════════════════════════

class CountingProblem(Problem):
    """
    Wrap a RouteProblem and count generated children the same way the
    assignment-style CLI expects.
    """

    def __init__(self, route_problem):
        self.inner = route_problem
        self.initial = route_problem.initial
        self.goal = route_problem.goals
        self.nodes_created = 1

    def actions(self, state):
        neighbours = self.inner.actions(state)
        self.nodes_created += len(neighbours)
        return neighbours

    def result(self, state, action):
        return self.inner.result(state, action)

    def goal_test(self, state):
        return self.inner.goal_test(state)

    def path_cost(self, c, s1, a, s2):
        return self.inner.path_cost(c, s1, a, s2)

    def h(self, node):
        return self.inner.h(node)


# ══════════════════════════════════════════════════════════════════
#  CLI DISPATCHER
# ══════════════════════════════════════════════════════════════════

ALGORITHM_MAP = {
    'DFS':  depth_first_graph_search,
    'BFS':  breadth_first_graph_search,
    'GBFS': greedy_best_first_graph_search,
    'AS':   astar_search,
    'CUS1': iterative_deepening_search,
    'CUS2': weighted_astar_search,
}


def run_search(filename, method):
    method = method.upper()
    if method not in ALGORITHM_MAP:
        print(f"Error: Unknown method '{method}'. Choose from: {', '.join(ALGORITHM_MAP)}")
        sys.exit(1)

    nodes_coords, edges, origin, destinations = parse_problem(filename)
    route = RouteProblem(origin, destinations, nodes_coords, edges)
    wrapped = CountingProblem(route)
    goal_node = ALGORITHM_MAP[method](wrapped)

    print(f"{filename} {method}")
    if goal_node is None:
        print("No solution found.")
    else:
        path = [n.state for n in goal_node.path()]
        print(f"{goal_node.state} {wrapped.nodes_created}")
        print(' -> '.join(str(n) for n in path))


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python search.py <filename> <method>")
        print(f"Methods: {', '.join(ALGORITHM_MAP)}")
        sys.exit(1)
    run_search(sys.argv[1], sys.argv[2])
