import networkx as nx

# Create a directed graph for the Antikythera mechanism
graph = nx.DiGraph()

# Add gears/celestial bodies as nodes with properties
graph.add_node("Sun", gear_ratio=1, perspective="center")
graph.add_node("Moon", gear_ratio=13.368, perspective="lunar")
graph.add_node("Mars", gear_ratio=1.8808, perspective="martian")

# Add relationships (edges) between gears
graph.add_edge("Sun", "Moon", ratio=13.368)
graph.add_edge("Sun", "Mars", ratio=1.8808)

# Visualize relationships
print(graph.nodes(data=True))
print(graph.edges(data=True))
