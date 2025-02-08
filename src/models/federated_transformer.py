class FederatedTransformer:
    def __init__(self, graph):
        self.graph = graph
        self.models = {}
        
        # Create a transformer for each node in the graph
        for node, data in graph.nodes(data=True):
            self.models[node] = GearTransformer(node, data["gear_ratio"], data["perspective"])

    def forward(self, inputs):
        outputs = {}

        # Pass inputs through each transformer's forward method
        for node in self.graph.nodes:
            model = self.models[node]
            outputs[node] = model.forward(inputs[node])
        
        # Combine outputs based on topology (e.g., weighted by edge ratios)
        for source, target, data in self.graph.edges(data=True):
            ratio = data["ratio"]
            outputs[target] += outputs[source] * ratio

        return outputs
