class FeedbackFederatedTransformer(FederatedTransformer):
    def forward(self, inputs, num_steps=10):
        states = {node: torch.zeros_like(inputs[node]) for node in self.graph.nodes}

        for t in range(num_steps):
            outputs = super().forward(states)  # Process current state
            for node in self.graph.nodes:
                states[node] = outputs[node]  # Update states with feedback

        return states
