def compatibility_distance(self, genome1, genome2):
    # Count matching genes
    matching_genes = 0
    for connection1 in genome1.connections:
        for connection2 in genome2.connections:
            if connection1.innovation_number == connection2.innovation_number:
                matching_genes += 1
                break
    # Calculate compatibility distance
    excess_genes = abs(len(genome1.connections) - len(genome2.connections))
    if len(genome1.connections) > 20 and len(genome2.connections) > 20:
        N = max(len(genome1.connections), len(genome2.connections))
    else:
        N = 1
    compatibility_distance = ((self.weight_mutation_rate * (sum([abs(connection1.weight - connection2.weight) for connection1 in genome1.connections for connection2 in genome2.connections if connection1.innovation_number == connection2.innovation_number])/matching_genes)) +
                              (self.connection_add_rate * excess_genes / N))
    return compatibility_distance