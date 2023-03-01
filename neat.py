import random

from genotype import NodeGene, ConnectionGene, Genome, Species
from util import compatibility_distance

class NEAT:
    def __init__(self, num_inputs, num_outputs):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.innovation_number = 0
        self.population = []
        self.species = []
        self.generation = 0
        
        # Hyperparameters
        self.compatibility_threshold = 3.0 # Adjust as needed
        self.crossover_rate = 0.75 # Adjust as needed
        self.mutation_rate = 0.5 # Adjust as needed
        self.weight_mutation_rate = 0.8 # Adjust as needed
        self.weight_perturb_rate = 0.9 # Adjust as needed
        self.node_add_rate = 0.03 # Adjust as needed
        self.connection_add_rate = 0.05 # Adjust as needed
        
    def create_initial_population(self, num_genomes):
        for i in range(num_genomes):
            nodes = []
            connections = []
            # Create input nodes
            for j in range(self.num_inputs):
                nodes.append(NodeGene(j, 'input'))
            # Create output nodes
            for j in range(self.num_outputs):
                nodes.append(NodeGene(j + self.num_inputs, 'output'))
            # Create initial connections between input and output nodes
            for j in range(self.num_inputs):
                for k in range(self.num_outputs):
                    self.innovation_number += 1
                    connections.append(ConnectionGene(j, k + self.num_inputs, random.uniform(-1, 1), True, self.innovation_number))
            genome = Genome(nodes, connections)
            self.population.append(genome)

            self.population_size = len(self.population)
            
    def mutate(self, genome):
        # Mutate weights
        for connection in genome.connections:
            if random.random() < self.weight_mutation_rate:
                if random.random() < self.weight_perturb_rate:
                    connection.weight += random.uniform(-0.5, 0.5)
                else:
                    connection.weight = random.uniform(-1, 1)
        # Mutate node genes
        if random.random() < self.node_add_rate:
            # Add a new node
            connection = random.choice(genome.connections)
            new_node_id = len(genome.nodes)
            new_node = NodeGene(new_node_id, 'hidden')
            genome.nodes.append(new_node)
            connection.enabled = False
            connection1 = ConnectionGene(connection.in_node, new_node_id, 1.0, True, self.innovation_number)
            connection2 = ConnectionGene(new_node_id, connection.out_node, connection.weight, True, self.innovation_number + 1)
            genome.connections.append(connection1)
            genome.connections.append(connection2)
            self.innovation_number += 2

        if random.random() < self.connection_add_rate:
          # Add a new connection
          node1 = random.choice(genome.nodes)
          node2 = random.choice(genome.nodes)
          if node1.node_type == 'output' and node2.node_type == 'input':
            # Swap nodes to avoid output to input connections
            node1, node2 = node2, node1
          existing_connection = False
          for connection in genome.connections:
            if connection.in_node == node1.node_id and connection.out_node == node2.node_id:
                existing_connection = True
                break
          if not existing_connection:
            self.innovation_number += 1
            connection = ConnectionGene(node1.node_id, node2.node_id, random.uniform(-1, 1), True, self.innovation_number)
            genome.connections.append(connection)
            
        return genome
            
    def crossover(self, parent1, parent2):
      child_nodes = []
      child_connections = []
      for node1 in parent1.nodes:
          matching_node = None
          for node2 in parent2.nodes:
              if node1.node_id == node2.node_id:
                matching_node = node2
                break
          if matching_node is None or random.random() < 0.5:
            child_nodes.append(node1)
          else:
            child_nodes.append(matching_node)
      for connection1 in parent1.connections:
        matching_connection = None
        for connection2 in parent2.connections:
            if connection1.innovation_number == connection2.innovation_number:
                matching_connection = connection2
                break
        if matching_connection is None or not connection1.enabled or not matching_connection.enabled or random.random() < 0.5:
            child_connections.append(connection1)
        else:
            child_connections.append(matching_connection)
      child_genome = Genome(child_nodes, child_connections)
      return child_genome
    
    def speciate(self):
      for species in self.species:
        species.members.clear()
      for genome in self.population:
        matching_species = None
        for species in self.species:
            if self.compatibility_distance(genome, species.representative) < self.compatibility_threshold:
                matching_species = species
                break
        if matching_species is None:
            new_species = Species(genome)
            self.species.append(new_species)
        else:
            matching_species.members.append(genome)
      self.species = [species for species in self.species if len(species.members) > 0]
      self.adjust_fitness()
    
    def adjust_fitness(self):
      for genome in self.population:
        genome.fitness = genome.fitness / len(self.species)
      for species in self.species:
        species.fitness = sum(genome.fitness for genome in species.members)
        species.adjusted_fitness = species.fitness / len(species.members)
        
    def evolve(self):
      new_population = []
      for species in self.species:
        if species.adjusted_fitness > 0:
            species_members_sorted = sorted(species.members, key=lambda genome: genome.fitness, reverse=True)
            num_offspring = int(round(species.adjusted_fitness * len(self.population)))
            if num_offspring < 1:
                num_offspring = 1
            offspring = []
            for i in range(num_offspring):
                if i == 0 or random.random() < self.crossover_rate:
                    parent1 = species_members_sorted[random.randint(0, min(4, len(species.members) - 1))]
                    parent2 = species_members_sorted[random.randint(0, min(4, len(species.members) - 1))]
                    child = self.crossover(parent1, parent2)
                else:
                    parent =  species_members_sorted[random.randint(0, min(4, len(species.members) - 1))]
                    child = self.mutate(parent)
                offspring.append(child)
            new_population.extend(offspring)
      while len(new_population) < self.population_size:
        # Add new genomes to the population by mutating existing genomes
        parent = random.choice(self.population)
        child = self.mutate(parent)
        new_population.append(child)
      self.population = new_population
      self.generation += 1
      #self.speciate()