from typing import List
from graph import Graph
import numpy as np
import random 
from collections import defaultdict

class WalkGenerator:

    def __init__(self , g : Graph , p : float , q : float , walk_length : int , per_node : int) -> None:
        
        self.graph = g
        self.p = p
        self.q = q
        self.walk_length = walk_length
        self.per_node = per_node

        # store direct neighbors and convert weights to probabilities with sum = 1.0
        self.direct_nb = { node : [ nb[0] for nb in nbs  ] for (node,nbs) in g.items()}
        self.direct_nb_distrib = { node : np.array([ nb[1] for nb in nbs ]) for (node,nbs) in g.items()}
        for x in self.direct_nb_distrib:
            self.direct_nb_distrib[x] /= np.sum(self.direct_nb_distrib[x])

        # this dictionary holds the transition probability from a state (a , b) so standing
        # at node b given that it came from a (same as above, distribution)
        self.biased_walk_nb = {}
        self.biased_walk_nb_distrib = {}

        # just to make it convenient to store nodes ids
        self.nodes_ids = list(g.adj.keys())

        # init dicts
        for x in self.nodes_ids:
            self.biased_walk_nb[x] = {}
            self.biased_walk_nb_distrib[x] = {}
            for nxt in self.direct_nb[x]:
                self.biased_walk_nb[x][nxt] = []
                self.biased_walk_nb_distrib[x][nxt] = []

        self.preprocess()
    
    def construct(self):
        return defaultdict(list)

    def preprocess(self):
        # a function to calculate transition probabilities
        for parent in self.nodes_ids:
            for current in self.direct_nb[parent]:
                for (nxt , w) in zip(self.direct_nb[current] , self.direct_nb_distrib[current]):
                    if current == parent:
                        w *= 1.0 / self.p
                    elif nxt in self.direct_nb[parent]:
                        # just keep it this way for clarity
                        w *= 1.0
                    else:
                        w *= 1.0 / self.q
                    self.biased_walk_nb[parent][current].append(nxt)
                    self.biased_walk_nb_distrib[parent][current].append(w)
        
        for x in self.nodes_ids:
            for y in self.biased_walk_nb[x]:
                self.biased_walk_nb_distrib[x][y] = np.array(self.biased_walk_nb_distrib[x][y]) / np.sum(self.biased_walk_nb_distrib[x][y])
        
        #print(self.biased_walk_nb)
        #print(self.biased_walk_nb_distrib)
        
    def walk(self , node):
        # start from a node and pick one of its neoghbours
        pre_last = node
        last = np.random.choice(self.direct_nb[node] , size = 1 , p = self.direct_nb_distrib[node])[0]
        walk = [pre_last , last]
        # now we know the last 2 nodes of the walk, we can keep appending according to calculated probabilites
        while len(walk) < self.walk_length:
            walk.append(np.random.choice(self.biased_walk_nb[pre_last][last] , size = 1 , p = self.biased_walk_nb_distrib[pre_last][last])[0])
            pre_last , last = walk[-2:]
        return walk


    def generate(self) -> List[int]:
        for node in self.nodes_ids:
            for i in range(self.per_node):
                yield self.walk(node)
        

if __name__ == '__main__':
    xx = Graph(5)
    xx.add_edge(1 , 2 , 1)
    xx.add_edge(1 , 3 , 3)
    xx.add_edge(3 , 4 , 5)
    xx.add_edge(3 , 5 , 5)
    gen = WalkGenerator(xx , 1 , 1 , 10 , 1)
    for arr in gen.generate():
        print(arr)