from typing import List
from graph import Graph

class WalkGenerator:
    
    def __init__(self , g : Graph , p : float , q : float , walk_length : int , per_node : int) -> None:
        self.graph = g
        self.p = p
        self.q = q
        self.walk_length = walk_length
        self.per_node = per_node
        self.preprocess()

    def preprocess(self):
        # a function to calculate transition probabilities
        pass

    def generate(self) -> List[int]:
        # a function that yields walks
        for x in range(5):
            yield [1,2,3,4]

    