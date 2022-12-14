from collections import defaultdict


class Graph:
    def __init__(self , n : int) -> None:
        self.n = n
        self.adj = defaultdict(list)

    def add_edge(self , u : int , v : int , w : float):
        self.adj[u].append( (v , float(w)) )
        self.adj[v].append( (u , float(w)) )

    def items(self):
        return self.adj.items()

    

