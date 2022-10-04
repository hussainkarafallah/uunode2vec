from preprocessing import load_and_process
from walk_generator import WalkGenerator
from skipgram import SkipGram

def node2vec(
    dim : int,
    p : float , 
    q : float , 
    walk_length : int , 
    walks_per_node : int , 
    k : int , 
    lr : float
):
    graph = load_and_process()
    generator = WalkGenerator(graph , p , q , walk_length , walks_per_node)
    skip_gram = SkipGram(graph.n , dim , k , lr)
    skip_gram.train(generator.generate())



