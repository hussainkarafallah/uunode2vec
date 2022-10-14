from preprocessing import load_and_process
from walk_generator import WalkGenerator
from skipgram import skipgram

def node2vec(
    dim : int,
    p : float , 
    q : float , 
    walk_length : int , 
    walks_per_node : int , 
    k : int , 
    lr : float
):
    pass
    
#outside of function so it runs and we can test it

graph = load_and_process()
print("PASSED PREPROCESS")
generator = WalkGenerator(graph , p=1, q=1, walk_length=10 , per_node=5)
print("PASSED GEN")
generator.construct()
walks = generator.generate()
skip_gram = skipgram(walks, 10, 50, 1, 0.01, 50000)
print("PASSED SKIPGRAM")
skip_gram.trainModel()
print("TRAINED")