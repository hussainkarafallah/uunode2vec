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
generator = WalkGenerator(graph , p=1, q=1, walk_length=10 , walks_per_node=5)
skip_gram = skipgram(generator.generate(), embeddingSize=10, batch_size=2, windowSize=1, learningRate=0.01, epochs=150000)
skip_gram.trainModel()



