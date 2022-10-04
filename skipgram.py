# I recommend looking for a pytorch or tensorflow >= 2.0 implementation
# I never implemented this myself (always used gensim)
# feel free to get any implementation


from typing import Generator


class SkipGram:

    # This can be a neural network or whatever
    # I recommend looking for a tf>2 implementation (tf-keras) or pytorch
    # will be easier to review and debug

    def __init__(self , n : int , dim : int , k : int , lr : float):
        self.lr = lr
        self.k = k
        self.dim = dim
        self.n = n
        # self.embeddings = np.array((n , dim))
        # ....

    def train(self , walk_generator : Generator):
        for walk in walk_generator:
            # do some shit
            print(walk)

