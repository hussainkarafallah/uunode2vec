from node2vec import node2vec
import argparse

if __name__ == '__main__':
    # follow these examples to add hyperparameters we may need
    parser = argparse.ArgumentParser()
    parser.add_argument("-dim" , action='store' , type=int , default = 64 , help = "embeddings dimensions")

    parser.add_argument("-p" , action = 'store' , type = float , default = 1.0 , help = "parameter p of node2vec")
    parser.add_argument("-q" , action= 'store' , type = float , default = 1.0 , help = "parameter q of node2vec")
    parser.add_argument("-l" , action='store' , type=int , default= 20 , help= "random walk length")
    parser.add_argument("-m" , action='store' , type = int , default=10 , help = "walks per node")

    parser.add_argument("-lr" , action='store' , type=float , default=0.01 , help = "skip gram learning rate")
    parser.add_argument("-k" , action='store' , type=float , default=5 , help = "skip gram window")

    args = parser.parse_args()

    node2vec(
        args.dim,
        args.p,
        args.q,
        args.l,
        args.m,
        args.k,
        args.lr
    )


