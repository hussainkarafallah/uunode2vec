import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from walk_generator import WalkGenerator
from tqdm import tqdm

#this is a pytorch implementation. Code is throughly commented

#defining the model as a pytorch model
class skipgramModel(nn.Module):

    vocabSize = 19240
    embeddingSize = 10

    def __init__(self):

        print("instatiated model")

        super(skipgramModel, self).__init__()

        #initial weights, each node has embeddingSize weights associated to it, this later becomes the embedding
        self.embedding = nn.Embedding(self.vocabSize, self.embeddingSize)

        #first layer weights
        self.W1 = nn.Linear(self.embeddingSize, self.embeddingSize, bias=False)
        #output layer weights
        self.W2 = nn.Linear(self.embeddingSize, self.vocabSize, bias=False)

    def forward(self, X):
        #receives input node
        embeddings = self.embedding(X)
        #feeds input node into hidden layer, which goes to a relu activation
        hidden_layer = nn.functional.relu(self.W1(embeddings))
        #activates the output
        output_layer = self.W2(hidden_layer)
        return output_layer


class skipgram:

    walks = []
    embeddingSize = 10
    vocabSize = 19240
    batch_size = 2
    windowSize = 1
    learningRate = 0.001
    epochs = 150000

    def __init__(self, walks, embeddingSize, batch_size, windowSize, learningRate, epochs):
        #should be number of nodes
        self.vocabSize=19240
        #number of weights/attributes associated with each node
        self.embeddingSize=embeddingSize
        
        self.walks = walks
        self.batch_size = batch_size
        self.windowSize = windowSize
        self.learningRate = learningRate
        self.epochs = epochs

    #function for generating batches
    def randomBatch(self, skipGrams):
        randomInputs = []
        randomLabels = []

        #generates a range of random indexes of size batch_size, replace false means generated indexes are unique
        randomIndex = np.random.choice(range(len(skipGrams)), self.batch_size, replace=False)

        #for every randomly generated index, appends target and context to their arrays
        for i in randomIndex:
            randomInputs.append(skipGrams[i][0])  # target
            randomLabels.append(skipGrams[i][1])  # context word
            print(skipGrams[i])
        return randomInputs, randomLabels

    #generates node pairs between target nodes and their possible contexts
    def generateSkipgram(self, walk):
        skipGrams = []
        for i in range(self.windowSize, len(walk) - self.windowSize):
            target = walk[i]
            #change this if changing windowSize
            context = [walk[i - self.windowSize], walk[i + self.windowSize]]
            for w in context:
                skipGrams.append([target, w])

        return skipGrams

    def generateAllSkipgrams(self):
        skipGrams = []
        for walk in self.walks:
            skipGrams.append(self.generateSkipgram(walk))
        return skipGrams

    def trainModel(self):

        #instantiates skipgram
        model = skipgramModel()
        #loss function
        criterion = nn.CrossEntropyLoss()
        #pytorch optimizer
        optimizer = optim.Adam(model.parameters(), self.learningRate)

        #genereates all skipgrams for all walks
        skipGrams = self.generateAllSkipgrams()
        #forward and backproprag of the model using generated skipgrams
        #te quiero demasiado
        for epoch in tqdm(range(self.epochs), total=len(skipGrams)):
            #generates random batches
            input_batch, target_batch = self.randomBatch(skipGrams)
            #puts random batches in a pytorch longtensor for faster computing
            input_batch = torch.LongTensor(input_batch)
            target_batch = torch.LongTensor(target_batch)

            #reset gradient
            optimizer.zero_grad()

            #forward proprag
            output = model(input_batch)

            #calculate loss
            loss = criterion(output, target_batch)

            #show loss every 10000 epochs
            if (epoch + 1) % 10000 == 0:
                print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

            #backward proprag
            loss.backward(retain_graph=True)

            #applies calculated correction
            optimizer.step()