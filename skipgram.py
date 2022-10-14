import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from walk_generator import WalkGenerator
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchviz import make_dot

#this is a pytorch implementation. Code is throughly commented

#defining the model as a pytorch model
class skipgramModel(nn.Module):

    vocabSize = 652
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

    def get_player_emdedding(self, player, player2idx):
        player = torch.tensor([player2idx[player]])
        return self.embedding(player).view(1,-1)


class skipgram:

    skipGrams = []
    player2idx = {}
    idx2player = {}
    walks = []
    walksArray = []
    players = []
    embeddingSize = 10
    vocabSize = 652
    batch_size = 2
    windowSize = 1
    learningRate = 0.001
    epochs = 150000


    def tokenize(self):
        for walk in self.walks:
            self.walksArray.append(walk)
            for node in walk:
                if node not in self.players:
                    self.players.append(node)

        self.player2idx = {w: idx for (idx, w) in enumerate(self.players)}
        self.idx2player = {idx: w for (idx, w) in enumerate(self.players)}

        self.vocabSize = len(self.players)

    def __init__(self, walks, embeddingSize, batch_size, windowSize, learningRate, epochs):
        #number of weights/attributes associated with each node
        self.vocabSize = 652
        self.embeddingSize=embeddingSize
        self.walks = walks
        self.batch_size = batch_size
        self.windowSize = windowSize
        self.learningRate = learningRate
        self.epochs = epochs

        self.tokenize()

    #function for generating batches
    def randomBatch(self):
        randomInputs = []
        randomLabels = []

        #generates a range of random indexes of size batch_size, replace false means generated indexes are unique
        randomIndex = np.random.choice(range(len(self.skipGrams)), self.batch_size, replace=False)

        #for every randomly generated index, appends target and context to their arrays
        for i in randomIndex:
            randomInputs.append(self.skipGrams[i][0])  # target
            randomLabels.append(self.skipGrams[i][1])  # context word
            #print(skipGrams[i])
        return randomInputs, randomLabels

    #generates node pairs between target nodes and their possible contexts
    def generateSkipgram(self, walk):
        for i in range(self.windowSize, len(walk) - self.windowSize):
            target = self.player2idx[walk[i]]
            #change this if changing windowSize
            context = [self.player2idx[walk[i- self.windowSize]], self.player2idx[walk[i+ self.windowSize]]]
            for w in context:
                self.skipGrams.append([target, w])

        #print(self.skipGrams)

    def generateAllSkipgrams(self):
        for walk in self.walksArray:
            self.generateSkipgram(walk)

    def skipgramTest(self, test_data, model):
        correct_ct = 0

        for i in range(len(test_data)):
            input_batch, target_batch = self.randomBatch()
            input_batch = torch.LongTensor(input_batch)
            target_batch = torch.LongTensor(target_batch)

            model.zero_grad()
            _, predicted = torch.max(model(input_batch), 1)




            if predicted[0] == target_batch[0]:
                    correct_ct += 1

        print('Accuracy: {:.1f}% ({:d}/{:d})'.format(correct_ct/len(test_data)*100, correct_ct, len(test_data)))

    def trainModel(self):

        #instantiates skipgram
        model = skipgramModel()
        #loss function
        criterion = nn.CrossEntropyLoss()
        #pytorch optimizer
        optimizer = optim.Adam(model.parameters(), self.learningRate)

        #genereates all skipgrams for all walks
        self.generateAllSkipgrams()
        #forward and backproprag of the model using generated skipgrams
        #te quiero demasiado

        for epoch in tqdm(range(self.epochs)):
            #generates random batches
            input_batch, target_batch = self.randomBatch()
            #puts random batches in a pytorch longtensor for faster computing
            input_batch = torch.LongTensor(input_batch)
            target_batch = torch.LongTensor(target_batch)

            #reset gradient
            optimizer.zero_grad()

            #forward proprag
            #print("aaaa" + str(input_batch))
            output = model(input_batch)

            #calculate loss
            #print(np.shape(output), np.shape(target_batch))
            loss = criterion(output, target_batch)

            #show loss every 10000 epochs
            if (epoch + 1) % 10000 == 0:
                print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

            #backward proprag
            loss.backward(retain_graph=True)

            #applies calculated correction
            optimizer.step()

        #self.skipgramTest(self.skipGrams,model)

        plt.figure(figsize=(20,15))
        for player in self.players[int(len(self.players)*0.4):int(len(self.players)*0.6)]:
            x = model.get_player_emdedding(player, self.player2idx).detach().data.numpy()[0][0]
            y = model.get_player_emdedding(player, self.player2idx).detach().data.numpy()[0][1]
            plt.scatter(x, y)
            plt.annotate(player, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
        plt.show()

        make_dot(input_batch, params=dict(list(model.named_parameters()))).render("rnn_torchviz", format="png")