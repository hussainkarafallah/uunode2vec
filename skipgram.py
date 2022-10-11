# I recommend looking for a pytorch or tensorflow >= 2.0 implementation
# I never implemented this myself (always used gensim)
# feel free to get any implementation

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

#should be number of nodes
vocabSize=19240
#number of weights/attributes associated with each word
embeddingSize=10

#generates word pairs between target words and their possible contexts
def generateSkipgram(walk, windowSize=1):
   skip_grams = []
   for i in range(windowSize, len(walk) - windowSize):
       target = walk[i]
       #change this if changing windowSize
       context = [walk[i - windowSize], walk[i + windowSize]]
       for w in context:
           skip_grams.append([target, w])

   return skip_grams

class skipgramModel(nn.Module):

    def __init__(self):

        super(skipgramModel, self).__init__()

        #initial weights, each node has embeddingSize weights associated to it, this later becomes the embedding
        self.embedding = nn.Embedding(vocabSize, embeddingSize)

        #first layer weights
        self.W1 = nn.Linear(embeddingSize, embeddingSize, bias=False)
        #output layer weights
        self.W2 = nn.Linear(embeddingSize, vocabSize, bias=False)

    def forward(self, X):
        #receives input node
        embeddings = self.embedding(X)
        #feeds input node into hidden layer, which goes to a relu activation
        hidden_layer = nn.functional.relu(self.W1(embeddings))
        #activates the output
        output_layer = self.W2(hidden_layer)
        return output_layer

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

#everythin below hasn't been reviewed
for epoch in tqdm(range(150000), total=len(generateSkipgram(walk))):
    input_batch, target_batch = random_batch(generateSkipgram(walk))
    input_batch = torch.LongTensor(input_batch)
    target_batch = torch.LongTensor(target_batch)

    optimizer.zero_grad()
    output = model(input_batch)

    # output : [batch_size, voc_size], target_batch : [batch_size] (LongTensor, not one-hot)
    loss = criterion(output, target_batch)
    if (epoch + 1) % 10000 == 0:
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

    loss.backward(retain_graph=True)
    optimizer.step()

def testSkipgram(test_data, model):
    correct_ct = 0

    for i in range(len(test_data)):
        input_batch, target_batch = random_batch(test_data)
        input_batch = torch.LongTensor(input_batch)
        target_batch = torch.LongTensor(target_batch)

        model.zero_grad()
        _, predicted = torch.max(model(input_batch), 1)




        if predicted[0] == target_batch[0]:
                correct_ct += 1

    print('Accuracy: {:.1f}% ({:d}/{:d})'.format(correct_ct/len(test_data)*100, correct_ct, len(test_data)))