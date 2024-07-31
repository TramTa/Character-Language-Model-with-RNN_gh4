# models.py

import numpy as np
import collections
from torch import optim
import random

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import scipy.spatial.distance


class RNNOverWords(nn.Module):
    def __init__(self, dict_size, input_size, hidden_size, dropout, rnn_type='lstm'):
        super(RNNOverWords, self).__init__()
        self.word_embedding = nn.Embedding(dict_size, input_size)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn_type = rnn_type
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers=1, dropout=dropout)    
        self.log_softmax = nn.LogSoftmax(dim=-1)      
        self.hidden2 = nn.Linear(hidden_size, 2)     # hidden to 2: vowel + constant
        self.hidden27 = nn.Linear(hidden_size, 27)   # hidden to 27 char 
        self.init_weight() 


    def init_weight(self):
        # This is a randomly initialized RNN.
        # Bounds from https://openreview.net/pdf?id=BkgPajAcY7
        # Note: this is to make a random LSTM; these are *NOT* necessarily good weights for initializing learning!
        nn.init.uniform_(self.rnn.weight_hh_l0, a=-1.0/np.sqrt(self.hidden_size), b=1.0/np.sqrt(self.hidden_size))
        nn.init.uniform_(self.rnn.weight_ih_l0, a=-1.0 / np.sqrt(self.hidden_size),
                        b=1.0 / np.sqrt(self.hidden_size))
        nn.init.uniform_(self.rnn.bias_hh_l0, a=-1.0 / np.sqrt(self.hidden_size),
                        b=1.0 / np.sqrt(self.hidden_size))
        nn.init.uniform_(self.rnn.bias_ih_l0, a=-1.0 / np.sqrt(self.hidden_size),
                        b=1.0 / np.sqrt(self.hidden_size))

    def forward(self, input):
        embedded_input = self.word_embedding(input)
        # RNN expects a batch
        embedded_input = embedded_input.unsqueeze(1)
        # Note: the hidden state and cell state are 1x1xdim tensor: num layers * num directions x batch_size x dimensionality
        # So we need to unsqueeze to add these 1-dims.
        init_state = (torch.from_numpy(np.zeros(self.hidden_size)).unsqueeze(0).unsqueeze(1).float(),
                      torch.from_numpy(np.zeros(self.hidden_size)).unsqueeze(0).unsqueeze(1).float())
        output, (hidden_state, cell_state) = self.rnn(embedded_input, init_state)
        log_probs = self.log_softmax( self.hidden2(hidden_state) )    
        log_probs_many = self.log_softmax( self.hidden27(output) )    

        return output, hidden_state, cell_state, log_probs, log_probs_many 


# =========================================================================
# PART 1 

class ConsonantVowelClassifier(object):
    def predict(self, context):
        """
        :return: 1 if vowel, 0 if consonant
        """
        raise Exception("Call subclasses")


class FrequencyBasedClassifier(ConsonantVowelClassifier):
    """
    Classifier based on the last letter before the space. If it has occurred with more consonants than vowels,
    classify as consonant, otherwise as vowel.
    """
    def __init__(self, consonant_counts, vowel_counts):
        self.consonant_counts = consonant_counts
        self.vowel_counts = vowel_counts

    def predict(self, context):
        # Look two back to find the letter before the space
        if self.consonant_counts[context[-1]] > self.vowel_counts[context[-1]]:
            return 0
        else:
            return 1


class RNNClassifier(ConsonantVowelClassifier):
    def __init__(self, rnn_module, vocab_index):
        self.rnn_module = rnn_module
        self.vocab_index = vocab_index

    def predict(self, context):         
        tmp = [self.vocab_index.index_of(cc) for cc in context]
        embed_sen = torch.tensor(tmp , dtype=torch.long)        # x = torch.from_numpy(tmp).long()
        output, hidden_state, cell_state, log_probs, log_probs_many = self.rnn_module.forward(embed_sen)         
        prediction = torch.argmax(log_probs)
        return prediction


def train_frequency_based_classifier(cons_exs, vowel_exs):
    consonant_counts = collections.Counter()
    vowel_counts = collections.Counter()
    for ex in cons_exs:
        consonant_counts[ex[-1]] += 1
    for ex in vowel_exs:
        vowel_counts[ex[-1]] += 1
    return FrequencyBasedClassifier(consonant_counts, vowel_counts)


# =========================================================================

def train_rnn_classifier(args, train_cons_exs, train_vowel_exs, dev_cons_exs, dev_vowel_exs, vocab_index):
    """
    :param args: command-line args
    :param train_cons_exs: list of strings followed by consonants
    :param train_vowel_exs: list of strings followed by vowels
    :param dev_cons_exs: list of strings followed by consonants
    :param dev_vowel_exs: list of strings followed by vowels
    :param vocab_index: an Indexer of the character vocabulary (27 characters)
    :return: an RNNClassifier instance trained on the given data
    """
    # args 
    num_epochs = 3 
    learning_rate = 0.001
    num_classes = 2

    rnn_module = RNNOverWords(dict_size=27, input_size=50, hidden_size=128, dropout=0.0)
    optimizer = optim.Adam(rnn_module.parameters(), lr=learning_rate)

    train_exs = train_cons_exs + train_vowel_exs
    train_labels = [0 for i in range(len(train_cons_exs))] + [1 for j in range(len(train_vowel_exs))]
    indices = [i for i in range(len(train_exs))]

    for epoch in range(num_epochs):
        random.shuffle(indices)
        total_loss = 0.0

        for ii, idx in enumerate(indices):
            sen = train_exs[idx]
            tmp = [vocab_index.index_of(cc) for cc in sen]
            embed_sen = torch.tensor(tmp , dtype=torch.long)   

            y = train_labels[idx]
            y_onehot = torch.zeros(num_classes)                 
            y_onehot.scatter_(0, torch.from_numpy( np.asarray(y, dtype=np.int64) ), 1)

            rnn_module.zero_grad()
            output, hidden_state, cell_state, log_probs, log_probs_many = rnn_module.forward(embed_sen)      

            loss = torch.neg(log_probs.squeeze()).dot(y_onehot)    
            total_loss += loss

            loss.backward()
            optimizer.step()
        print("total loss on epoch %i: %f" %(epoch, total_loss))     
    return RNNClassifier(rnn_module, vocab_index)



# =========================================================================
# PART 2 MODELS 


class LanguageModel(object):
    def get_next_char_log_probs(self, context) -> np.ndarray:
        """
        Returns a log probability distribution over the next characters given a context.
        The log should be base e
        :param context: a single character to score
        :return: A numpy vector log P(y | context) where y ranges over the output vocabulary.
        """
        raise Exception("Only implemented in subclasses")


    def get_log_prob_sequence(self, next_chars, context) -> float:
        """
        Scores a bunch of characters following context. That is, returns
        log P(nc1, nc2, nc3, ... | context) = log P(nc1 | context) + log P(nc2 | context, nc1), ...
        The log should be base e
        :param next_chars:
        :param context:
        :return: The float probability
        """
        raise Exception("Only implemented in subclasses")


class UniformLanguageModel(LanguageModel):
    def __init__(self, voc_size):
        self.voc_size = voc_size

    def get_next_char_log_probs(self, context):
        return np.ones([self.voc_size]) * np.log(1.0/self.voc_size)

    def get_log_prob_sequence(self, next_chars, context):
        return np.log(1.0/self.voc_size) * len(next_chars)



# =========================================================================

class RNNLanguageModel(LanguageModel):      
    def __init__(self, rnn_module, vocab_index):
        self.rnn_module = rnn_module
        self.vocab_index = vocab_index 
        
    def get_next_char_log_probs(self, context):
        sen = context   
        tmp = [self.vocab_index.index_of(cc) for cc in sen]
        x = torch.tensor(tmp , dtype=torch.long)    
        output, hidden_state, cell_state, log_probs, log_probs_many = self.rnn_module.forward(x)     

        res = log_probs_many.squeeze(1)[-1]
        return res.detach().numpy()

    def get_log_prob_sequence(self, next_chars, context):
        total_log_prob = 0.0
        for i in range(len(next_chars)):
            log_prob_dist = self.get_next_char_log_probs(context + next_chars[0:i])
            total_log_prob += log_prob_dist[self.vocab_index.index_of(next_chars[i])]

        return total_log_prob    

            


def train_lm(args, train_text, dev_text, vocab_index):      
    """
    :param args: command-line args 
    :param train_text: train text as a sequence of characters
    :param dev_text: dev texts as a sequence of characters
    :param vocab_index: an Indexer of the character vocabulary (27 characters)
    :return: an RNNLanguageModel instance trained on the given data
    """

    num_epochs = 3
    learning_rate = 0.001
    num_classes = 27

    rnn_module = RNNOverWords(dict_size=27, input_size=50, hidden_size=128, dropout=0.0)
    optimizer = optim.Adam(rnn_module.parameters(), lr=learning_rate)

    train_exs = train_text
    chunk_size = 10
    num_chunk = len(train_text) // chunk_size   # num of whole chunk 

    chunks = []
    chunks.append( train_exs[0 : chunk_size-1] )
    for i in range(1, num_chunk):
        chunks.append( train_exs[(i*chunk_size -7) : (i*chunk_size + chunk_size -1)] )  # chunks overlap by 3 or 7
    if num_chunk*chunk_size < len(train_exs):
        chunks.append( train_exs[(num_chunk*chunk_size -3) : len(train_exs)] )     # append last chunk

    indices = [i for i in range(len(chunks))] 

    for epoch in range(num_epochs):
        random.shuffle(indices)
        total_loss = 0.0

        for ii, idx in enumerate(indices):                      # idx is index of the chunk
            sen = " " + chunks[idx]                             # a sentence 
            tmp = [vocab_index.index_of(cc) for cc in sen]      # list of index of char 
            x = torch.tensor(tmp , dtype=torch.long)    

            rnn_module.zero_grad()
            output, hidden_state, cell_state, log_probs, log_probs_many = rnn_module.forward(x)             
            
            tmp2 = [vocab_index.index_of(cc) for cc in chunks[idx]]
            tmp3 = torch.tensor(tmp2 , dtype=torch.long)

            loss = nn.NLLLoss()
            output = loss(log_probs_many.squeeze()[ :-1], tmp3)      
            output.backward()
            optimizer.step()

            total_loss += output  
        print("total loss on epoch %i: %f" %(epoch, total_loss))    
    return RNNLanguageModel(rnn_module, vocab_index)







