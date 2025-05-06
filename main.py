import numpy as np


from data import train_data,test_data
import rnn

all_sentences = list(train_data.keys())
all_words= set(word for sentence in all_sentences for word in sentence.split())
word2index = {word: i for i, word in enumerate(sorted(all_words))}
vocab_size = len(word2index)

def encode_sentence(sentence,word2index):
    vectors = []
    for word in sentence.split():
        vec = np.zeros((len(word2index),1))
        vec[word2index[word]] = 1
        vectors.append(vec)
    return vectors

X_train = [encode_sentence(sent,word2index) for sent in train_data]
y_train = [int(train_data[sent]) for sent in train_data]

print(type(X_train[0][0]))         # <class 'numpy.ndarray'>
print(X_train[0][0].shape)         # (vocab_size, 1)


model = rnn.RNN(input_size=vocab_size, hidden_size=16, output_size=1)
model.train(X_train, y_train, epochs=20)



#test
X_test = [encode_sentence(sent, word2index) for sent in test_data]
y_test = [int(test_data[sent]) for sent in test_data]

correct = 0
for i, x in enumerate(X_test):
    pred = model.predict(x)
    real = y_test[i]
    print(f"'{list(test_data.keys())[i]}': Tahmin = {pred}, Gerçek = {real}")
    if pred == real:
        correct += 1

print(f"\nTest doğruluğu: {correct}/{len(X_test)} ({correct/len(X_test):.2%})")


#2. model
import torch
import torch.nn as nn
import torch.optim as optim
import Model2

# Tek bir cümleyi encode edip modele vereceğimiz tensor:
def sentence_to_tensor(sentence, word2index):
    encoded = encode_sentence(sentence, word2index)
    tensor = torch.stack([
        torch.tensor(vec, dtype=torch.float32).squeeze(1) for vec in encoded
    ])
    return tensor.unsqueeze(0)  # [1, seq_len, input_size]

for sent in train_data:
    input_tensor = sentence_to_tensor(sent, word2index)
    print(input_tensor.shape)  # Her biri farklı uzunlukta olabilir, bu normal


model = Model2.TorchRNN(input_size=vocab_size, hidden_size=16, output_size=1)
output = model(input_tensor)  # output shape: [1, 1]
prediction = 1 if output.item() > 0.5 else 0
train_losses = Model2.TorchRNN.train_torch_rnn(model, X_train, y_train, epochs=20)