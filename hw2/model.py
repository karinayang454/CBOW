import torch.nn as nn
import torch
import torch.nn.functional as F

class CBOW(nn.Module):

    def __init__(self, vocab_size, embedding_dim, window):
        super(CBOW, self).__init__()

        self.embed = nn.Embedding(vocab_size, embedding_dim)
        # self.linear1 = nn.Linear(embedding_dim, 32)
        # self.activation_function1 = nn.ReLU()
        
        self.linear2 = nn.Linear(embedding_dim, vocab_size)
        # self.activation_function2 = nn.LogSoftmax(dim = -1)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        

    def forward(self, inputs):
        embeds = torch.sum(self.embed(inputs), dim=1)
        out = self.linear2(embeds)
        # out = self.activation_function2(out)
        # print(out)
        return out

    # def __init__(self, vocab_size, embedding_dim, context_size):
    #     super(CBOW, self).__init__()
    #     self.embeddings = nn.Embedding(vocab_size, embedding_dim)
    #     self.linear1 = nn.Linear(context_size * embedding_dim, 128)
    #     self.linear2 = nn.Linear(128, vocab_size)

    # def forward(self, inputs):
    #     embeds = self.embeddings(inputs).view((1, -1))  # -1 implies size inferred for that index from the size of the data
    #     #print(np.mean(np.mean(self.linear2.weight.data.numpy())))
    #     out1 = F.relu(self.linear1(embeds)) # output of first layer
    #     out2 = self.linear2(out1)           # output of second layer
    #     #print(embeds)
    #     log_probs = F.log_softmax(out2, dim=1)
    #     return log_probs

    # def __init__(self, vocab_size, embedding_dim, window_size):
    #     super(CBOW, self).__init__()
    #     self.embeddings = nn.Embedding(vocab_size, embedding_dim)
    #     self.linear = nn.Linear(embedding_dim, vocab_size)
    #     self.window_size = window_size

    # def forward(self, inputs):

    #     embeds = torch.sum(self.embeddings(inputs), dim=1) # [200, 4, 50] => [200, 50]
    #     # embeds = self.embeddings(inputs).view((batch_size, -1))
    #     out = self.linear(embeds) # nonlinear + projection
    #     log_probs = F.log_softmax(out, dim=1) # softmax compute log probability

    #     return log_probs