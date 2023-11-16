# Multi head:
# Attention = softmax(Q.K.T/sqrt(embedding_size)*V

import torch
import torch.nn as nn
embedding_size = 1024
input_size = 512
batch_size = 32
number_of_head = 8
sequence_length = 20

class Multihead_Attention(nn.Module):
    def __init__(self,embedding_size,input_size,number_of_heads,apply_mask = True):
        super().__init__()
        self.embedding_size = embedding_size
        self.input_size = input_size
        self.number_of_heads = number_of_heads
        self.linear_layer0 = nn.Linear(input_size, embedding_size*3)
        self.linear_layer1 = nn.Linear(embedding_size,embedding_size)
        self.apply_mask = apply_mask

    def forward(self,x):
        # Get size
        batch_size,sequence_length = x.size()[0],x.size()[1]
        # Create Q K V Vector by multiplying by 3
        x = self.linear_layer0(x) # Shape: batch_size * sequence_lengt *,embedding_size*3
        # Embedding multi-head
        x = torch.reshape(x,shape=(batch_size,sequence_length,self.number_of_heads,-1))
        # Shape: batch_size * sequence_length * number_of_head * embedding_size*3/num_head

        # Change the shape
        x = torch.permute(x,dims=(0,2,1,-1))
        # Shape: batch_size  * number_of_head * sequence_length * embedding_size*3/num_head

        # Apply attention
        attention,x = self.attention_mechanism(x,self.apply_mask)

        # New value
        x = torch.reshape(x,shape=(batch_size,sequence_length,-1))   
        return x

    def attention_mechanism(self,x,mask):
        # Split to Q,K,V Vector
        Q_vector, K_vector, V_vector = torch.chunk(x, chunks=3, dim=-1)
        # Compute attention
        scaled = torch.matmul(Q_vector,torch.transpose(K_vector,-2,-1))

        # Reduce the variable
        scaled /= torch.sqrt(torch.tensor(self.input_size))

        # Apply mask if True
        if mask:
            # Initilize with -inf value
            mask = torch.full(scaled.size(),fill_value=-torch.inf)
            mask = torch.triu(mask,diagonal=1)

            # Calculate scale
            scaled += mask
        attention = torch.softmax(scaled,dim=-1)
        new_v = torch.matmul(attention,V_vector)
        return attention,new_v

# Input shape: batch_size * sequence_length * input_size
input_vector = torch.randn(size=(batch_size,sequence_length,input_size))
attention = Multihead_Attention(embedding_size,input_size,number_of_head,apply_mask=False)
output = attention(input_vector)
