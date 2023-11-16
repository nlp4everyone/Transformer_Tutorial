import torch
# Define parameter
embedding_space = 100
# Sequence length
sequence_length = 10

# Define vector Q,K,V
vector_Q = torch.randn(size=(sequence_length,embedding_space))
vector_K = torch.randn(size=(sequence_length,embedding_space))
vector_V = torch.randn(size=(sequence_length,embedding_space))

output = torch.matmul(vector_Q,vector_K.T)
# Calculate mask
mask = torch.ones_like(output)
mask =  torch.tril(mask)
mask[mask==0] = -torch.inf
mask[mask==1] = 0

# Calculate output
output = (output+mask)/torch.sqrt(torch.tensor(embedding_space))

# Self attention
# self-attention = softmax(Q.K.T/sqrt(dk))*V
self_attention = torch.softmax(output,dim=1)
self_attention = torch.matmul(self_attention,vector_V)
print(self_attention)