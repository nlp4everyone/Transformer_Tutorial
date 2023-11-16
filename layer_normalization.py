import torch
torch.random.manual_seed(5)
weight = torch.randn(size=(2,100))
# Before normalization
mean = torch.mean(weight,dim=1)
std = torch.std(weight,dim=1)
print(f"Mean of distribution is: {mean}")
print(f"Standard Deviation of distribution is: {std}")

print("\n")
# After Normalization
# new_weight = (weight-mean)/std
# new_mean = torch.mean(new_weight,dim=1)
# new_std = torch.std(new_weight,dim=1)
new_weight = torch.sub(weight,mean.reshape(2,-1))
new_weight = torch.div(new_weight,std.reshape(2,-1))
new_mean = torch.mean(new_weight,dim=1)
new_std = torch.std(new_weight,dim=1)
print(f"Mean of distribution is: {new_mean}")
print(f"Standard Deviation of distribution is: {new_std}")
