import os
import torch

query = torch.randn(1,8*8, 1280)
b, pix_num, dim = query.shape
res = int(pix_num ** 0.5)
query_2d = query.permute(0,2,1).reshape(b,dim,res,res)
print(query_2d.shape)

