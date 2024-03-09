import torch

normal_cls_loss = torch.tensor(0.0, requires_grad = True)
print(normal_cls_loss.shape)

a = torch.randn((3,4,5))
a = a.mean()
print(a.shape)