import torch
import os
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.allow_tf32 = True
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

data = torch.randn([2, 48, 128, 128], dtype=torch.float, device='cuda', requires_grad=True)
data2 = torch.randn([2, 256, 128, 128], dtype=torch.float, device='cuda', requires_grad=True)
data3 = torch.cat((data, data2), dim=1)
net = torch.nn.Conv2d(304, 256, kernel_size=[3, 3], padding=[1, 1], stride=[1, 1], dilation=[1, 1], groups=1)
net = net.cuda().float()
out = net(data)
out.backward(torch.randn_like(out))
torch.cuda.synchronize()