from tqdm import tqdm
import numpy as np
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, CIFAR10
from torchvision import transforms
from torchvision.utils import save_image, make_grid
import os
import matplotlib.pyplot as plt
from diffusers import UNet2DModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

### naming consistent with the paper
batch_size = 2
s_0 = 2
s_1 = 200
mu_0 = 0.95
sigma_max = 80.0
sigma_min = 0.002
rho = 7.0
num_step_test = 41


### define network architecture
def get_model(n_channel=3):
    # DownBlock2D: a regular ResNet downsampling block
    # AttnDownBlock2D: a ResNet downsampling block with spatial self-attention
    if n_channel == 1:
        block_out_channels=(128, 128, 256, 256, 512)
        down_block_types=( "DownBlock2D", "DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D",)
        up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D")
    elif n_channel == 3:
        block_out_channels=(128, 256, 256, 256)
        down_block_types=("DownBlock2D", "AttnDownBlock2D", "DownBlock2D", "DownBlock2D")
        up_block_types=("UpBlock2D", "UpBlock2D", "AttnUpBlock2D", "UpBlock2D")
    else:
        raise ValueError(f"Unknown n_channel: {n_channel}")
    
    model = UNet2DModel(sample_size=32, in_channels=n_channel, out_channels=n_channel, layers_per_block=2, 
                        block_out_channels=block_out_channels, down_block_types=down_block_types, up_block_types=up_block_types)
    return model


class UnetModel(nn.Module):
    def __init__(self, n_channel) -> None:
        super(UnetModel, self).__init__()
        self.unet2d = get_model(n_channel=n_channel)

    def forward(self, x, t) -> torch.Tensor:
        x_ori = x
        x = self.unet2d(x, t[:, 0]).sample
        ### from https://arxiv.org/pdf/2303.01469 Appendix C
        sigma = t
        sigma_data = 0.5
        sigma_min = 0.002
        c_skip_t = (sigma_data ** 2) / ((sigma - sigma_min) ** 2 + sigma_data ** 2)
        c_out_t = (sigma - sigma_min) * sigma_data / (sigma ** 2 + sigma_data**2) ** 0.5
        return c_skip_t[:, :, None, None] * x_ori + c_out_t[:, :, None, None] * x


### define datasets
def mnist_dl():
    tf = transforms.Compose([transforms.Pad(2), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5)),])
    dataset = MNIST("./data", train=True, download=True, transform=tf,)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    return dataset, dataloader


def cifar10_dl():
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),])
    dataset = CIFAR10("./data", train=True, download=True, transform=tf,)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    return dataset, dataloader


### from official implementation
@torch.no_grad()
def euler_solver(samples, t, next_t, x0):
    x = samples
    denoiser = x0
    d = (x - denoiser) / t[:, :, None, None]
    samples = x + d * (next_t - t)[:, :, None, None]
    return samples


def train(n_channels=1, name="mnist", n_epoch=50):
    
    if name == "cifar10":
        dataset, dataloader = cifar10_dl()
    elif name == "mnist":
        dataset, dataloader = mnist_dl()
    else:
        raise ValueError(f"Unknown dataset: {name}")
    
    output_dir = "./results/{:}".format(name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    model = UnetModel(n_channels)
    model.to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Define \theta_{-}, which is EMA of the params
    ema_model = UnetModel(n_channels)
    ema_model.to(device)
    ema_model.load_state_dict(model.state_dict())

    n_iter = n_epoch * (len(dataset) // batch_size)
    iter = 0
    print('dataset:', name, len(dataset), batch_size, n_iter)
    
    for epoch in tqdm(range(1, n_epoch+1)):

        pbar = tqdm(dataloader)
        model.train()
        for x, _ in pbar:
            iter += 1
            ### from https://arxiv.org/pdf/2303.01469 Appendix C
            N_k = np.ceil(np.sqrt((iter / n_iter) * ((s_1 + 1) ** 2 - s_0**2) + s_0**2) - 1).astype(np.int32)

            optim.zero_grad()
            x = x.to(device)

            noise = torch.randn_like(x)

            ### karras scheduler
            indices = torch.randint(0, N_k - 1, (x.shape[0], 1), device=device)

            t = sigma_max ** (1 / rho) + indices / (N_k - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
            t = t**rho

            t2 = sigma_max**(1 / rho)+(indices+1)/ (N_k - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
            t2 = t2**rho

            ### define loss function
            x_t = x + noise * t[:, :, None, None]
            x1 = model(x_t, t)
            x_t2 = euler_solver(x_t, t, t2, x).detach()
            with torch.no_grad():
                x2 = ema_model(x_t2, t2)

            loss = F.mse_loss(x1, x2)
            # print('mse_loss:', loss.item())
            
            loss.backward()

            optim.step()

            with torch.no_grad():
                ### from https://arxiv.org/pdf/2303.01469 Appendix C
                mu_k = np.exp(s_0 * np.log(mu_0) / N_k)
                ### update \theta_{-} (teacher model)
                for p, ema_p in zip(model.parameters(), ema_model.parameters()):
                    ema_p.mul_(mu_k).add_(p, alpha=1 - mu_k)

            pbar.set_description(f"loss: {loss:.8f}, mu: {mu_k:.8f}")

            # break

        model.eval()
        with torch.no_grad():
            t_max = sigma_max
            t_min = sigma_min
            t_max_rho = t_max ** (1 / rho)
            t_min_rho = t_min ** (1 / rho)

            test_steps = [5, 10, 20]

            ### multi-step consitency sampling, from official implementation
            for num_step in test_steps:
                xh = torch.randn(64, x.shape[-3], x.shape[-2], x.shape[-1]).float().to(device) * sigma_max
                ts = np.linspace(0, num_step_test-1, num_step+1).astype(np.int32)
                
                for i in range(len(ts) - 1):
                    t = (t_max_rho + ts[i] / (num_step_test - 1) * (t_min_rho - t_max_rho)) ** rho
                    t = torch.tensor([t]).unsqueeze(0).float().to(device)
                    x0 = model(xh, t)
                    # x0 = torch.clamp(x0, -1, 1)     # clip_denoised: is it needed
                    next_t = (t_max_rho + ts[i + 1] / (num_step_test - 1) * (t_min_rho - t_max_rho)) ** rho
                    next_t = np.clip(next_t, t_min, t_max)
                    xh = x0 + torch.randn(64, x.shape[-3], x.shape[-2], x.shape[-1]).float().to(device) * np.sqrt(next_t**2 - t_min**2)
                xh = (xh * 0.5 + 0.5).clamp(0, 1)
                grid = make_grid(xh, nrow=8)
                save_image(grid, "./{:}/ct_{:}_sample_{:02d}step.png".format(output_dir, name, num_step))


if __name__ == "__main__":
    train(n_channels=1, name="mnist", n_epoch=50)
    train(n_channels=3, name="cifar10", n_epoch=200)
    
