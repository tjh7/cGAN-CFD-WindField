

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from tqdm import tqdm

from config import Config
from my_dataset import WindDataset
from model import UNetGenerator, Discriminator

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def denormalize(tensor):
    return tensor.clamp(0, 1)


class ExtremeWeightedL1Loss(nn.Module):
    def __init__(self, threshold=15.0, alpha=5.0):

        super().__init__()
        self.threshold = threshold
        self.alpha = alpha

    def forward(self, pred, target):
        abs_error = torch.abs(pred - target)
        weights = torch.ones_like(target)
        weights[target > self.threshold] = self.alpha
        loss = (abs_error * weights).mean()
        return loss


def train():
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    dataset = WindDataset(Config.CONDITION_CSV, image_size=Config.IMAGE_SIZE)
    dataloader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True)

    G = UNetGenerator(condition_dim=4).to(device)
    D = Discriminator(in_channels=3 + 1).to(device)

    criterion_GAN = nn.BCEWithLogitsLoss()
    criterion_L1 = ExtremeWeightedL1Loss(threshold=15.0, alpha=5.0)  # 使用极值加权 L1

    optimizer_G = optim.Adam(G.parameters(), lr=Config.LEARNING_RATE, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(D.parameters(), lr=Config.LEARNING_RATE, betas=(0.5, 0.999))

    train_losses_G, train_losses_D, train_losses_L1 = [], [], []

    for epoch in range(1, Config.EPOCHS + 1):
        G.train()
        D.train() 
        epoch_loss_G, epoch_loss_D, epoch_loss_L1 = 0.0, 0.0, 0.0

        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch}/{Config.EPOCHS}")):
            building = batch['building'].to(device)
            real_windfield = batch['windfield'].to(device)
            condition = batch['condition'].to(device)


            real_input_D = torch.cat([real_windfield, building], dim=1)
            output_real = D(real_input_D)
            label_real = torch.ones_like(output_real, device=device)

            optimizer_D.zero_grad()
            loss_D_real = criterion_GAN(output_real, label_real)

            fake_windfield = G(building, condition)
            fake_input_D = torch.cat([fake_windfield.detach(), building], dim=1)
            output_fake = D(fake_input_D)
            label_fake = torch.zeros_like(output_fake, device=device)

            loss_D_fake = criterion_GAN(output_fake, label_fake)
            loss_D = (loss_D_real + loss_D_fake) * 0.5
            loss_D.backward()
            optimizer_D.step()


            optimizer_G.zero_grad()
            fake_input_D_for_G = torch.cat([fake_windfield, building], dim=1)
            output_fake_for_G = D(fake_input_D_for_G)
            label_real_for_G = torch.ones_like(output_fake_for_G, device=device)

            loss_G_GAN = criterion_GAN(output_fake_for_G, label_real_for_G)
            loss_G_L1 = criterion_L1(fake_windfield, real_windfield) * Config.L1_LAMBDA
            loss_G = loss_G_GAN + loss_G_L1
            loss_G.backward()
            optimizer_G.step()

            epoch_loss_D += loss_D.item()
            epoch_loss_G += loss_G.item()
            epoch_loss_L1 += loss_G_L1.item()

        avg_loss_D = epoch_loss_D / len(dataloader)
        avg_loss_G = epoch_loss_G / len(dataloader)
        avg_loss_L1 = epoch_loss_L1 / len(dataloader)

        train_losses_D.append(avg_loss_D)
        train_losses_G.append(avg_loss_G)
        train_losses_L1.append(avg_loss_L1)

        print(f"Epoch {epoch} - D_loss: {avg_loss_D:.6f}, G_loss: {avg_loss_G:.6f}, L1_loss: {avg_loss_L1:.6f}")


        if epoch % 100 == 0:
            G.eval()
            with torch.no_grad():
                sample = next(iter(dataloader))
                building = sample['building'].to(device)
                windfield = sample['windfield'].to(device)
                condition = sample['condition'].to(device)

                pred = G(building, condition)
                building_denorm = denormalize(building)
                windfield_denorm = denormalize(windfield)
                pred_denorm = denormalize(pred)

                save_path = Config.SAMPLE_SAVE_PATH
                os.makedirs(save_path, exist_ok=True)
                save_image(building_denorm, os.path.join(save_path, f"building_epoch_{epoch}.png"),
                           nrow=Config.BATCH_SIZE, padding=2)
                save_image(windfield_denorm, os.path.join(save_path, f"windfield_epoch_{epoch}.png"),
                           nrow=Config.BATCH_SIZE, padding=2)
                save_image(pred_denorm, os.path.join(save_path, f"predicted_epoch_{epoch}.png"),
                           nrow=Config.BATCH_SIZE, padding=2)

        # 保存模型
        if epoch % 100 == 0:
            checkpoint_dir = r'D:\gan\xin\taifeng_qiangduiliu\checkpoint_90'                         
            os.makedirs(checkpoint_dir, exist_ok=True)
            g_path = os.path.join(checkpoint_dir, f"generator_epoch_{epoch}.pth")
            d_path = os.path.join(checkpoint_dir, f"discriminator_epoch_{epoch}.pth")
            torch.save(G.state_dict(), g_path)
            torch.save(D.state_dict(), d_path)
            print(f"模型已保存: {g_path} 和 {d_path}")

    total_time = time.time() - start_time
    print(f"训练完成，总耗时: {total_time / 60:.2f} 分钟")

    # 绘制 Loss 曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, Config.EPOCHS + 1), train_losses_D, label='Discriminator Loss')
    plt.plot(range(1, Config.EPOCHS + 1), train_losses_G, label='Generator Loss')
    plt.plot(range(1, Config.EPOCHS + 1), train_losses_L1, label='L1 Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.grid(True)
    plt.legend()

    loss_plot_path = Config.LOSS_SAVE_PATH
    os.makedirs(loss_plot_path, exist_ok=True)
    plt.savefig(os.path.join(loss_plot_path, 'loss_curve_MaxAE_90_20000.png'), dpi=600, bbox_inches='tight')          
    plt.close()
    print("Loss 曲线已保存")


if __name__ == "__main__":
    train()
