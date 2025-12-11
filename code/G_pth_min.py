
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from my_dataset import WindDataset
from model import UNetGenerator, Discriminator
from config import Config


class ExtremeWeightedL1Loss(nn.Module):
    def __init__(self, threshold=15.0, alpha=5.0):
        """
        threshold: 超过该风速值就认为是极值 (m/s)
        alpha: 极值点的权重系数
        """
        super().__init__()
        self.threshold = threshold
        self.alpha = alpha

    def forward(self, pred, target):
        abs_error = torch.abs(pred - target)
        weights = torch.ones_like(target)
        weights[target > self.threshold] = self.alpha
        loss = (abs_error * weights).mean()
        return loss

def get_dataloader():
    dataset = WindDataset(Config.CONDITION_CSV, image_size=Config.IMAGE_SIZE)
    return DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=False)


@torch.no_grad()
def evaluate_generator(g_path, d_path, dataloader, device):
 
    G = UNetGenerator(condition_dim=4).to(device)
    D = Discriminator(in_channels=3 + 1).to(device)
    G.load_state_dict(torch.load(g_path, map_location=device))
    D.load_state_dict(torch.load(d_path, map_location=device))
    G.eval()
    D.eval()

    criterion_GAN = nn.BCEWithLogitsLoss()
    criterion_L1_weighted = ExtremeWeightedL1Loss(threshold=15.0, alpha=5.0)  # 与训练一致
    criterion_L1_raw = nn.L1Loss() 

    total_L1_plot = 0.0
    total_G = 0.0
    total_raw_mae = 0.0
    count = 0

    for batch in dataloader:
        building = batch['building'].to(device)
        real_windfield = batch['windfield'].to(device)
        condition = batch['condition'].to(device)

        fake_windfield = G(building, condition)

      
        fake_input_D_for_G = torch.cat([fake_windfield, building], dim=1)
        output_fake_for_G = D(fake_input_D_for_G)
        label_real_for_G = torch.ones_like(output_fake_for_G, device=device)

        loss_G_GAN = criterion_GAN(output_fake_for_G, label_real_for_G)
    
        loss_G_L1_weighted = criterion_L1_weighted(fake_windfield, real_windfield) * Config.L1_LAMBDA
        loss_G = loss_G_GAN + loss_G_L1_weighted

  
        raw_mae = criterion_L1_raw(fake_windfield, real_windfield)

        total_L1_plot += loss_G_L1_weighted.item()
        total_G += loss_G.item()
        total_raw_mae += raw_mae.item()
        count += 1

    avg_L1_plot_loss = total_L1_plot / count
    avg_G_loss = total_G / count
    avg_raw_mae = total_raw_mae / count

    return avg_L1_plot_loss, avg_G_loss, avg_raw_mae



def find_best_generator(checkpoint_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloader = get_dataloader()

    best_metric = float("inf")
    best_g_path = None
    best_d_path = None
    records = {}  # epoch -> dict(metrics)

    print("\n====== 按与训练曲线一致的 L1 Loss 选最优 checkpoint ======\n")

    for fname in os.listdir(checkpoint_dir):
        if fname.startswith("generator_epoch") and fname.endswith(".pth"):
            epoch = fname.split("_")[-1].split(".")[0]
            g_path = os.path.join(checkpoint_dir, fname)
            d_path = os.path.join(checkpoint_dir, f"discriminator_epoch_{epoch}.pth")
            if not os.path.exists(d_path):
                continue

            L1_plot_loss, G_loss, raw_mae = evaluate_generator(g_path, d_path, dataloader, device)
            records[int(epoch)] = {
                "L1_plot_loss": L1_plot_loss,
                "G_loss": G_loss,
                "raw_mae": raw_mae,
                "g_path": g_path,
                "d_path": d_path
            }

            print(f"Epoch {epoch}:  L1_plot={L1_plot_loss:.6f}  |  G_loss={G_loss:.6f}  |  raw_MAE={raw_mae:.6f}")

            if L1_plot_loss < best_metric:
                best_metric = L1_plot_loss
                best_g_path = g_path
                best_d_path = d_path

  
    print("\n====== 最优模型（按曲线 L1 Loss 选取） ======")
    print(f"Generator Path     : {best_g_path}")
    print(f"Discriminator Path : {best_d_path}")
    print(f"最小 L1_plot Loss  : {best_metric:.6f}\n")

  
    top = sorted(records.items(), key=lambda kv: kv[1]["L1_plot_loss"])[:5]
    if top:
        print("Top-5 (by L1_plot):")
        for ep, rec in top:
            print(f"  Epoch {ep:>5}: L1_plot={rec['L1_plot_loss']:.6f} | G_loss={rec['G_loss']:.6f} | raw_MAE={rec['raw_mae']:.6f}")

    return best_g_path, best_d_path, records



if __name__ == "__main__":
    checkpoint_dir = Config.CHECKPOINT_DIR
    best_g_path, best_d_path, all_records = find_best_generator(checkpoint_dir)
