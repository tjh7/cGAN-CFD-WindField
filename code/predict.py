
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from PIL import Image
import torchvision.transforms as transforms
import time
from model import UNetGenerator
from config import Config
from skimage.metrics import structural_similarity as ssim
import matplotlib.colors as mcolors
import matplotlib.cm as cm


def predict(building_path, condition, cfd_path, save_path=r'D:\gan\预测_\pix2pix_pred.png',
            vmax=20.0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # ================== 加载模型 ==================
    model = UNetGenerator().to(device)
    model_path = r"D:\gan\xin\taifeng_qiangduiliu\checkpoint_90/generator_epoch_11000.pth"               #####
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
        transforms.ToTensor()
    ])

    # ================== 输入准备 ==================
    building = transform(Image.open(building_path).convert('L')).unsqueeze(0).to(device)
    cfd = transform(Image.open(cfd_path).convert('RGB')).unsqueeze(0).to(device)
    condition_tensor = torch.tensor([condition], dtype=torch.float32).to(device)

    # ================== 预测 ==================
    with torch.no_grad():
        start_time = time.time()
        output = model(building, condition_tensor)
        elapsed = time.time() - start_time
        print(f"预测耗时: {elapsed:.4f} 秒")

        save_image(output.cpu(), save_path)
        print(f"预测完成，结果已保存为 {save_path}")

        # ================== 误差计算 ==================
        output_ms = output * vmax
        cfd_ms = cfd * vmax

        abs_error = torch.abs(output_ms - cfd_ms).mean(dim=1, keepdim=True)
        mae = F.l1_loss(output_ms, cfd_ms).item()
        rmse = torch.sqrt(F.mse_loss(output_ms, cfd_ms)).item()

        abs_error_np = abs_error.squeeze().cpu().numpy()
        print(f"MAE: {mae:.3f} m/s, RMSE: {rmse:.3f} m/s")

        # ================== SSIM计算 ==================
        output_np = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
        cfd_np = cfd.squeeze(0).permute(1, 2, 0).cpu().numpy()

        try:
            ssim_mean, ssim_map = ssim(
                cfd_np, output_np,
                channel_axis=-1,
                data_range=1.0,
                full=True,
                win_size=7
            )
        except Exception as e:
            print(f"⚠️ 多通道SSIM失败 ({e})，自动降为灰度模式。")
            cfd_gray = np.mean(cfd_np, axis=-1)
            output_gray = np.mean(output_np, axis=-1)
            ssim_mean, ssim_map = ssim(
                cfd_gray, output_gray,
                data_range=1.0,
                full=True,
                win_size=7
            )

        if isinstance(ssim_mean, np.ndarray):
            ssim_mean = float(np.mean(ssim_mean))

        print(f"Mean SSIM: {ssim_mean:.3f}")

        # ================== 绘制SSIM分布图 ==================
        plt.figure(figsize=(6, 6))
        plt.imshow(ssim_map, cmap='RdYlBu_r', vmin=0, vmax=1)
        plt.colorbar(label="SSIM", fraction=0.046, pad=0.04)
        plt.axis("off")
        plt.text(10, 10, f"Mean SSIM={ssim_mean:.3f}", color='black', fontsize=12,
                  bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))

        ssim_save_path = save_path.replace(".png", "_SSIM_map.png")
        plt.savefig(ssim_save_path, dpi=600, bbox_inches='tight')
        plt.close()

        # ================== 绘制误差直方图 ==================
        bins = np.linspace(0, 12, 50)
        hist, bin_edges = np.histogram(abs_error_np.flatten(), bins=bins, density=True)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        bar_colors = []
        for left, right in zip(bin_edges[:-1], bin_edges[1:]):
            if right <= 3:
                bar_colors.append((112/255, 163/255, 196/255))
            elif right <= 10:
                bar_colors.append((199/255, 229/255, 236/255))
            else:
                bar_colors.append((245/255, 180/255, 111/255))

        plt.figure(figsize=(6, 6))
        plt.axvspan(0, 3, facecolor=(112/255, 163/255, 196/255, 0.2))
        plt.axvspan(3, 10, facecolor=(199/255, 229/255, 236/255, 0.2))
        plt.axvspan(10, 12, facecolor=(245/255, 180/255, 111/255, 0.2))
        plt.bar(bin_centers, hist, width=bin_edges[1]-bin_edges[0], color=bar_colors,
                edgecolor='black', alpha=0.8)
        plt.xlim(0, 12)
        plt.xlabel("Absolute Error (m/s)")
        plt.ylabel("Proportion")

        total_count = abs_error_np.size
        percent_0_3 = np.sum((abs_error_np >= 0) & (abs_error_np <= 3)) / total_count * 100
        percent_3_10 = np.sum((abs_error_np > 3) & (abs_error_np <= 10)) / total_count * 100
        percent_10_up = np.sum(abs_error_np > 10) / total_count * 100

        plt.text(1.5, max(hist)*0.9, f"{percent_0_3:.1f}%", ha='center')
        plt.text(6.5, max(hist)*0.9, f"{percent_3_10:.1f}%", ha='center')
        plt.text(11, max(hist)*0.9, f"{percent_10_up:.1f}%", ha='center')

        hist_save_path = save_path.replace(".png", "_error_hist.png")
        plt.savefig(hist_save_path, dpi=600, bbox_inches='tight')
        plt.close()

        # ================== AE分段填色 ==================
        plt.figure(figsize=(6, 6))
        bounds = [0, 2, 4, 6, 8, 10, np.max(abs_error_np)]
        n_bounds = len(bounds) - 1
        base_cmap = cm.get_cmap("RdYlBu_r", n_bounds)
        colors = base_cmap(range(n_bounds))
        cmap = mcolors.ListedColormap(colors)
        norm = mcolors.BoundaryNorm(bounds, cmap.N)

        im = plt.imshow(abs_error_np, cmap=cmap, norm=norm)
        plt.axis("off")
        cbar = plt.colorbar(im, boundaries=bounds, fraction=0.046, pad=0.04, extend='both')
        cbar.set_label("Absolute Error (m/s)")
        cbar.ax.tick_params(labelsize=12)
        cbar.set_ticks(bounds[:-1])
        cbar.set_ticklabels([str(b) for b in bounds[:-1]])

        ae_save_path = save_path.replace(".png", "_AE_map.png")
        plt.savefig(ae_save_path, dpi=600, bbox_inches='tight')
        plt.close()


        return mae, rmse, ssim_mean
    
if __name__ == "__main__":

    results = []  # 用于保存所有结果

    r1 = predict(
        building_path=r"D:/gan/xin/buildings/10m.png",
        condition=[-19.49, 4.77, -0.79, 135], 
        cfd_path=r"D:\gan\xin\taifeng_qiangduiliu\cfd\t10_southeast.png",
        save_path=r"D:\gan\xin\taifeng_qiangduiliu\图片3\10m_135_11000_t.png"
    )
    results.append(("135°",) + r1)

    r2 = predict(
        building_path=r"D:/gan/xin/buildings/10m.png",
        condition=[-19.49, 4.77, -0.79, 180], 
        cfd_path=r"D:\gan\xin\taifeng_qiangduiliu\cfd\t10_south.png",
        save_path=r"D:\gan\xin\taifeng_qiangduiliu\图片3\10m_180_11000_t.png"
    )
    results.append(("180°",) + r2)

    r3 = predict(
        building_path=r"D:/gan/xin/buildings/10m.png",
        condition=[-19.49, 4.77, -0.79, 315], 
        cfd_path=r"D:\gan\xin\taifeng_qiangduiliu\cfd\t10_northwest.png",
        save_path=r"D:\gan\xin\taifeng_qiangduiliu\图片3\10m_315_11000_t.png"
    )
    results.append(("315°",) + r3)

    # ================== 输出每个方向的指标 ==================
    print("\n====== 各入流方向误差指标 ======")
    for direction, mae, rmse, ssim_val in results:
        print(f"{direction}: MAE={mae:.3f}, RMSE={rmse:.3f}, SSIM={ssim_val:.3f}")

    # ================== 计算平均 ==================
    maes = [r[1] for r in results]
    rmses = [r[2] for r in results]
    ssims = [r[3] for r in results]

    print("\n====== 平均误差指标 ======")
    print(f"平均 MAE:  {np.mean(maes):.3f}")
    print(f"平均 RMSE: {np.mean(rmses):.3f}")
    print(f"平均 SSIM: {np.mean(ssims):.3f}")
    
    #强对流
if __name__ == "__main__":

    results = []  # 用于保存所有结果

    r1 = predict(
        building_path=r"D:/gan/xin/buildings/10m.png",
        condition=[-16.34, 4.77, -16.34, 135],
        cfd_path=r"D:\gan\xin\taifeng_qiangduiliu\cfd\10_southeast.png",
        save_path=r"D:\gan\xin\taifeng_qiangduiliu\图片3\10m_135_11000_q.png"
    )
    results.append(("135°",) + r1)

    r2 = predict(
        building_path=r"D:/gan/xin/buildings/10m.png",
        condition=[-16.34, 4.77, -16.34, 180],
        cfd_path=r"D:\gan\xin\taifeng_qiangduiliu\cfd\10_south.png",
        save_path=r"D:\gan\xin\taifeng_qiangduiliu\图片3\10m_180_11000_q.png"
    )
    results.append(("180°",) + r2)

    r3 = predict(
        building_path=r"D:/gan/xin/buildings/10m.png",
        condition=[-16.34, 4.77, -16.34, 315],
        cfd_path=r"D:\gan\xin\taifeng_qiangduiliu\cfd\10_northwest.png",
        save_path=r"D:\gan\xin\taifeng_qiangduiliu\图片3\10m_315_11000 _q.png"
    )
    results.append(("315°",) + r3)

    # ================== 输出每个方向的指标 ==================
    print("\n====== 各入流方向误差指标 ======")
    for direction, mae, rmse, ssim_val in results:
        print(f"{direction}: MAE={mae:.3f}, RMSE={rmse:.3f}, SSIM={ssim_val:.3f}")

    # ================== 计算平均 ==================
    maes = [r[1] for r in results]
    rmses = [r[2] for r in results]
    ssims = [r[3] for r in results]

    print("\n====== 平均误差指标 ======")
    print(f"平均 MAE:  {np.mean(maes):.3f}")
    print(f"平均 RMSE: {np.mean(rmses):.3f}")
    print(f"平均 SSIM: {np.mean(ssims):.3f}")




# #        condition=[-19.49, 4.77, -0.79, 135],  #台风
# #        condition=[-16.34, 4.77, -16.34, 135],  #强对流





































# ###加入风速横截面
# import matplotlib.pyplot as plt
# import numpy as np
# import torch
# import torch.nn.functional as F
# from torchvision.utils import save_image
# from PIL import Image
# import torchvision.transforms as transforms
# import time
# from model import UNetGenerator
# from config import Config
# from skimage.metrics import structural_similarity as ssim
# import matplotlib.colors as mcolors
# import matplotlib.cm as cm
# import os


# def predict(building_path, condition, cfd_path, save_path=r'D:\gan\预测_\pix2pix_pred.png',
#             vmax=20.0):
#     """
#     进行风场预测与误差评估（含 MAE / RMSE / SSIM / AE分布 / 平均截面风速对比）
#     """
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print(f"使用设备: {device}")

#     # ================== 加载模型 ==================
#     model = UNetGenerator().to(device)
#     model_path = r"D:\gan\xin\taifeng_qiangduiliu\checkpoint_90\generator_epoch_11000.pth"  # ← 修改为你的模型路径
#     model.load_state_dict(torch.load(model_path, map_location=device))
#     model.eval()

#     transform = transforms.Compose([
#         transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
#         transforms.ToTensor()
#     ])

#     # ================== 输入准备 ==================
#     building = transform(Image.open(building_path).convert('L')).unsqueeze(0).to(device)
#     cfd = transform(Image.open(cfd_path).convert('RGB')).unsqueeze(0).to(device)
#     condition_tensor = torch.tensor([condition], dtype=torch.float32).to(device)

#     # ================== 预测 ==================
#     with torch.no_grad():
#         start_time = time.time()
#         output = model(building, condition_tensor)
#         elapsed = time.time() - start_time
#         print(f"预测耗时: {elapsed:.4f} 秒")

#         os.makedirs(os.path.dirname(save_path), exist_ok=True)
#         save_image(output.cpu(), save_path)
#         print(f"预测完成，结果已保存为 {save_path}")

#         # ================== 误差计算 ==================
#         output_ms = output * vmax
#         cfd_ms = cfd * vmax

#         abs_error = torch.abs(output_ms - cfd_ms).mean(dim=1, keepdim=True)
#         mae = F.l1_loss(output_ms, cfd_ms).item()
#         rmse = torch.sqrt(F.mse_loss(output_ms, cfd_ms)).item()
#         abs_error_np = abs_error.squeeze().cpu().numpy()

#         print(f"MAE: {mae:.3f} m/s, RMSE: {rmse:.3f} m/s")

#         # ================== SSIM计算 ==================
#         output_np = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
#         cfd_np = cfd.squeeze(0).permute(1, 2, 0).cpu().numpy()

#         try:
#             ssim_mean, ssim_map = ssim(
#                 cfd_np, output_np,
#                 channel_axis=-1,
#                 data_range=1.0,
#                 full=True,
#                 win_size=7
#             )
#         except Exception as e:
#             print(f"⚠️ 多通道SSIM失败 ({e})，自动降为灰度模式。")
#             cfd_gray = np.mean(cfd_np, axis=-1)
#             output_gray = np.mean(output_np, axis=-1)
#             ssim_mean, ssim_map = ssim(
#                 cfd_gray, output_gray,
#                 data_range=1.0,
#                 full=True,
#                 win_size=7
#             )

#         if isinstance(ssim_mean, np.ndarray):
#             ssim_mean = float(np.mean(ssim_mean))

#         print(f"Mean SSIM: {ssim_mean:.3f}")

#         # ================== 绘制SSIM分布图 ==================
#         plt.figure(figsize=(6, 6))
#         plt.imshow(ssim_map, cmap='RdYlBu_r', vmin=0, vmax=1)
#         plt.colorbar(label="SSIM", fraction=0.046, pad=0.04)
#         plt.axis("off")
#         plt.text(10, 10, f"Mean SSIM={ssim_mean:.3f}", color='black', fontsize=12,
#                   bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))
#         plt.savefig(save_path.replace(".png", "_SSIM_map.png"), dpi=600, bbox_inches='tight')
#         plt.close()

#         # ================== 绘制误差直方图 ==================
#         bins = np.linspace(0, 12, 50)
#         hist, bin_edges = np.histogram(abs_error_np.flatten(), bins=bins, density=True)
#         bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

#         plt.figure(figsize=(6, 6))
#         plt.axvspan(0, 3, facecolor=(112/255, 163/255, 196/255, 0.2))
#         plt.axvspan(3, 10, facecolor=(199/255, 229/255, 236/255, 0.2))
#         plt.axvspan(10, 12, facecolor=(245/255, 180/255, 111/255, 0.2))
#         plt.bar(bin_centers, hist, width=bin_edges[1]-bin_edges[0], color='gray', edgecolor='black', alpha=0.8)
#         plt.xlim(0, 12)
#         plt.xlabel("Absolute Error (m/s)")
#         plt.ylabel("Proportion")

#         total_count = abs_error_np.size
#         percent_0_3 = np.sum((abs_error_np >= 0) & (abs_error_np <= 3)) / total_count * 100
#         percent_3_10 = np.sum((abs_error_np > 3) & (abs_error_np <= 10)) / total_count * 100
#         percent_10_up = np.sum(abs_error_np > 10) / total_count * 100

#         plt.text(1.5, max(hist)*0.9, f"{percent_0_3:.1f}%", ha='center')
#         plt.text(6.5, max(hist)*0.9, f"{percent_3_10:.1f}%", ha='center')
#         plt.text(11, max(hist)*0.9, f"{percent_10_up:.1f}%", ha='center')

#         plt.savefig(save_path.replace(".png", "_error_hist.png"), dpi=600, bbox_inches='tight')
#         plt.close()

#         # ================== AE分段填色 ==================
#         plt.figure(figsize=(6, 6))
#         bounds = [0, 2, 4, 6, 8, 10, np.max(abs_error_np)]
#         n_bounds = len(bounds) - 1
#         base_cmap = cm.get_cmap("RdYlBu_r", n_bounds)
#         colors = base_cmap(range(n_bounds))
#         cmap = mcolors.ListedColormap(colors)
#         norm = mcolors.BoundaryNorm(bounds, cmap.N)
#         im = plt.imshow(abs_error_np, cmap=cmap, norm=norm)
#         plt.axis("off")
#         cbar = plt.colorbar(im, boundaries=bounds, fraction=0.046, pad=0.04, extend='both')
#         cbar.set_label("Absolute Error (m/s)")
#         plt.savefig(save_path.replace(".png", "_AE_map.png"), dpi=600, bbox_inches='tight')
#         plt.close()

#         # ================== 平均截面风速曲线 ==================
#         pred_np = output_ms.squeeze(0).permute(1, 2, 0).cpu().numpy()
#         cfd_np_ms = cfd_ms.squeeze(0).permute(1, 2, 0).cpu().numpy()
#         pred_gray = np.mean(pred_np, axis=-1)
#         cfd_gray = np.mean(cfd_np_ms, axis=-1)

#         u_x_mean_pred = np.mean(pred_gray, axis=1)
#         u_x_mean_cfd = np.mean(cfd_gray, axis=1)
#         u_y_mean_pred = np.mean(pred_gray, axis=0)
#         u_y_mean_cfd = np.mean(cfd_gray, axis=0)

#         h = np.linspace(0, Config.IMAGE_SIZE - 1, len(u_x_mean_pred))

#         plt.figure(figsize=(12, 6))
#         plt.plot(h, u_x_mean_cfd, 'r-', lw=2, label='CFD - X avg')
#         plt.plot(h, u_x_mean_pred, 'r--', lw=2, label='GAN - X avg')
#         plt.plot(h, u_y_mean_cfd, 'b-', lw=2, label='CFD - Y avg')
#         plt.plot(h, u_y_mean_pred, 'b--', lw=2, label='GAN - Y avg')
#         plt.xlim(10, 250)
#         plt.xticks([])
#         plt.ylabel("Mean Velocity (m/s)")
#         plt.title("Mean Velocity Profiles")
#         plt.legend()
#         plt.grid(True, linestyle='--', alpha=0.5)
#         plt.savefig(save_path.replace(".png", "_MeanProfile.png"), dpi=600, bbox_inches='tight')
#         plt.close()

#         return mae, rmse, ssim_mean


# # ================== 多方向预测与统计 ==================
# if __name__ == "__main__":

#     results = []

#     directions = [
#         ("135°", [-19.49, 4.77, -0.79, 135], r"D:\gan\xin\taifeng_qiangduiliu\cfd\t10_southeast.png",
#          r"D:\gan\xin\taifeng_qiangduiliu\图片2\10m_135_11000_t.png"),
#         ("180°", [-19.49, 4.77, -0.79, 180], r"D:\gan\xin\taifeng_qiangduiliu\cfd\t10_south.png",
#          r"D:\gan\xin\taifeng_qiangduiliu\图片2\10m_180_11000_t.png"),
#         ("315°", [-19.49, 4.77, -0.79, 315], r"D:\gan\xin\taifeng_qiangduiliu\cfd\t10_northwest.png",
#          r"D:\gan\xin\taifeng_qiangduiliu\图片2\10m_315_11000_t.png"),
#     ]

#     for direction, cond, cfd_path, save_path in directions:
#         mae, rmse, ssim_val = predict(
#             building_path=r"D:/gan/xin/buildings/10m.png",
#             condition=cond,
#             cfd_path=cfd_path,
#             save_path=save_path
#         )
#         results.append((direction, mae, rmse, ssim_val))

#     # 输出各入流方向指标
#     print("\n====== 各入流方向误差指标 ======")
#     for direction, mae, rmse, ssim_val in results:
#         print(f"{direction}: MAE={mae:.3f}, RMSE={rmse:.3f}, SSIM={ssim_val:.3f}")

#     # 计算平均指标
#     maes = [r[1] for r in results]
#     rmses = [r[2] for r in results]
#     ssims = [r[3] for r in results]

#     print("\n====== 平均误差指标 ======")
#     print(f"平均 MAE:  {np.mean(maes):.3f}")
#     print(f"平均 RMSE: {np.mean(rmses):.3f}")
#     print(f"平均 SSIM: {np.mean(ssims):.3f}")
