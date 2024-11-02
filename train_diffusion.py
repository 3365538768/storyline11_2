import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from custom_dataset import CustomDataset, custom_collate_fn  # 确保已定义
from audioldm import LatentDiffusion  # 确保已定义
from audioldm.utils import default_audioldm_config, seed_everything, save_wave

def train_model(latent_diffusion, dataloader, epochs=10, lr=1e-4, save_every_epoch=1, save_dir="models/diffusion"):
    """
    训练模型
    Args:
        latent_diffusion (LatentDiffusion): 已实例化的LatentDiffusion模型
        dataloader (DataLoader): 数据加载器
        epochs (int): 训练轮数
        lr (float): 学习率
        save_every_epoch (int): 每训练多少个轮次保存一次模型
        save_dir (str): 模型保存的目录
    """
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)

    # 定义优化器和损失函数
    optimizer = optim.Adam(latent_diffusion.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    # 将模型移动到设备
    device = latent_diffusion.device
    latent_diffusion = latent_diffusion.to(device)

    # 开启训练模式
    latent_diffusion.train()

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}"):
            # 提取批次数据并移动到设备
            audios = batch['audio'].to(device)                    # [batch_size, max_audio_length]
            phoneme_feats = batch['phoneme_feat'].to(device)      # [batch_size, max_text_length, 3]
            hubert_feats = batch['hubert_feat'].to(device)        # [batch_size, 1, max_audio_length1, 1024]
            emotion_feats = batch['emotion_feat'].to(device)      # [batch_size, max_audio_length2, 9]
            texts = batch['text']                                 # list of strings, [batch_size]

            # 获取条件编码
            cond = latent_diffusion.get_learned_conditioning({
                'text': texts,
                'phoneme_feat': phoneme_feats,
                'hubert_feat': hubert_feats,
                'emotion_feat': emotion_feats
            })  # [batch_size, 704]

            # 随机时间步
            t = torch.randint(0, latent_diffusion.num_timesteps, (audios.size(0),), device=device).long()

            # 添加噪声
            noise = torch.randn_like(audios)
            x_noisy = latent_diffusion.q_sample(x_start=audios, t=t, noise=noise)  # [batch_size, max_audio_length]

            # 模型预测噪声
            noise_pred = latent_diffusion(x_noisy, t, cond)  # [batch_size, max_audio_length]

            # 计算损失
            loss = loss_fn(noise_pred, noise)  # MSE损失

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch}/{epochs} - Average Loss: {avg_loss:.6f}")

        # 保存模型权重
        if epoch % save_every_epoch == 0:
            save_path = os.path.join(save_dir, f"latent_diffusion_epoch_{epoch}.pth")
            torch.save(latent_diffusion.state_dict(), save_path)
            print(f"Model saved to {save_path}")

    print("Training completed.")

if __name__ == "__main__":
    # 设置随机种子以确保可复现性
    seed_everything(42)

    # 定义文件路径
    list_file = "resources/asr/shoulinrui.m4a/shoulinrui.m4a.list"  # 替换为您的.list文件路径
    phoneme_path = "resources/text2phonemes/shoulinrui.m4a"
    hubert_path = "resources/wave_hubert/shoulinrui.m4a"
    emotion_path = "resources/emotion/shoulinrui.m4a"

    # 实例化数据集和数据加载器
    dataset = CustomDataset(list_file, phoneme_path, hubert_path, emotion_path)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4, pin_memory=True, collate_fn=custom_collate_fn)

    # 加载模型配置
    model_name = "audioldm-s-full"  # 您的模型名称
    config = default_audioldm_config(model_name)

    # 实例化模型
    latent_diffusion = LatentDiffusion(
        unet_config=config["model"]["params"]["unet_config"],
        device="cuda" if torch.cuda.is_available() else "cpu",
        first_stage_config=config["model"]["params"]["first_stage_config"],
        cond_stage_config=config["model"]["params"]["cond_stage_config"],
        num_timesteps_cond=config["model"]["params"].get("num_timesteps_cond", 1),
        cond_stage_key="text",
        cond_stage_trainable=config["model"]["params"].get("cond_stage_trainable", False),
        concat_mode=config["model"]["params"].get("concat_mode", True),
        conditioning_key=config["model"]["params"].get("conditioning_key", "concat"),
        scale_factor=config["model"]["params"].get("scale_factor", 1.0),
        scale_by_std=config["model"]["params"].get("scale_by_std", False),
        base_learning_rate=config["model"]["params"].get("base_learning_rate", 1e-4),
        phoneme_key="phoneme_feat",
        hubert_key="hubert_feat",
        emotion_key="emotion_feat",
    )

    # 开始训练
    train_model(
        latent_diffusion=latent_diffusion,
        dataloader=dataloader,
        epochs=10,                # 设置为您需要的训练轮数
        lr=1e-4,                  # 设置为您需要的学习率
        save_every_epoch=1,       # 每个轮次保存一次模型
        save_dir="checkpoints"    # 模型保存目录
    )
