import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
import os

# 1. 数据集准备
class TTSDataset(Dataset):
    def __init__(self, text_files, audio_files, tokenizer, max_length=128):
        self.text_files = text_files
        self.audio_files = audio_files
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.text_files)

    def __getitem__(self, idx):
        # 加载文本
        with open(self.text_files[idx], 'r', encoding='utf-8') as f:
            text = f.read().strip()

        # 对文本进行分词和编码
        tokens = self.tokenizer(text, return_tensors='pt', padding='max_length',
                                truncation=True, max_length=self.max_length)
        input_ids = tokens['input_ids'].squeeze(0)  # 形状：[max_length]
        attention_mask = tokens['attention_mask'].squeeze(0)  # 形状：[max_length]

        # 加载音频
        waveform, sample_rate = torchaudio.load(self.audio_files[idx])  # 形状：[channels, time]
        waveform = waveform.mean(dim=0, keepdim=True)  # 如果不是单声道，转换为单声道

        # 可选：重采样到统一的采样率
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)
            sample_rate = 16000

        # 可选：裁剪或填充音频到固定长度
        max_audio_length = 16000  # 例如，1秒的音频
        if waveform.shape[1] > max_audio_length:
            waveform = waveform[:, :max_audio_length]
        else:
            padding = max_audio_length - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))

        return input_ids, attention_mask, waveform

# 2. 模型定义

# 音频编码器：将波形转换为一维向量
class AudioEncoder(nn.Module):
    def __init__(self, encoded_dim):
        super(AudioEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # 输出形状：[batch_size, 128, 1]
            nn.Flatten(),  # 输出形状：[batch_size, 128]
            nn.Linear(128, encoded_dim)  # 输出形状：[batch_size, encoded_dim]
        )

    def forward(self, x):
        return self.encoder(x)

# 音频解码器：将一维向量还原为波形
class AudioDecoder(nn.Module):
    def __init__(self, encoded_dim, output_length):
        super(AudioDecoder, self).__init__()
        self.output_length = output_length
        self.decoder = nn.Sequential(
            nn.Linear(encoded_dim, 128 * (output_length // 4)),
            nn.ReLU(),
            nn.Unflatten(1, (128, output_length // 4)),  # [batch_size, 128, output_length // 4]
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),  # 输出形状：[batch_size, 1, output_length]
        )

    def forward(self, x):
        x = self.decoder(x)
        return x

# 文本编码器
class TextEncoder(nn.Module):
    def __init__(self, embedding_dim):
        super(TextEncoder, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.projection = nn.Linear(768, embedding_dim)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = outputs.last_hidden_state  # [batch_size, seq_len, 768]
        embeddings = self.projection(embeddings)  # [batch_size, seq_len, embedding_dim]
        embeddings = embeddings.mean(dim=1)  # [batch_size, embedding_dim]
        return embeddings

# 扩散模型
class DiffusionModel(nn.Module):
    def __init__(self, audio_dim, text_embedding_dim):
        super(DiffusionModel, self).__init__()
        self.fc1 = nn.Linear(audio_dim + text_embedding_dim, 512)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, audio_dim)

    def forward(self, audio_vector, text_embedding):
        x = torch.cat([audio_vector, text_embedding], dim=-1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# 3. 训练循环
def train_model(dataset, text_encoder, audio_encoder, audio_decoder, diffusion_model,
                num_epochs=10, batch_size=16, learning_rate=1e-4, device='cpu'):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 优化器
    encoder_optimizer = optim.Adam(audio_encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(audio_decoder.parameters(), lr=learning_rate)
    diffusion_optimizer = optim.Adam(diffusion_model.parameters(), lr=learning_rate)

    # 损失函数
    criterion = nn.MSELoss()

    text_encoder.to(device)
    audio_encoder.to(device)
    audio_decoder.to(device)
    diffusion_model.to(device)

    text_encoder.eval()  # 冻结文本编码器

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for input_ids, attention_mask, waveform in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            waveform = waveform.to(device)

            # 编码文本
            with torch.no_grad():
                text_embedding = text_encoder(input_ids, attention_mask)  # [batch_size, text_embedding_dim]

            # 编码音频
            audio_vector = audio_encoder(waveform)  # [batch_size, audio_dim]

            # 添加噪声
            noise = torch.randn_like(audio_vector)
            noisy_audio_vector = audio_vector + noise

            # 预测噪声
            predicted_audio_vector = diffusion_model(noisy_audio_vector, text_embedding)

            # 计算损失
            loss = criterion(predicted_audio_vector, audio_vector)

            # 反向传播
            diffusion_optimizer.zero_grad()
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            loss.backward()

            diffusion_optimizer.step()
            encoder_optimizer.step()
            decoder_optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

    # 保存模型
    os.makedirs('saved_models', exist_ok=True)
    torch.save(audio_encoder.state_dict(), 'saved_models/audio_encoder.pth')
    torch.save(audio_decoder.state_dict(), 'saved_models/audio_decoder.pth')
    torch.save(diffusion_model.state_dict(), 'saved_models/diffusion_model.pth')
    print("模型已保存到 'saved_models/' 目录中。")

# 4. 推理函数
def inference(text, text_encoder, diffusion_model, audio_decoder, num_steps=50, device='cpu'):
    # 编码文本
    text_encoder.to(device)
    diffusion_model.to(device)
    audio_decoder.to(device)

    text_encoder.eval()
    diffusion_model.eval()
    audio_decoder.eval()

    tokenizer = text_encoder.tokenizer
    tokens = tokenizer(text, return_tensors='pt', padding=True)
    input_ids = tokens['input_ids'].to(device)
    attention_mask = tokens['attention_mask'].to(device)

    with torch.no_grad():
        text_embedding = text_encoder(input_ids, attention_mask)  # [1, embedding_dim]

    # 初始化随机音频向量
    audio_dim = diffusion_model.fc2.out_features
    audio_vector = torch.randn(1, audio_dim).to(device)

    # 扩散过程
    for step in range(num_steps):
        noise_level = (num_steps - step) / num_steps
        noise = torch.randn_like(audio_vector) * noise_level
        noisy_audio_vector = audio_vector + noise

        # 去噪
        with torch.no_grad():
            audio_vector = diffusion_model(noisy_audio_vector, text_embedding)

    # 解码音频向量
    generated_waveform = audio_decoder(audio_vector)
    generated_waveform = generated_waveform.squeeze(0).cpu()  # [time]

    return generated_waveform

# 5. 主函数
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 准备数据集（需要提供文本和音频文件的路径列表）
    text_files = ['test.txt']  # 替换为您的文本文件路径
    audio_files = ['resources/slice/shoulinrui.m4a/shoulinrui.m4a_0000063040_0000325440.wav']  # 替换为您的音频文件路径

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    dataset = TTSDataset(text_files, audio_files, tokenizer)

    # 模型参数
    encoded_dim = 256
    text_embedding_dim = 128
    output_length = 16000*5  # 例如，1秒的音频（16000采样点）

    # 初始化模型
    text_encoder = TextEncoder(text_embedding_dim)
    audio_encoder = AudioEncoder(encoded_dim)
    audio_decoder = AudioDecoder(encoded_dim, output_length)
    diffusion_model = DiffusionModel(encoded_dim, text_embedding_dim)

    # 训练模型
    train_model(dataset, text_encoder, audio_encoder, audio_decoder, diffusion_model,
                num_epochs=100, batch_size=4, learning_rate=1e-4, device=device)

    # 推理
    # 加载训练好的模型
    audio_encoder.load_state_dict(torch.load('saved_models/audio_encoder.pth', map_location=device))
    audio_decoder.load_state_dict(torch.load('saved_models/audio_decoder.pth', map_location=device))
    diffusion_model.load_state_dict(torch.load('saved_models/diffusion_model.pth', map_location=device))

    # 示例输入文本
    text = "啊，一些重要的公司会以各种"

    # 生成音频
    generated_waveform = inference(text, text_encoder, diffusion_model, audio_decoder, device=device)
    print(generated_waveform.shape)
    # 保存生成的音频
    sample_rate = 16000
    os.makedirs('generated_audio', exist_ok=True)
    output_path = 'generated_audio/generated_speech.wav'
    torchaudio.save(output_path, generated_waveform.detach(), sample_rate=sample_rate)
    print(f"生成的音频已保存为 '{output_path}'")

if __name__ == "__main__":
    main()
