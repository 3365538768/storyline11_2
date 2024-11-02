import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchaudio import load
import tqdm
import torchaudio.transforms as T


class CustomDataset(Dataset):
    def __init__(self, list_file, phoneme_path, hubert_path, emotion_path, max_time_steps=1216):
        """
        初始化数据集
        Args:
            list_file (str): .list文件的路径
            phoneme_path (str): 音素特征文件夹路径
            hubert_path (str): Hubert特征文件夹路径
            emotion_path (str): 情感特征文件夹路径
            max_time_steps (int): 固定的最大时间步长度
        """
        self.phoneme_path = phoneme_path
        self.hubert_path = hubert_path
        self.emotion_path = emotion_path
        self.max_time_steps = max_time_steps
        self.data = []
        self.mel_transform = T.MelSpectrogram(
            sample_rate=16000,
            n_mels=64,
            mel_scale='htk',
            # 您可以根据需要调整其他参数
        )

        # 解析 .list 文件
        with open(list_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue  # 跳过空行
                parts = line.split('|')
                if len(parts) != 4:
                    print(f"Skipping invalid line: {line}")
                    continue
                audio_segment_path, base_audio_file, language, text = parts
                self.data.append({
                    'audio_segment_path': audio_segment_path,
                    'base_audio_file': base_audio_file,
                    'language': language,
                    'text': text
                })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        获取指定索引的样本
        Args:
            idx (int): 索引
        Returns:
            dict: 包含音频、音素特征、Hubert特征、情感特征和文本描述
        """
        sample = self.data[idx]
        audio_segment_path = sample['audio_segment_path']
        base_audio_file = sample['base_audio_file']
        language = sample['language']
        text = sample['text']

        # 加载音频片段
        audio, sr = load(audio_segment_path)  # audio: [channels, samples]
        audio = audio.squeeze(0).float()  # 假设单声道，结果为 [samples]

        # 转换为梅尔频谱图
        mel_spec = self.mel_transform(audio)  # [n_mels, time_steps]
        mel_spec = mel_spec.unsqueeze(0)  # [1, n_mels, time_steps]

        # 填充或截断梅尔频谱图到固定的时间步长度
        if mel_spec.shape[2] < self.max_time_steps:
            pad_length = self.max_time_steps - mel_spec.shape[2]
            mel_spec = torch.nn.functional.pad(mel_spec, (0, pad_length), "constant", 0.0)  # pad time dimension
        elif mel_spec.shape[2] > self.max_time_steps:
            mel_spec = mel_spec[:, :, :self.max_time_steps]  # 截断

        # 加载音素特征
        # 假设音素特征文件名为：{基准音频文件名}_{segment_id}.pt
        segment_id = os.path.basename(audio_segment_path)
        phoneme_file = os.path.join(self.phoneme_path, f"{segment_id}.pt")
        phoneme_feat = torch.load(phoneme_file,weights_only=True,map_location='cpu').float()  # [文本长度, 3]

        # 加载Hubert特征
        hubert_file = os.path.join(self.hubert_path, f"{segment_id}.pt")
        hubert_feat = torch.load(hubert_file,weights_only=True,map_location='cpu').float()  # [1, 音频长度1, 1024]

        # 加载情感特征
        emotion_file = os.path.join(self.emotion_path, f"{segment_id}.pt")
        emotion_feat = torch.load(emotion_file,weights_only=True,map_location='cpu').float()  # [音频长度2, 9]

        return {
            'audio': mel_spec,
            'phoneme_feat': phoneme_feat,
            'hubert_feat': hubert_feat,
            'emotion_feat': emotion_feat,
            'text': text
        }
import torch
from torch.nn.utils.rnn import pad_sequence


import torch
from torch.nn.utils.rnn import pad_sequence

def custom_collate_fn(batch):
    """
    自定义的批次组装函数，用于填充不同长度的音频和特征
    Args:
        batch (list): 包含多个样本的列表
    Returns:
        dict: 包含填充后的批次数据
    """
    try:
        audios = [sample['audio'] for sample in batch]             # [1, n_mels, max_time_steps]
        phoneme_feats = [sample['phoneme_feat'] for sample in batch]
        hubert_feats = [sample['hubert_feat'] for sample in batch]
        emotion_feats = [sample['emotion_feat'] for sample in batch]
        texts = [sample['text'] for sample in batch]

        # 堆叠音频
        audios_padded = torch.stack(audios, dim=0)  # [batch_size, 1, n_mels, max_time_steps]

        # 填充音素特征到固定长度，例如 max_text_length=100
        max_text_length = 100  # 根据数据集调整
        phoneme_padded = []
        for feat in phoneme_feats:
            if feat.shape[0] < max_text_length:
                pad_length = max_text_length - feat.shape[0]
                padded_feat = torch.nn.functional.pad(feat, (0, 0, 0, pad_length), "constant", 0.0)
            else:
                padded_feat = feat[:max_text_length]
            phoneme_padded.append(padded_feat)
        phoneme_padded = torch.stack(phoneme_padded, dim=0)  # [batch_size, max_text_length, 3]

        # 填充Hubert特征到固定长度，例如 max_audio_length1=500
        max_audio_length1 = 500  # 根据数据集调整
        hubert_padded = []
        for feat in hubert_feats:
            feat = feat.squeeze(0)  # [audio_length1, 1024]
            if feat.shape[0] < max_audio_length1:
                pad_length = max_audio_length1 - feat.shape[0]
                padded_feat = torch.nn.functional.pad(feat, (0, 0, 0, pad_length), "constant", 0.0)
            else:
                padded_feat = feat[:max_audio_length1]
            hubert_padded.append(padded_feat)
        hubert_padded = torch.stack(hubert_padded, dim=0)  # [batch_size, max_audio_length1, 1024]
        hubert_padded = hubert_padded.unsqueeze(1)        # [batch_size, 1, max_audio_length1, 1024]

        # 填充情感特征到固定长度，例如 max_audio_length2=500
        max_audio_length2 = 500  # 根据数据集调整
        emotion_padded = []
        for feat in emotion_feats:
            if feat.shape[0] < max_audio_length2:
                pad_length = max_audio_length2 - feat.shape[0]
                padded_feat = torch.nn.functional.pad(feat, (0, 0, 0, pad_length), "constant", 0.0)
            else:
                padded_feat = feat[:max_audio_length2]
            emotion_padded.append(padded_feat)
        emotion_padded = torch.stack(emotion_padded, dim=0)  # [batch_size, max_audio_length2, 9]

        return {
            'audio': audios_padded,                # [batch_size, 1, n_mels, max_time_steps]
            'phoneme_feat': phoneme_padded,        # [batch_size, max_text_length, 3]
            'hubert_feat': hubert_padded,          # [batch_size, 1, max_audio_length1, 1024]
            'emotion_feat': emotion_padded,        # [batch_size, max_audio_length2, 9]
            'text': texts                          # list of strings, [batch_size]
        }
    except Exception as e:
        print(f"Error in collate_fn: {e}")
        raise e



# 示例数据加载
if __name__ == "__main__":
    # 数据集路径
    list_file = "resources/asr/shoulinrui.m4a/shoulinrui.m4a.list"  # 替换为您的.list文件路径
    phoneme_path = "resources/text2phonemes/shoulinrui.m4a"
    hubert_path = "resources/wave_hubert/shoulinrui.m4a"
    emotion_path = "resources/emotion/shoulinrui.m4a"

    # 实例化数据集
    dataset = CustomDataset(list_file, phoneme_path, hubert_path, emotion_path)

    # 创建数据加载器
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4, pin_memory=True,
                            collate_fn=custom_collate_fn)

    for batch in dataloader:
        # 提取批次数据并移动到设备
        audios = batch['audio'] # [batch_size, max_audio_length]
        phoneme_feats = batch['phoneme_feat']  # [batch_size, max_text_length, 3]
        hubert_feats = batch['hubert_feat'] # [batch_size, 1, max_audio_length1, 1024]
        emotion_feats = batch['emotion_feat']  # [batch_size, max_audio_length2, 9]
        texts = batch['text']  # list of strings, [batch_size]
        print(texts)
        print(f"Batch audio shape: {audios.shape}, Phoneme shape: {phoneme_feats.shape}, Hubert shape: {hubert_feats.shape}, Emotion shape: {emotion_feats.shape}")

#['过去，星巴克确实曾经靠品牌的精神内核以及全球伙伴的齐心协力度过了难关。', '啊，一些重要的公司会以各种主题反复出现在商业就是这样的节目中，例如啊星巴克。', '并最终送走了纳斯汗啊。从财报数据来看，星巴克在它最重要的两大板块，目前的情况确实都不太好。', '个性化和移动端订单的增长，让北美门店叫苦不迭。']
#Batch audio shape: torch.Size([4, 288000]), Phoneme shape: torch.Size([4, 42, 3]), Hubert shape: torch.Size([4, 1, 619, 1024]), Emotion shape: torch.Size([4, 9, 9])