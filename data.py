import os
import random

import numpy as np
import torch


class AudioDataset(torch.utils.data.Dataset):

    def __init__(self, data_root, speaker_dict, mcep_dict, files, seq_len=128):

        self.seq_len = seq_len
        self.speaker_dict = speaker_dict

        self.mcep_dict = {}
        for k, v in mcep_dict.items():
            label = self.speaker_dict[k]
            self.mcep_dict[label] = {
                "mean": np.array(v['mean'])[None, :],
                "std": np.array(v['std'])[None, :]
            }

        self.files = set(files)
        self.data = self.read_data(data_root)

    def mcep_normalize(self, mcep, label):
        speaker_dict = self.mcep_dict[label]
        mean, std = speaker_dict['mean'], speaker_dict['std']
        mcep = (mcep - mean) / std

        return mcep

    def read_data(self, data_root):
        data = []

        feature_dir = os.path.join(data_root, "feature")
        for speaker in os.listdir(feature_dir):
            speaker_label = self.speaker_dict[speaker]
            speaker_dir = os.path.join(feature_dir, speaker)
            for uttr in os.listdir(speaker_dir):
                if uttr in self.files:
                    uttr_dir = os.path.join(speaker_dir, uttr)
                    mcep = np.load(os.path.join(uttr_dir, "mcep.npy"))
                    mcep = self.mcep_normalize(mcep, speaker_label)
                    data.append((mcep.T, speaker_label))

        return data

    def __getitem__(self, index):
        mcep, label = self.data[index]

        max_start = np.shape(mcep)[1] - self.seq_len
        start = random.randint(0, max_start)
        mcep = mcep[:, start:start + self.seq_len]

        mcep = torch.from_numpy(mcep).float()
        label = torch.from_numpy(np.array(label)).long()

        return mcep, label

    def __len__(self):
        return len(self.files)
