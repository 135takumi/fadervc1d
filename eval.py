import argparse
import json
import os

import librosa
import numpy as np
import torch
from tqdm import tqdm

import hparams
from model import FaderVC1D
from utils import (save_wav, speech_synthesis, world_decompose, world_encode_spectral_envelop)


def files_to_list(filepath):
    with open(filepath, encoding="utf-8") as f:
        files = f.readlines()

    files = [f.rstrip() for f in files]
    return files


def mcep_normalize(mcep, label, mcep_dict):
    speaker_dict = mcep_dict[label]
    mean, std = speaker_dict['mean'], speaker_dict['std']
    mcep = (mcep - mean) / std

    return mcep


def mcep_denormalize(mcep, label, mcep_dict):
    speaker_dict = mcep_dict[label]
    mean, std = speaker_dict['mean'], speaker_dict['std']
    mcep = mcep * std + mean

    return mcep


def pitch_conversion(f0, source, target, f0_dict):
    mean_source, std_source = f0_dict[source]["mean"], f0_dict[source]["std"]
    mean_target, std_target = f0_dict[target]["mean"], f0_dict[target]["std"]

    f0_converted = np.exp((np.log(f0 + 1e-6) - mean_source) / std_source * std_target + mean_target)

    return f0_converted


def convert(wav_path, save_path, source, target, speaker_dict, f0_dict, mcep_dict, model, device):
    wav, _ = librosa.core.load(wav_path, sr=hparams.sampling_rate)
    wav = librosa.util.normalize(wav)

    wav, _ = librosa.effects.trim(wav, top_db=60)

    f0, time_axis, sp, ap = world_decompose(wav, hparams.sampling_rate)

    mcep = world_encode_spectral_envelop(sp, hparams.sampling_rate, hparams.mcep_channels + 1)
    mcep_normalized = mcep_normalize(mcep[:, 1:], source, mcep_dict).T

    # 16の整数のフレーム長になるようpad
    pad = 16 - mcep_normalized.shape[1] % 16
    mcep_normalized = np.pad(mcep_normalized, [(0, 0), (0, pad)], 'constant')

    target_label = speaker_dict[target]

    with torch.no_grad():
        x = torch.from_numpy(mcep_normalized).float().unsqueeze(0).to(device)
        label = torch.from_numpy(np.array(target_label)).long().unsqueeze(0).to(device)

        z, _, _ = model.encode(x, label)
        converted_x = model.decode(z, label)
        converted_x = converted_x.squeeze().cpu().numpy()

    # ターゲット話者の平均分散を使って正規化から戻す
    mcep_converted = mcep_denormalize(converted_x.T, target, mcep_dict)

    # 16の整数のフレーム長になるようpadしたのを削除
    mcep_converted = mcep_converted[:-pad, :]

    # パワー項を転写
    mcep_converted = np.concatenate([mcep[:, 0:1], mcep_converted], axis=1)
    mcep_converted = mcep_converted.copy(order='C')

    # f0は線形変換
    f0_converted = pitch_conversion(f0, source, target, f0_dict)

    # ボコーダーで合成
    converted_wav = speech_synthesis(f0_converted, mcep_converted, ap, hparams.sampling_rate)

    # [1.0, -1.0]の範囲を超えることがあるので正規化して0.99かけておく
    converted_wav = librosa.util.normalize(converted_wav) * 0.99

    save_wav(save_path, hparams.sampling_rate, converted_wav)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight', type=str, default="VAEVC-latest.pth")
    parser.add_argument('--exp_name', type=str, default=hparams.exp_name)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(hparams.data_root / "speaker.json", 'r') as f:
        speaker_dict = json.load(f)

    with open(hparams.data_root / "f0_statistics.json", 'r') as f:
        f0_dict = json.load(f)

    mcep_dict = {}
    with open(hparams.data_root / "mcep_statistics.json", 'r') as f:
        for k, v in json.load(f).items():
            mcep_dict[k] = {
                "mean": np.array(v['mean'])[None, :],
                "std": np.array(v['std'])[None, :]
            }

    test_files = files_to_list(hparams.data_root / "test_files.txt")

    model = FaderVC1D(hparams.mcep_channels, hparams.speaker_num).to(device)
    model.load_state_dict(torch.load(hparams.data_root / "model_pth" / args.exp_name / args.weight,
                                     map_location=device)['model'])

    for test_file in tqdm(test_files):
        source, file_name = test_file.split("_", 1)
        wav_path = hparams.data_root / "wav" / source / f"{file_name}.wav"

        for target, _ in speaker_dict.items():
            save_dir = hparams.data_root / "result" / args.exp_name / source / file_name
            os.makedirs(save_dir, exist_ok=True)

            save_path = save_dir / f"{target}.wav"

            convert(wav_path, save_path, source, target, speaker_dict, f0_dict, mcep_dict, model,
                    device)


if __name__ == '__main__':
    main()
