import argparse
import json
import os
import pathlib
import random

import librosa
import numpy as np
from tqdm import tqdm

import hparams
from utils import world_decompose, world_encode_spectral_envelop


def pitch_statistics(f0s):
    f0 = np.concatenate(f0s)

    log_f0 = np.log(f0)
    mean = log_f0.mean()
    std = log_f0.std()

    return mean, std


def mcep_statistics(mceps):
    mcep = np.concatenate(mceps, axis=0)

    mean = list(np.mean(mcep, axis=0, keepdims=True).squeeze())
    std = list(np.std(mcep, axis=0, keepdims=True).squeeze())

    return mean, std


def extract_feature(wav_path, save_dir):
    # 読み込み 正規化
    wav, _ = librosa.core.load(wav_path, sr=hparams.sampling_rate)
    wav = librosa.util.normalize(wav)

    # 前後の無音を除去 top dbでどれぐらい厳しく削除するか決める
    wav, _ = librosa.effects.trim(wav, top_db=60)

    # WORLDを利用して特徴量を取得
    f0, time_axis, sp, ap = world_decompose(wav, hparams.sampling_rate)

    # ケプストラムをメルケプストラムに
    # パワー項も次元数に含まれているので+1
    mcep = world_encode_spectral_envelop(sp, hparams.sampling_rate, hparams.mcep_channels + 1)

    # 0次元目はパワー項なので削除
    mcep = mcep[:, 1:]

    # 長さが短いものを除く
    if mcep.shape[0] < hparams.seq_len:
        print(f"{wav_path} is too short")
        return None

    f0_path = os.path.join(save_dir, "f0.npy")
    mcep_path = os.path.join(save_dir, "mcep.npy")
    ap_path = os.path.join(save_dir, "ap.npy")

    np.save(f0_path, f0, allow_pickle=False)
    np.save(mcep_path, mcep, allow_pickle=False)
    np.save(ap_path, ap, allow_pickle=False)

    return f0, mcep


def main():
    # 実験ディレクトリ作成
    feature_dir = hparams.data_root / "feature"
    log_dir = hparams.data_root / "log"
    model_pth_dir = hparams.data_root / "model_pth"
    result_dir = hparams.data_root / "result"

    os.makedirs(feature_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_pth_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    wav_dir = hparams.data_root / "wav"
    speaker_dict = {}
    f0_dict = {}
    mcep_dict = {}
    files = []
    for speaker_index, speaker in enumerate(tqdm(sorted(os.listdir(wav_dir)))):
        speaker_dir = wav_dir / speaker
        if not os.path.isdir(speaker_dir):
            continue

        speaker_dict[speaker] = speaker_index

        # f0とmcepから話者ごとに平均・分散を求める
        f0s = []
        mceps = []
        for wav_file in os.listdir(speaker_dir):
            if not wav_file.endswith(".wav"):
                continue

            filename = wav_file.replace(".wav", "")
            wav_path = speaker_dir / wav_file

            file_id = f"{speaker}_{filename}"
            save_dir = feature_dir / speaker / file_id
            os.makedirs(save_dir, exist_ok=True)

            ret = extract_feature(wav_path, save_dir)
            if ret is not None:
                f0, mcep = ret
                f0 = [f for f in f0 if f > 0.0]
                f0s.append(f0)
                mceps.append(mcep)
                files.append(file_id)

        f0_mean, f0_std = pitch_statistics(f0s)
        f0_dict[speaker] = {'mean': f0_mean, 'std': f0_std}
        mcep_mean, mcep_std = mcep_statistics(mceps)
        mcep_dict[speaker] = {'mean': mcep_mean, 'std': mcep_std}

    # 学習用とテスト用ファイルに分割
    random.shuffle(files)
    train_files = files[:-hparams.test_file_num]
    test_files = files[-hparams.test_file_num:]

    with open(hparams.data_root / "f0_statistics.json", 'w') as f:
        json.dump(f0_dict, f, indent=2)

    with open(hparams.data_root / "mcep_statistics.json", 'w') as f:
        json.dump(mcep_dict, f, indent=2)

    with open(hparams.data_root / "speaker.json", 'w') as f:
        json.dump(speaker_dict, f, indent=2)

    with open(hparams.data_root / "train_files.txt", mode='w') as f:
        f.write('\n'.join(train_files))

    with open(hparams.data_root / "test_files.txt", mode='w') as f:
        f.write('\n'.join(test_files))


if __name__ == '__main__':
    main()
