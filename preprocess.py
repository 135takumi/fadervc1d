import json
import os
import random
import shutil

import librosa
import numpy as np
from tqdm import tqdm

import hparams as hp
from utils import world_decompose, world_encode_spectral_envelop


def pitch_statistics(f0s):
    f0 = np.concatenate(f0s)

    log_f0 = np.log(f0)
    mean = log_f0.mean()
    std = log_f0.std()

    return mean, std


def make_exp_dir():
    """
    実験に使うディレクトリの作成
    """

    os.makedirs(hp.wav_dir / "male", exist_ok=True)
    os.makedirs(hp.wav_dir / "female", exist_ok=True)
    os.makedirs(hp.tng_data_dir, exist_ok=True)
    os.makedirs(hp.val_data_dir, exist_ok=True)
    os.makedirs(hp.test_data_dir, exist_ok=True)
    os.makedirs(hp.tng_result_dir, exist_ok=True)
    os.makedirs(hp.test_result_dir, exist_ok=True)

def mcep_statistics(mceps):
    mcep = np.concatenate(mceps, axis=0)

    mean = list(np.mean(mcep, axis=0, keepdims=True).squeeze())
    std = list(np.std(mcep, axis=0, keepdims=True).squeeze())

    return mean, std


def extract_feature(wav_path, save_dir):
    # 読み込み 正規化
    wav, _ = librosa.core.load(wav_path, sr=hp.sampling_rate)
    wav = librosa.util.normalize(wav)

    # 前後の無音を除去 top dbでどれぐらい厳しく削除するか決める
    wav, _ = librosa.effects.trim(wav, top_db=60)

    # WORLDを利用して特徴量を取得
    f0, time_axis, sp, ap = world_decompose(wav, hp.sampling_rate)

    # ケプストラムをメルケプストラムに
    # パワー項も次元数に含まれているので+1
    mcep = world_encode_spectral_envelop(sp, hp.sampling_rate, hp.mcep_channels + 1)

    # 0次元目はパワー項なので削除
    power = mcep[:, 0:1]
    mcep = mcep[:, 1:]


    # 長さが短いものを除く
    if mcep.shape[0] < hp.seq_len:
        print(f"{wav_path} is too short")
        return None

    f0_path = save_dir / "f0.npy"
    mcep_path = save_dir / "mcep.npy"
    power_path = save_dir / "power.npy"
    ap_path = save_dir / "ap.npy"

    np.save(f0_path, f0, allow_pickle=False)
    np.save(mcep_path, mcep, allow_pickle=False)
    np.save(power_path, power, allow_pickle=False)
    np.save(ap_path, ap, allow_pickle=False)

    return f0, mcep


def download_wav(speaker_list, gender):
    for speaker in speaker_list:
        jvs_wav_dir = hp.dir_path_jvs / speaker / 'parallel100' / 'wav24kHz16bit'
        save_dir = hp.wav_dir / gender / speaker
        os.makedirs(save_dir, exist_ok=True)
        shutil.copytree(jvs_wav_dir, save_dir)


def prepare_wav():
    with open(hp.dir_path_jvs / 'gender_f0range.txt') as f:
        speaker_info_list = f.readlines()
    male_speaker_list = [si[0:6] for si in speaker_info_list if si[7] == 'M']
    female_speaker_list = [si[0:6] for si in speaker_info_list if si[7] == 'F']

    sampled_male_speaker_list = random.sample(male_speaker_list, (hp.seen_speaker_num / 2))
    sampled_female_speaker_list = random.sample(female_speaker_list, (hp.seen_speaker_num / 2))

    download_wav(sampled_male_speaker_list, 'male')
    download_wav(sampled_female_speaker_list, 'female')


def make_seen_dataset(seen_speaker_lst, seen_test_speaker_lst, mcep_dct, f0_dct):
    """
    学習用データの作成(train用話者の時と，test用話者の時で分岐あり)
    """
    mcep_lst = []
    seen_speaker_dct = {}
    seen_test_speaker_dct = {}

    for speaker_index, speaker in enumerate(tqdm(seen_speaker_lst)):
        f0_lst = []
        seen_speaker_dct[speaker] = speaker_index

        wav_lst = list(hp.wav_dir.glob("*/"+speaker+"/*.wav"))
        train_wav_lst = wav_lst[:hp.train_wav_num]
        valid_wav_lst = wav_lst[hp.train_wav_num:hp.train_wav_num+hp.valid_wav_num]

        if speaker in seen_test_speaker_lst:
            seen_test_speaker_dct[speaker] = speaker_index
            test_wav_lst = wav_lst[-hp.test_wav_num:]

        for train_wav in train_wav_lst:
            f0, mcep = extract_feature(train_wav, hp.tng_data_dir / speaker)
            f0 = [f for f in f0 if f > 0.0]
            mcep_lst.append(mcep)
            f0_lst.append(f0)

        for valid_wav in valid_wav_lst:
            _, _ = extract_feature(valid_wav, hp.val_data_dir / speaker)

        for test_wav in test_wav_lst:
            _, _ = extract_feature(test_wav, hp.val_data_dir / speaker)

        f0_mean, f0_std = pitch_statistics(f0_lst)
        f0_dct[speaker] = {"mean": f0_mean, "std": f0_std}

    for speaker in seen_speaker_lst:
        mcep_mean, mcep_std = mcep_statistics(mcep_lst)
        mcep_dct[speaker] = {"mean": mcep_mean, "std": mcep_std}

    return seen_speaker_dct, seen_test_speaker_dct


def make_unseen_dataset(test_speaker_lst, f0_dct=None):
    """
    テスト用データの作成(seen話者とunseen話者で分岐あり)
    """
    unseen_speaker_dct={}

    for speaker_index, speaker in enumerate(tqdm(test_speaker_lst)):
        f0_lst = []

        wav_lst = list(hp.wav_dir.glob("*/" + speaker + "/*.wav"))
        test_wav_lst = wav_lst[:hp.test_wav_num]

        for test_wav in test_wav_lst:
            f0, mcep = extract_feature(test_wav, hp.test_data_dir / speaker)
            f0 = [f for f in f0 if f > 0.0]
            f0_lst.append(f0)

        f0_mean, f0_std = pitch_statistics(f0_lst)
        f0_dct[speaker] = {"mean": f0_mean, "std": f0_std}

    return unseen_speaker_dct


def make_dataset():
    mcep_dct = {}
    f0_dct = {}

    male_speaker_lst = [f for f in os.listdir(hp.wav_dir / "male") if not f.startswith('.')]
    male_speaker_lst = random.sample(sorted(male_speaker_lst), len(male_speaker_lst))
    female_speaker_lst = [f for f in os.listdir(hp.wav_dir / "female") if not f.startswith('.')]
    female_speaker_lst = random.sample(sorted(female_speaker_lst), len(female_speaker_lst))

    seen_speaker_lst = male_speaker_lst[:int(hp.seen_speaker_num / 2)] + \
                       female_speaker_lst[:int(hp.seen_speaker_num / 2)]
    seen_test_speaker_lst = male_speaker_lst[:int(hp.seen_test_speaker_num / 2)] + \
                            female_speaker_lst[:int(hp.seen_test_speaker_num / 2)]
    unseen_speaker_lst = male_speaker_lst[-int(hp.unseen_speaker_num / 2):] + \
                         female_speaker_lst[-int(hp.unseen_speaker_num / 2):]


    seen_speaker_dct, seen_test_speaker_dct = make_seen_dataset(seen_speaker_lst,
                                                                seen_test_speaker_lst,
                                                                mcep_dct, f0_dct)

    unseen_speaker_dct = make_unseen_dataset(unseen_speaker_lst, f0_dct=f0_dct)

    with open(hp.session_dir / "f0_statistics.json", 'w') as f:
        json.dump(f0_dct, f, indent=2)
    with open(hp.session_dir / "mcep_statistics.json", 'w') as f:
        json.dump(mcep_dct, f, indent=2)
    with open(hp.session_dir / "seen_speaker.json", 'w') as f:
        json.dump(seen_speaker_dct, f, indent=2)
    with open(hp.session_dir / "seen_test_speaker.json", 'w') as f:
        json.dump(seen_test_speaker_dct, f, indent=2)
    with open(hp.session_dir / "unseen_speaker.json", 'w') as f:
        json.dump(unseen_speaker_dct, f, indent=2)


def main():
    make_exp_dir()
    prepare_wav()
    make_dataset()

if __name__ == '__main__':
    main()
