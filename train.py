import argparse
import json
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import hparams
from data import AudioDataset
from model import FaderVC1D


def files_to_list(filepath):
    with open(filepath, encoding="utf-8") as f:
        files = f.readlines()

    files = [f.rstrip() for f in files]
    return files


def save_checkpoint(filepath, model, optimizer, epoch):
    state = {
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }

    torch.save(state, filepath)


def kl_div(mean, logvar):
    kld = torch.mean(torch.sum(-0.5 * torch.sum(1 + logvar - mean**2 - logvar.exp(), dim=1), dim=1),
                     dim=0)

    return kld


def get_lambda(lat_dis_lambda, n_iter, lambda_schedule):
    if lambda_schedule == 0:
        return lat_dis_lambda
    else:
        return lat_dis_lambda * float(min(n_iter, lambda_schedule)) / lambda_schedule


def train(model, vae_optimizer, lat_dis_optimizer, train_loader, writer, epoch, debug):
    model.train()

    running_loss = 0
    running_loss_rec = 0
    running_loss_kl = 0
    running_loss_ce = 0
    counter = 1

    for x_t, label in train_loader:
        x_t, label = x_t.to(hparams.device), label.to(hparams.device)

        # lat_disの学習
        with torch.no_grad():
            z, _, _ = model.encode(x_t, label)

        discriminator_result = model.discriminate(z)
        lat_dis_loss = F.cross_entropy(discriminator_result, label)

        lat_dis_optimizer.zero_grad()
        lat_dis_loss.backward()
        lat_dis_optimizer.step()

        #  損失関数
        # reconstruction loss: 対数尤度の最大化
        # Gaussianの対数尤度の最大化 = MSE
        z, mean, logvar = model.encode(x_t, label)
        x_recon_t = model.decode(z, label)

        with torch.no_grad():
            discriminator_result = model.discriminate(z)

        reconstract_loss = F.mse_loss(x_recon_t, x_t)

        # kl loss: klダイバージェンスの最小化
        kl_loss = kl_div(mean, logvar)

        # CELoss
        total_iter = (epoch - 1) * len(train_loader) + counter
        ce_loss = F.cross_entropy(discriminator_result, label)
        ce_lambda = get_lambda(hparams.lat_dis_lambda, total_iter, hparams.lambda_schedule)

        # VAEのloss
        loss = reconstract_loss + hparams.beta * kl_loss - ce_lambda * ce_loss

        vae_optimizer.zero_grad()
        loss.backward()
        vae_optimizer.step()

        running_loss += loss.item()
        running_loss_rec += reconstract_loss.item()
        running_loss_kl += kl_loss.item()
        running_loss_ce += ce_loss.item()

        counter += 1

        if counter > 3 and debug == True:
            break

    denominator = len(train_loader)
    writer.add_scalar("train/model", running_loss / denominator, epoch)
    writer.add_scalar("train/reconstract", running_loss_rec / denominator, epoch)
    writer.add_scalar("train/kl", running_loss_kl / denominator, epoch)
    writer.add_scalar("train/ce", running_loss_ce / denominator, epoch)


def valid(model, valid_loader, writer, epoch, debug):
    model.eval()

    running_loss = 0
    running_loss_rec = 0
    running_loss_kl = 0
    running_loss_ce = 0
    counter = 1

    with torch.no_grad():
        for x_t, label in valid_loader:
            x_t, label = x_t.to(hparams.device), label.to(hparams.device)

            z, mean, logvar = model.encode(x_t, label)
            discriminator_result = model.discriminate(z)
            x_recon_t = model.decode(z, label)


            reconstract_loss = F.mse_loss(x_recon_t, x_t)
            kl_loss = kl_div(mean, logvar)
            ce_loss = F.cross_entropy(discriminator_result, label)
            loss = reconstract_loss + hparams.beta * kl_loss - hparams.lat_dis_lambda * ce_loss

            running_loss += loss.item()
            running_loss_rec += reconstract_loss.item()
            running_loss_kl += kl_loss.item()
            running_loss_ce += ce_loss.item()

            counter += 1

            if counter > 3 and debug == True:
                break

    denominator = len(valid_loader)
    writer.add_scalar("valid/model", running_loss / denominator, epoch)
    writer.add_scalar("valid/reconstract", running_loss_rec / denominator, epoch)
    writer.add_scalar("valid/kl", running_loss_kl / denominator, epoch)
    writer.add_scalar("valid/ce", running_loss_ce / denominator, epoch)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default=hparams.exp_name)
    parser.add_argument('--debug', type=bool, default=hparams.debug)

    args = parser.parse_args()

    # ロガーの作成
    writer = SummaryWriter(hparams.data_root / "log" / args.exp_name)

    # モデル
    model = FaderVC1D(hparams.mcep_channels, hparams.speaker_num).to(hparams.device)

    # optimizer
    vae_optimizer = torch.optim.Adam(model.vae.parameters(), lr=hparams.lr, betas=(0.9, 0.9))
    lat_dis_optimizer = torch.optim.Adam(model.lat_dis.parameters(),
                                         lr=hparams.lr, betas=(0.9, 0.9))

    with open(hparams.data_root / "speaker.json", 'r') as f:
        speaker_dict = json.load(f)

    with open(hparams.data_root / "mcep_statistics.json", 'r') as f:
        mcep_dict = json.load(f)

    train_files = files_to_list(hparams.data_root / "train_files.txt")
    valid_files = files_to_list(hparams.data_root / "valid_files.txt")

    # Create data loaders
    train_data = AudioDataset(hparams.data_root,
                              speaker_dict,
                              mcep_dict,
                              train_files,
                              seq_len=hparams.seq_len)
    valid_data = AudioDataset(hparams.data_root,
                              speaker_dict,
                              mcep_dict,
                              valid_files,
                              seq_len=hparams.seq_len)

    train_loader = DataLoader(train_data, batch_size=hparams.batch_size, num_workers=4)
    valid_loader = DataLoader(valid_data, batch_size=hparams.batch_size, num_workers=4)

    # model保存用ディレクトリ作成
    save_dir = hparams.data_root / "model_pth" / args.exp_name
    os.makedirs(save_dir, exist_ok=True)

    for epoch in tqdm(range(1, hparams.epochs + 1)):
        train(model, vae_optimizer, lat_dis_optimizer, train_loader, writer, epoch, args.debug)
        valid(model, valid_loader, writer, epoch, args.debug)

        if epoch == hparams.epochs:
            save_path = save_dir / "VAEVC-latest.pth"
            save_checkpoint(save_path, model, vae_optimizer, epoch)
        elif epoch % hparams.save_interval == 0:
            save_path = save_dir / f"VAEVC-{epoch:03}.pth"
            save_checkpoint(save_path, model, vae_optimizer, epoch)


if __name__ == "__main__":
    main()
