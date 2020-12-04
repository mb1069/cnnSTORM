from src import data_module
from src.data.data_processing import process_STORM_tif
from src.train import LitModel, im_size
import argparse
import torch
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--tiff')
    parser.add_argument('-c', '--csvfile')
    parser.add_argument('-m', '--model',
                        default='/Users/miguelboland/Projects/uni/phd/smlm_z/cnnSTORM/src/trained_models/wandb_test.pth')
    return parser.parse_args()


def predict(model, psfs):
    model = LitModel.load_from_checkpoint(model, im_size=im_size)
    model = model.eval()

    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'

    model = model.to(device=device)
    psfs = torch.from_numpy(psfs).to(device=device)

    predictions = model(psfs).detach().cpu()
    return predictions


def main(model, tiff, csvfile):
    psfs, true_z = process_STORM_tif(tiff, csvfile)
    psfs = psfs.astype(data_module.dtype)
    psfs = psfs[:, np.newaxis, :, :]

    pred_z = predict(model, psfs)

    for true, pred in zip(true_z, pred_z):
        print(true, pred, abs(true-pred) / abs(true), abs(true-pred))


if __name__ == '__main__':
    args = parse_args()

    fname = '0'
    tiff = args.tiff or f'/Users/miguelboland/Projects/uni/phd/smlm_z/cnnSTORM/src/raw_data/{fname}.tif'
    csvfile = args.csvfile or f'/Users/miguelboland/Projects/uni/phd/smlm_z/cnnSTORM/src/raw_data/{fname}.csv'

    main(args.model, tiff, csvfile)
