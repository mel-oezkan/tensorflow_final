import argparse
import os
import random
import numpy as np
import tensorflow as tf

# from models.beta_vae import BetaVAE
from utils import load_dataset, training
from models import initalizer


if __name__ == "__main__":

    # Parser
    parser = argparse.ArgumentParser(
        description="Student-t Variational Autoencoder for Robust Density Estimation.")

    parser.add_argument(
        "--data", type=str, default="fashion-MNIST",
        help="Select a dataset: \n ellipse || fashion-MNIST")

    parser.add_argument(
        "--net", type=str, default="beta-VAE",
        help="Selct the netowrk: \n beta-VAE || spatial-VAE || hyper-VAE")

    parser.add_argument(
        "--lr", type=float, default=1e-4,
        help="Learning Rate for the VAE.")

    parser.add_argument(
        "--seed", type=int, default=42, help="Random Seed.")

    parser.add_argument(
        "--batch_size", type=int, default=50, help="Random Seed.")


    args = parser.parse_args()


    # initalize parameters and check if correct
    dataset = args.dataset
    assert dataset in ["ellipse", "fashion-MNIST"], "Selected dataset is not supported"

    net = args.decoder
    assert net in ["beta-VAE", "spatial-VAE", "hyperVAE"]

    lr = args.learning_rate
    seed = args.seed

    batch_size = args.batch_size


    # select seeds
    random.seed(seed)
    np.random.seed(seed)


    # check for image size of the networks
    if net == "hyperVAE":
        im_size = 64
    else:
        im_size = 56


    # define parameters for the network
    model = initalizer(net)
    optimizer = tf.keras.optimize.Adam(learning_rate=lr)


    # load the data as (train_ds, test_ds, samples)
    data = load_dataset.data(ds=dataset, bs=batch_size, im_size=im_size)


    # start the training loop
    training.fit(model, )










# end
