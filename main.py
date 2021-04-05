
import os
import random
import argparse
import datetime
import numpy as np
import tensorflow as tf
from tensorboard import notebook


# from models.beta_vae import BetaVAE
from utils import load_dataset, training
from models.initalizer import initalize


if __name__ == "__main__":

    # Parser
    parser = argparse.ArgumentParser(
        description="Student-t Variational Autoencoder for Robust Density Estimation.")

    parser.add_argument(
        "--data", type=str, default="ellipse",
        help="Select a dataset: \n ellipse || fashion-MNIST")

    parser.add_argument(
        "--net", type=str, default="beta-VAE",
        help="Select the netowrk: \n beta-VAE || spatial-VAE || hyper-VAE")

    parser.add_argument(
        "--epochs", type=int, default="20",
        help="Select a epoch size which divides 30000 without rest")

    parser.add_argument(
        "--batch_size", type=int, default=50, help="Random Seed.")

    parser.add_argument(
        "--latent_dims", type=int, default=4, help="Select the latent dimension")

    parser.add_argument(
        "--lr", type=float, default=1e-4,
        help="Learning Rate for the VAE.")

    parser.add_argument(
        "--seed", type=int, default=42, help="Random Seed.")

    

    args = parser.parse_args()


    # initalize parameters and check if correct
    dataset = args.data
    assert dataset in ["ellipse", "fashion-MNIST"], "Selected dataset is not supported"

    net = args.net
    assert net in ["beta-VAE", "spatial-VAE", "hyperVAE"]

    epochs = args.epochs
    latent_dims = args.latent_dims
    batch_size = args.batch_size
    lr = args.lr
    seed = args.seed

    

    # select seeds
    random.seed(seed)
    np.random.seed(seed)


    # check for image size of the networks
    if net == "hyperVAE":
        im_size = 64
    else:
        im_size = 56


    # TENSORBOARD
    # create log file directory
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
    test_log_dir = 'logs/gradient_tape/' + current_time + '/test'
    train_writer = tf.summary.create_file_writer(train_log_dir)
    test_writer = tf.summary.create_file_writer(test_log_dir)
    # create directory for images
    logdir = "logs/image_data/" + current_time
    image_writer = tf.summary.create_file_writer(logdir)


    # define parameters for the network
    model = initalize(net, latent_dims)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)


    # load the data as (train_ds, test_ds, samples), (train_samples, test_samples)
    data, data_samples = load_dataset.data(ds=dataset, bs=batch_size, im_size=im_size)


    # start the training loop
    training.fit(model, data, epochs, optimizer,data_samples,
        train_writer, test_writer, image_writer)

    notebook.display(port=6006, height=1000)










# end
