import argparse
import os
import random

from models.beta_vae import BetaVAE


if __name__ == "__main__":

    # Parser 
    parser = argparse.ArgumentParser(
        description="Student-t Variational Autoencoder for Robust Density Estimation.")
    
    parser.add_argument(
        "--dataset", type=str, default="fashion-MNIST", 
        help="Select a dataset \n ellipse || fashion-MNIST")

    parser.add_argument(
        "--network", type=str, default="beta-VAE", 
        help="Selct the netowrk \n beta-VAE || spatial-VAE || hyperVAE")
    
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4, 
        help="Learning Rate for VAE.")
    
    parser.add_argument(
        "--seed", type=int, default=42, help="Random Seed.")