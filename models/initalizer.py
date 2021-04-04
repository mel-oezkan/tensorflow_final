import tensorflow as tf
from beta_vae import BetaVAE
from broadcast_vae import BroadcastVAE
from hyper_vae import HyperVAE

def initalize(model_name):
    if model_name == "beta-VAE":
        return BetaVAE

    elif model_name == "spatial-VAE":
        return BroadcastVAE

    elif model_name == "hyperVAE"
        return HyperVAE
