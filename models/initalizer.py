import tensorflow as tf
from .beta_vae import BetaVAE
from .broadcast_vae import BroadcastVAE
from .hyper_vae import HyperVAE

def initalize(model_name, latent_dims):
    if model_name == "beta-VAE":
        return BetaVAE(latent_dims)

    elif model_name == "spatial-VAE":
        return BroadcastVAE(latent_dims)

    elif model_name == "hyperVAE":
        return HyperVAE(latent_dims)
