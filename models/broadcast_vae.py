import tensorflow_probability as tfp
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, ReLU, InputLayer
import tensorflow as tf

class Encoder(tf.keras.layers.Layer):
    """"Encoder class for use in convolutional VAE

    Args:
        latent_dim: dimensionality of latent distribution

    Attributes:
        encoder_conv: convolution layers of encoder
        fc_mu: fully connected layer for mean in latent space
        fc_log_var: fully connceted layers for log variance in latent space
    """

    def __init__(self, latent_dim=6):
        super().__init__()
        self.latent_dim = latent_dim

        self.encoder_conv = tf.keras.Sequential(
            [
            # shape: [batch_size, 56, 56, 1]
            InputLayer(input_shape=(56, 56, 1)),

            # shape: [batch_size, 28, 28, 64 ]
            Conv2D(filters = 64,
                   kernel_size=4,
                   strides=2,
                   padding="same",
                   activation='relu'),

            # shape: [batch_size, 14, 14, 64]
            Conv2D(filters = 64,
                   kernel_size=4,
                   strides=2,
                   padding="same",
                   activation='relu'),

            # shape: [batch_size, 7, 7, 64]
            Conv2D(filters = 64,
                   kernel_size=4,
                   strides=2,
                   padding="same",
                   activation='relu'),

            # shape: [batch_size, 4, 4, 64]
            Conv2D(filters = 64,
                   kernel_size=4,
                   strides=2,
                   padding="same",
                   activation='relu'),

            # shape: [batch_size, 1024]
            Flatten(),

            # shape: [batch_size, 256]
            Dense(256),
            ReLU()
            ]
        )


        # shape: [batch_size, self.latent_dim]
        self.fc_mu = tf.keras.Sequential(
            Dense(self.latent_dim),
        )
        self.fc_log_var = tf.keras.Sequential(
            Dense(self.latent_dim),
        )

    def forward(self, inp):
        out = self.encoder_conv(inp)
        mu = self.fc_mu(out)
        log_var = self.fc_log_var(out)
        return [mu, log_var]


class SpatialBroadcastDecoder(tf.keras.layers.Layer):
    """SBD class for use in VAE, structure based on paper
       https://arxiv.org/pdf/1901.07017.pdf

    Args:
        latent_dim: dimensionality of latent distribution

    Attributes:
        img_size: image size (necessary for tiling)
        decoder_conv: convolution layers of decoder (also upsampling)
    """

    def __init__(self, latent_dim):
        super().__init__()

        self.latent_dim = latent_dim
        self.img_size = 56


        self.decoder_conv = tf.keras.Sequential(
            [
            # Input_shape [batch_size, 56, 56, latent_dim + 2]


            # shape: [batch_size, 56, 56, 64]
            Conv2D(filters = 64,
                   strides=(1, 1),
                   kernel_size=(3,3),
                   padding="same",
                   activation = "relu"),

           # shape [batch_size, 56, 56, 64]
            Conv2D(filters = 64,
                   strides=(1,1),
                   kernel_size=(3, 3),
                   padding="same",
                   activation = "relu"),

            # shape [batch_size, 56, 56, 1]
            Conv2D(filters = 1,
                   strides=(1,1),
                   kernel_size=(3, 3),
                   padding="same"),

            ]
        )


    def call(self, z):

        batch_size = z.shape[0]

        # broadcast (tile) latent sample of size k to image width w, height h

        h = w = self.img_size

        # z.shape [batch_size, latent_dim] LATENTS
        # z_b.shape [batch_size, 25088] TILED LATENTS
        z_b = tf.tile(z, [1, h * w])


        # Reshape tensor
        # z_b.shape [batch_size, 56, 56, latent_dim]
        z_b = tf.reshape(z_b, [batch_size, h, w, self.latent_dim])


        # Fixed coordinate channels  -->  X, Y COORDINATE CHANNELS
        x = tf.linspace(tf.constant(-1, tf.float32), tf.constant(1, tf.float32), w)
        y = tf.linspace(tf.constant(-1, tf.float32), tf.constant(1, tf.float32), w)


        # Reshape operations

        # shape [56, 56]
        xb, yb = tf.meshgrid(x, y)

        # shape [56, 56, 1]
        xb = tf.expand_dims(xb, 2)
        yb = tf.expand_dims(yb, 2)


        def concat(element):
          """ This function concatenates z_b, xb, y_b
          --> TILED LATENTS + X,Y COORDINATES  """

          # shape [56, 56, latent_dim +2]
          res = tf.concat(axis=2, values=[element, xb, yb])
          return res


        # shape [batch_size, 56, 56, latent_dim +2)]
        z_sb = tf.map_fn(lambda m: concat(m), z_b)


        # Apply convolutional layers (!unstrided!)
        mu_D = self.decoder_conv(z_sb)

        return mu_D


class BroadcastVAE(tf.keras.Model):
    """A simple VAE class

    Args:
        vae_tpe: type of VAE either 'Standard' or 'SBD'
        latent_dim: dimensionality of latent distribution
        fixed_var: fixed variance of decoder distribution
    """

    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim

        self.model_name = "spatial-VAE"

        self.decoder = SpatialBroadcastDecoder(latent_dim=latent_dim)
        self.encoder = Encoder(latent_dim=latent_dim)


    @tf.function
    def sample(self, epsilon=None):
        if epsilon is None:
            epsilon = tf.random.normal(shape=(100, self.latent_dim))
        
        return self.decode(epsilon, apply_sigmoid=True)


    def encode(self, x):
        mean, logvar = self.encoder.forward(x)
        return mean, logvar


    def decode(self, z, apply_sigmoid=False):
        # get decoder distribution parameters
        mu_D = self.decoder(z)

        # if sigmoid is applied
        if apply_sigmoid:
          probs = tf.sigmoid(mu_D)
          return probs

        return mu_D
