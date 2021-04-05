import tensorflow as tf


class Encoder(tf.keras.layers.Layer):
    """ Encoder """

    def __init__(self, latent_dim):
        super(Encoder, self).__init__()

        self.latent_dim = latent_dim
        l2 = tf.keras.regularizers.l2(0.1)

        self.conv1 = tf.keras.layers.Conv2D(
            filters=32, kernel_size=3, strides=(2, 2), activation='relu',
            kernel_regularizer=l2, bias_regularizer=l2
        )

        self.conv2 = tf.keras.layers.Conv2D(
            filters=64, kernel_size=3, strides=(2, 2), activation='relu',
            bias_regularizer=l2
        )

        self.conv3 = tf.keras.layers.Conv2D(
            filters=64, kernel_size=3, strides=(2, 2), activation='relu',
            bias_regularizer=l2
        )

        self.flatten = tf.keras.layers.Flatten()

        self.mu = tf.keras.layers.Dense(
            self.latent_dim, name="mu",
            kernel_regularizer=l2, bias_regularizer=l2
        )

        self.sig = tf.keras.layers.Dense(
            self.latent_dim, activation="relu", name="sigma"
        )


    def call(self, x, training=False):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)

        mu = self.mu(x)
        sig = self.sig(x)

        return mu, sig


class Decoder(tf.keras.Model):
    """ Decoder """

    def __init__(self):
        super(Decoder, self).__init__()

        l2 = tf.keras.regularizers.l2(0.1)

        self.inLayer = tf.keras.layers.Dense(
            units=7*7*64, activation=tf.nn.relu)
        self.shape = tf.keras.layers.Reshape(target_shape=(7, 7, 64))
        self.convT1 = tf.keras.layers.Conv2DTranspose(
            filters=64, kernel_size=3, strides=2,
            padding='same', activation='relu',
            bias_regularizer=l2
        )

        self.convT1_1 = tf.keras.layers.Conv2DTranspose(
            filters=64, kernel_size=3, strides=2,
            padding='same', activation='relu',
            bias_regularizer=l2
        )

        self.convT2 = tf.keras.layers.Conv2DTranspose(
            filters=32, kernel_size=3, strides=2,
            padding='same', activation='relu',
            bias_regularizer=l2
        )

        self.convT3 = tf.keras.layers.Conv2DTranspose(
            filters=1, kernel_size=3, strides=1, padding='same',
            kernel_regularizer=l2, bias_regularizer=l2
        )


    def call(self, x, training=False):
        x = self.inLayer(x)
        x = self.shape(x)
        x = self.convT1(x)
        x = self.convT1_1(x)
        x = self.convT2(x)
        x = self.convT3(x)

        return x


class BetaVAE(tf.keras.Model):
    """Disentangled Variational Autoencoder with"""

    def __init__(self, latent_dim):
        super(BetaVAE, self).__init__()
        self.latent_dim = latent_dim

        self.model_name = "beta-VAE"

        # Define the model as encoder and decoder
        self.encoder = Encoder(latent_dim=self.latent_dim)
        self.decoder = Decoder()


    # returns sample from a normal distribution
    @tf.function
    def sample(self, epsilon=None):
        if epsilon is None:
            epsilon = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(epsilon, apply_sigmoid=True)


    # pass data through the encoder
    def encode(self, x, training=False):
        mean, logvar = self.encoder(x, training=training)
        return mean, logvar


    # pass data through the decoder
    def decode(self, z, training=False, apply_sigmoid=False):
        logits = self.decoder(z, training=training)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits
