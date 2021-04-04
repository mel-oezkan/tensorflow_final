import tensorflow as tf

class Encoder(tf.keras.layers.Layer):
    def __init__(self, latent_dims=32, fc_units=256):
        super(Encoder, self).__init__()

        self.fc_units = fc_units
        self.latent_dims = latent_dims

        # input: (batch, 64, 64, 1)

        self.conv1 = Conv2D(32, 4, 2, padding="same", activation="relu",
                    kernel_regularizer=tf.keras.regularizers.L1(0.01))

        self.conv2 = Conv2D(32, 4, 2, padding="same", activation="relu",
                    kernel_regularizer=tf.keras.regularizers.L1(0.01))

        self.conv3 = Conv2D(64, 4, 2, padding="same", activation="relu",
                    kernel_regularizer=tf.keras.regularizers.L1(0.01))

        self.conv4 = Conv2D(64, 4, 2, padding="same", activation="relu",
                    kernel_regularizer=tf.keras.regularizers.L1(0.01))

        # gets flattened here first
        self.flat = Flatten()
        self.fc1 = Dense(fc_units, activation="relu")

        self.re = Reshape([self.latent_dims, self.latent_dims])
        self.a = Dense(self.latent_dims * self.latent_dims, activation=None)

        self.mu = Dense(self.latent_dims, activation=None)

    def call(self, x):
        # INPUT: (batch, 64, 64, 32)

        z = self.conv1(x) #(batch, 32, 32, 32)
        z = self.conv2(z) #(batch, 16, 16, 32)
        z = self.conv3(z) #(batch,  8,  8, 64)
        z = self.conv4(z) #(batch,  4,  4, 64)

        z = self.flat(z) # (batch, 1024)
        z = self.fc1(z) # (batch, 256)

        mu = self.mu(z) # (batch, 32)

        A = self.a(z)   # (batch, 32x32)
        A = self.re(A)  # (batch, 32, 32)


        # band part is clipping for a matrix
        L = tf.linalg.band_part(A, -1, 0)

        # returns the batched diagonal part of a batched tensor
        # given input tensor get the rank values
        # Input shape: (batch_size,3,4) → (batch_size, 3)
        diag_A = tf.linalg.diag_part(A)

        # softplus is just relu with only positive values
        # probably used so the log_prob does not get nan
        diag = tf.nn.softplus(diag_A) + 1e-4


        L = tf.linalg.set_diag(L, diag)
        L_LT = tf.matmul(L, L, transpose_b=True)

        sigma = L_LT + 1e-4 * tf.eye(self.latent_dims)

        return mu, sigma

class Decoder(tf.keras.Model):
    """
    :param
    """
    def __init__(self, latent_dims=32, channels=1):
        super(Decoder, self).__init__()

        self.latent_dims = latent_dims
        self.channels = channels

        self.fc1 = Dense(self.latent_dims, activation="relu")
        self.fc2 = Dense(4*4*64, activation="relu")
        self.re = Reshape([4, 4, 64])

        self.t_conv1 = Conv2DTranspose(32, 4, 2, padding="same", activation="relu")
        self.t_conv2 = Conv2DTranspose(32, 4, 2, padding="same", activation="relu")
        self.t_conv3 = Conv2DTranspose(64, 4, 2, padding="same", activation="relu")
        self.x_logits = Conv2DTranspose(self.channels, 4, 2, padding="same", activation=None)

    def call(self, x):
        # INPUT: (batch, 32)

        z = self.fc1(x)
        z = self.fc2(z)
        z = self.re(z)

        z = self.t_conv1(z)
        z = self.t_conv2(z)
        z = self.t_conv3(z)

        x_logits = self.x_logits(z)

        return x_logits


class HyperVAE(tf.keras.Model):
    def __init__(self, latent_dims=4, channels=1, nu=200):
        """
        param latent_dims (int):
        the size of latent variables

        param channels (int):
        selecting either 1 = gray or 3 = (rgb/brg) etc.

        param nu :
        used initalize the degree of freedom when initalizing
        the Wishart dist

        param prior_cov:
        """
        super(CHyVAE, self).__init__()

        self.latent_dims = latent_dims
        self.channels = channels

        self.nu = nu
        self.prior_cov = np.eye(self.latent_dims)
        scale = max(self.nu - self.latent_dims - 1, 1)
        self.psi = scale * self.prior_cov

        self.fc_units = 128

        self._encoder = Encoder(latent_dims=self.latent_dims)
        self._decoder = Decoder(latent_dims=self.latent_dims)

        self.opt = tf.keras.optimizers.Adam(learning_rate=1e-4)

    def _regularizer(self, z, mu, sig, psi, nu, b):
        psi_zzT = psi + tf.matmul(z,z, transpose_b=True)
        mu = tf.expand_dims(mu, -1)
        sigma_mumuT_psi = sig + tf.matmul(mu, mu, transpose_b=True) + psi

        return -(
            .5 * (nu +1) * (tf.linalg.logdet(psi_zzT)) +
            .5 * tf.linalg.logdet(sig) -
            .5 * (nu + b) * tf.linalg.trace(tf.matmul(sigma_mumuT_psi, tf.linalg.inv(psi_zzT)))
        )


    def compare_reconstuction(self, test_images):
        n_images = len(test_images)

        for i, img in enumerate(test_images):
            # reconstruction given the test images
            # 1. create the latent embedding
            mu, sig = self._encoder(img)

            scale = tf.linalg.cholesky(sig)
            mvn = tfp.distributions.MultivariateNormalTriL(
                    loc=mu, scale_tril=scale)

            z_test = mvn.sample()

            # 2. generate an image from that embedding
            real_recon = self._decoder(z_test)
            real_recon = tf.nn.sigmoid(real_recon)

            # generating an image from a random noisy sample
            rand_noise = sample_noise(self.psi, self.nu, 10)
            fake_recon = self._decoder(rand_noise)
            fake_recon = tf.nn.sigmoid(fake_recon)

            plt.subplot(3, n_images, 1+i), plt.axis("off")
            plt.imshow(img[0, :,:,0], cmap="gray")

            plt.subplot(3, n_images, 1+i+(n_images*1)), plt.axis("off")
            plt.imshow(real_recon[0, :, :, 0], cmap="gray")

            plt.subplot(3, n_images, 1+i+(n_images*2)), plt.axis("off")
            plt.imshow(fake_recon[0, :, :, 0], cmap="gray")

        plt.show()

        # return x_recon, fake_recon
