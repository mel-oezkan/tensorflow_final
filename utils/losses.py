import tensorflow as tf
import tensorflow_probability as tfp


def kl_divergence_loss(mean, std):
    kl_divergence = 0.5 * tf.reduce_sum(tf.math.exp(std) + tf.math.square(mean) - 1 - std, axis=1)
    
    return kl_divergence


def reparameterize(mean, std):
      epsilon = tf.random.normal(shape=mean.shape)
      return mean + epsilon * (1.0 / 2) * std


def hyper_loss(model, input_image):

    mu, sig = model._encoder(input_image)

    # get a sample
    mvn = tfp.distributions.MultivariateNormalFullCovariance(
            loc=mu, covariance_matrix=sig+1e-6)

    z = mvn.sample()
    z2 = tf.transpose(mvn.sample(1), perm=[1, 2, 0])

    x_hat_logits = model._decoder(z)

    # try out with cross entropy
    loglikelihood = tf.reduce_mean(tf.reduce_sum(
        tf.nn.sigmoid_cross_entropy_with_logits(
            labels=input_image, logits=x_hat_logits), [1, 2, 3]))
    loglikelihood = tf.cast(loglikelihood, dtype=tf.float64)


    regularizer = -tf.reduce_mean(regularizer(
                z2, mu, sig, model.psi, model.nu, 1))
    regularizer = tf.cast(regularizer, dtype=tf.double)
    
    loss = loglikelihood + regularizer

    return loss, regularizer, loglikelihood


@tf.function
def total_loss(model, x, beta=6, training=False):
    
    if model.model_name == "hyper-VAE":
        return hyper_loss(hyper_loss, x)

    # get the mean and sigma from the model
    mean, std = model.encode(x, training=training)

    z = reparameterize(mean, std)

    # Running the latent vector trough the decoder will yield the decoded(/generated) image 
    x_hat = model.decode(z, training=training)

    # reconstruction loss: compare input with output
    fidelity_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_hat, labels=x)
    fidelity_loss = -tf.reduce_sum(fidelity_loss, axis=[1, 2, 3])

    # compute loss of representation
    kl_divergence = kl_divergence_loss(mean, std)

    # compute the loss as a combination of reconstruction loss and kl diverigence
    loss = fidelity_loss + beta * kl_divergence

    return loss


# uses math formulars from the paper
def regularizer(z, mu, sig, psi, nu, b):
    # reformulation of the objective into subparts 
    # which are combinded together 

    # combining equation terms in variables so the calculation is easier
    psi_zzT = psi + tf.matmul(z,z, transpose_b=True)
    mu = tf.expand_dims(mu, -1)
    sigma_mumuT_psi = sig + tf.matmul(mu, mu, transpose_b=True) + psi

    return -(
        .5 * (nu +1) * (tf.linalg.logdet(psi_zzT)) +
        .5 * tf.linalg.logdet(sig) -
        .5 * (nu + b) * tf.linalg.trace(tf.matmul(sigma_mumuT_psi, tf.linalg.inv(psi_zzT)))
    )


