import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def plot_disentanglement(model, n, latent_dim, digit_size=28):

      # create array with normal distributed values
      norm = tfp.distributions.Normal(0, 1)
      normalize = 1.6448536
      grid_x = norm.quantile(np.linspace(0.05, 0.95, n))/normalize
      # shape of the plot
      image_width = digit_size*n
      image_height = digit_size*latent_dim
      # create black image
      image = np.zeros((image_height, image_width))
      image_latent = test_batch[0:1, :, :, :]

      # varied every single latent variable
      for i, yi in enumerate(range(latent_dim)):

            # get the latent dim of an example image
            mean, std = model.encode(image_latent)
            z_values = model.reparameterize(mean, std)
            z_values = tf.squeeze(z_values).numpy()

            # let the z_value take number of the normal distribution
            for j, xi in enumerate(grid_x):
                  z_values[i] = grid_x[j].numpy()
                  z = np.array([z_values])

                  # generate sample
                  x_decoded = model.sample(z)
                  # insert reconstruction into the full image
                  digit = tf.reshape(x_decoded[0], (digit_size, digit_size))
                  image[i * digit_size: (i + 1) * digit_size,
                        j * digit_size: (j + 1) * digit_size] = digit.numpy()
                  # insert white vertical edges
                  image[i * digit_size: (i + 1) * digit_size,
                        j * digit_size] = 0.4
                  # insert white horizontal edges
                  image[i * digit_size,
                        j * digit_size: (j + 1) * digit_size] = 0.4

      plt.figure(figsize=(10, 10))
      plt.imshow(image, cmap='Greys_r')
      plt.axis('Off')
      plt.show()


def generate_and_show_images(model, test_sample):
      
      # Forward step
      mean, logvar = model.encode(test_sample)
      z = model.reparameterize(mean, logvar)
      predictions = model.sample(z)
      
      plt.figure(figsize=(4, 4)), plt.gray()
      for i in range(predictions.shape[0]):
            plt.subplot(4, 4, i + 1), plt.axis('off')
            plt.imshow(predictions[i, :, :, 0])
            

      # tight_layout minimizes the overlap between 2 sub-plots
      plt.show()

      return predictions[i, :, :, 0]


# generating images for Tensorboard
def generate_images(model, test_sample):
  mean, logvar = model.encode(test_sample, training=False)
  z = model.reparameterize(mean, logvar)
  predictions = model.sample(z)
  return predictions