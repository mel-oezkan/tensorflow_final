import time 
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Progbar

from .losses import total_loss
from .plotting import generate_images, generate_and_show_images


def train_step(model, input_batch, optimizer, training=True, save_params=False):
    with tf.GradientTape() as tape:
    # compute loss together with regularization loss
        loss = total_loss(model, input_batch, training=training) + tf.reduce_sum(model.losses)
    
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    if save_params:
        return loss, gradients, model.trainable_weights
    
    return loss, None, None


def fit(model, data, epochs, optimizer, data_samples, 
    train_summary_writer, test_summary_writer, image_summary_writer,
    metrics=['loss'], save_params=False):

    (train_ds, test_ds, samples) = data
    num_training_samples, _ = data_samples


    metrics_names = ['loss']

    for epoch in range(1, epochs+1):

        train_loss = tf.keras.metrics.Mean()
        test_loss = tf.keras.metrics.Mean()


        pb_i = Progbar(num_training_samples, stateful_metrics=metrics_names)

        avg_loss = []
        # run trough training data
        for train_batch in train_ds:
            loss, gradients, weights = train_step(model, train_batch, optimizer, save_params=save_params)
            train_loss(loss)

            avg_loss.append(loss)

            pb_i.add(1, values=[("loss", loss)] )

        # TensorBoard: Train loss, weights, gradients
        with train_summary_writer.as_default():
            tf.summary.scalar('loss', train_loss.result(), step=epoch)
            if save_params:
                for i, layer in enumerate(weights):
                    tf.summary.histogram(f"weights: {layer.name}", gradients[i], step=epoch)
                    tf.summary.histogram(f"gradients: {layer.name}", layer, step=epoch)


        # run trough test data
        for test_batch in test_ds:
            test_loss(total_loss(model, test_batch, training=False))

        with test_summary_writer.as_default():
            tf.summary.scalar('loss', test_loss.result(), step=epoch)

        with image_summary_writer.as_default():
            tf.summary.image("Reconstruction", generate_images(model, samples), step=epoch, max_outputs=8)
            tf.summary.image("Images", samples, step=epoch, max_outputs=8)

        # reset losses
        train_loss.reset_states()
        test_loss.reset_states()

        # generate and save images
        generate_and_show_images(model, samples)
