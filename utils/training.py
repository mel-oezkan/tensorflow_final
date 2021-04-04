def kl_divergence_loss(mean, std):
    kl_divergence = (0.5 * tf.reduce_sum(
        tf.math.exp(std) + tf.math.square(mean) - 1 - std, axis=1))

    return kl_divergence


def fit(model, data, epochs, metrics=['loss'], save_grads=False):

    (train_ds, test_ds, samples) = data

    metrics_names = ['loss']

    for epoch in range(1, epochs+1):

        train_loss = tf.keras.metrics.Mean()
        test_loss = tf.keras.metrics.Mean()

        start_time = time.time()
        pb_i = Progbar(num_training_samples, stateful_metrics=metrics_names)

        # run trough training data
        for train_batch in train_ds:
            loss, gradients, weights = train_step(model, train_x, optimizer)
            train_loss(loss)

        end_time = time.time()

        # TensorBoard: Train loss, weights, gradients
        with train_summary_writer.as_default():
            tf.summary.scalar('loss', train_loss.result(), step=epoch)
            for i, layer in enumerate(weights):
              tf.summary.histogram(f"weights: {layer.name}", gradients[i], step=epoch)
              tf.summary.histogram(f"gradients: {layer.name}", layer, step=epoch)


        # run trough test data
        for test_batch in test_ds:
