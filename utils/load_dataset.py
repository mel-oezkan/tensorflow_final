from ellipse_generator import continous_ellipses_data


def data(ds="fashion-MNIST", bs=32, im_size=im_size):


    if ds == "fashion-MNIST":

        fashion_mnist = tf.keras.datasets.fashion_mnist
        (train_images, _), (test_images, _) = fashion_mnist.load_data()

        train_images = train_images / 255.0
        test_images = test_images / 255.0

        train_images = tf.expand_dims(train_images, -1)
        test_images = tf.expand_dims(test_images, -1)

        train_ds = tf.data.Dataset.from_tensor_slices(train_images)
        train_ds = train_ds.map(lambda img: tf.image.resize(img, [im_size, im_size]))
        train_ds = train_ds.shuffle(60000).batch(bs)
        train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

        test_ds = tf.data.Dataset.from_tensor_slices(test_images)
        test_ds = test_ds.map(lambda img: tf.image.resize(img, [im_size, im_size]))
        test_ds = test_ds.shuffle(10000).batch(bs)

        samples =

    else:
        train_ds, test_ds, samples = continous_ellipses_data(img_size=img_size)

    return train_ds, test_ds
