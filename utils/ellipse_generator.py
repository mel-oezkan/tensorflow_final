from skimage.draw import ellipse
import tensorflow as tf
import numpy as np


def create_ellipse(im_size, xPos, yPos, scale, rotation):
    # xPos: 0-31 (left to right)
    # yPos: 0-31(top to bottom)
    # scale: 0-5 (small to big)
    # rotation: 0-39

    # check if parameters are valid
    if (xPos < 0 or xPos > 31 or yPos < 0 or yPos > 31 or
        scale < 0 or scale > 5 or rotation < 0 or rotation > 39):
        raise Exception("Ellipse Parameters not in accepted range")

    """ Adjust parameters """
    # Stretch of the Ellipse
    stretch = 2
    # guarantee a min size:
    scale += 2
    # width and height according to scale and stretch factor
    width = int(scale * 0.015*im_size)
    height = int(scale*stretch * 0.015*im_size)
    # adjust coordinates to imsize
    xPos = int(xPos/39 * (im_size-2*height) + height)
    yPos = int(yPos/39 * (im_size-2*height) + height)
    # convert rotation factor to 0-pi
    rotation = rotation/39 * np.pi

    """ Create Image """
    # Image with Ellipses
    image = np.zeros((im_size, im_size))
    rr, cc = ellipse(xPos, yPos, width, height, rotation=rotation)
    image[rr,cc] = 255

    return image



def generate_continous_params(xPos, dirX, yPos, dirY, scale, dirScale, orientation, dirOrient):
    """ the ellipse will move continously right/left and up/down in pixel space
      and will change its orientation continuously """

    # increase every variable slightly
    xPos += dirX * np.random.randint(3)
    yPos += dirY * np.random.randint(3)
    # increase or decrease scale and orientation
    scale += dirScale * np.random.randint(2)
    orientation += dirOrient * np.random.randint(2)

    # change direction of change if variables reach limits
    if xPos >= 30: dirX = -1
    if xPos <= 1: dirX = 1

    if yPos >= 30: dirY = -1
    if yPos <= 1: dirY = 1

    if scale >= 5: dirScale = -1
    if scale <= 0: dirScale = 1

    if orientation >= 39: dirOrient = -1
    if orientation <= 0: dirOrient = 1

    return xPos, dirX, yPos, dirY,scale, dirScale, orientation, dirOrient


def continous_ellipses_data(n_trainImages=30000, n_testImages=5000, batch_size=32, im_size=28):
    """
    creates a tf.dataset of correlated ellipses

    Args:
        n_trainImages(int)
        n_testImages(int)
        batch_size(int)

    Returns:
        train_ds(tf.data)
        test_ds(tf.data)
    """

    assert n_trainImages % batch_size == 0, f"Choose a different batch size. \n {batch_size} does not divide {n_trainImages} without rest"
    assert n_testImages % batch_size == 0, f"Choose a different batch size. \n {batch_size} does not divide {n_testImages} without rest"

    train_images = []
    xPos, yPos, scale, orient = 0, 0, 0, 0
    dirX, dirY, dirScale, dirOrient = 1, 1, 1, 1
    for _ in range(n_trainImages):
        # continuous generative factors
        xPos, dirX, yPos, dirY, scale, dirScale, orient, dirOrient = generate_continous_params(xPos, dirX, yPos, dirY, scale, dirScale, orient, dirOrient)
        # generate image and append to list
        train_images.append(create_ellipse(im_size, xPos, yPos, scale, orient))

    test_images = []
    xPos, yPos, scale, rotation = 0, 0, 0, 0
    for i in range(n_testImages):
        xPos = np.random.randint(0,32)
        yPos = np.random.randint(0,32)
        scale = np.random.randint(0,6)
        rotation = np.random.randint(0,40)
        test_images.append(
            create_ellipse(im_size, xPos, yPos, scale, rotation))

    train_ds = tf.convert_to_tensor(train_images)
    test_images = tf.convert_to_tensor(test_images)

    train_images = tf.expand_dims(train_images, axis=-1)
    test_images = tf.expand_dims(test_images, axis=-1)

    train_ds = tf.data.Dataset.from_tensor_slices(train_images)
    test_ds = tf.data.Dataset.from_tensor_slices(test_images)

    train_ds = train_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_ds, test_ds
