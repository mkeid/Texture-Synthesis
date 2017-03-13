import numpy as np
import os
import skimage
import skimage.io
import skimage.transform
from scipy.misc import toimage


# Return a numpy array of an image specified by its path
def load_image(path):
    # Load image [height, width, depth]
    img = skimage.io.imread(path) / 255.0
    assert (0 <= img).all() and (img <= 1.0).all()

    # Crop image from center
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    shape = list(img.shape)

    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    resized_img = skimage.transform.resize(crop_img, (shape[0], shape[1]))
    return resized_img, shape


# Return a resized numpy array of an image specified by its path
def load_image2(path, height=None, width=None):
    # Load image
    img = skimage.io.imread(path) / 255.0
    if height is not None and width is not None:
        ny = height
        nx = width
    elif height is not None:
        ny = height
        nx = img.shape[1] * ny / img.shape[0]
    elif width is not None:
        nx = width
        ny = img.shape[0] * nx / img.shape[1]
    else:
        ny = img.shape[0]
        nx = img.shape[1]
    return skimage.transform.resize(img, (ny, nx))


# Render the generated image given a tensorflow session and a variable image (x)
def render_img(session, x, save=False, out_path=None):
    shape = x.get_shape().as_list()
    img = np.clip(session.run(x), 0, 1)

    if save:
        toimage(np.reshape(img, shape[1:])).save(out_path)
    else:
        toimage(np.reshape(img, shape[1:])).show()