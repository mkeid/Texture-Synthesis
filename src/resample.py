#!/usr/bin/python3
# Mohamed K. Eid (mohamedkeid@gmail.com)
# Description: TensorFlow implementation of "Texture-Synthesis Using Convolutional Neural Networks"

import argparse
import custom_vgg19 as vgg19
import logging
import numpy as np
import os
import tensorflow as tf
import time
import utils
from functools import reduce

# Model hyperparams
TEXTURE_LAYERS = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
EPOCHS = 300
LEARNING_RATE = .02
TOTAL_VARIATION_SMOOTHING = 1.5
NORM_TERM = 6.

# Loss term weights
TEXTURE_WEIGHT = 3.
NORM_WEIGHT = .1
TV_WEIGHT = .1

# Default image paths
DIR_PATH = os.path.dirname(os.path.realpath(__file__))
OUT_PATH = DIR_PATH + '/../output/out_%.0f.jpg' % time.time()
INPUT_PATH, TEXTURE_PATH = None, None

# Logging params
PRINT_TRAINING_STATUS = True
PRINT_N = 100

# Logging config
log_dir = DIR_PATH + '/../log/'
if not os.path.isdir(log_dir):
    os.makedirs(log_dir)
    print('Directory "%s" was created for logging.' % log_dir)
log_path = ''.join([log_dir, str(time.time()), '.log'])
logging.basicConfig(filename=log_path, level=logging.INFO)


# Given an activated filter maps of any particular layer, return its respected gram matrix
def convert_to_gram(filter_maps):
    # Get the dimensions of the filter maps to reshape them into two dimenions
    dimension = filter_maps.get_shape().as_list()
    reshaped_maps = tf.reshape(filter_maps, [dimension[1] * dimension[2], dimension[3]])

    # Compute the inner product to get the gram matrix
    if dimension[1] * dimension[2] > dimension[3]:
        return tf.matmul(reshaped_maps, reshaped_maps, transpose_a=True)
    else:
        return tf.matmul(reshaped_maps, reshaped_maps, transpose_b=True)


# Compute the L2-norm divided by squared number of dimensions
def get_l2_norm_loss(diffs):
    shape = diffs.get_shape().as_list()
    size = reduce(lambda x, y: x * y, shape) ** 2
    sum_of_squared_diffs = tf.reduce_sum(tf.square(diffs))
    return sum_of_squared_diffs / size


# Compute texture loss given a variable image (x) and a texture image (s)
def get_texture_loss(x, s):
    with tf.name_scope('get_style_loss'):
        texture_layer_losses = [get_texture_loss_for_layer(x, s, l) for l in TEXTURE_LAYERS]
        texture_weights = tf.constant([1. / len(texture_layer_losses)] * len(texture_layer_losses), tf.float32)
        weighted_layer_losses = tf.multiply(texture_weights, tf.convert_to_tensor(texture_layer_losses))
        return tf.reduce_sum(weighted_layer_losses)


# Compute style loss for a layer (l) given the variable image (x) and the style image (s)
def get_texture_loss_for_layer(x, s, l):
    with tf.name_scope('get_style_loss_for_layer'):
        # Compute gram matrices using the activated filter maps of the art and generated images
        x_layer_maps = getattr(x, l)
        t_layer_maps = getattr(s, l)
        x_layer_gram = convert_to_gram(x_layer_maps)
        t_layer_gram = convert_to_gram(t_layer_maps)

        # Make sure the feature map dimensions are the same
        assert_equal_shapes = tf.assert_equal(x_layer_maps.get_shape(), t_layer_maps.get_shape())
        with tf.control_dependencies([assert_equal_shapes]):
            # Compute and return the normalized gram loss using the gram matrices
            shape = x_layer_maps.get_shape().as_list()
            size = reduce(lambda a, b: a * b, shape) ** 2
            gram_loss = get_l2_norm_loss(x_layer_gram - t_layer_gram)
            return gram_loss / size


# Compute total variation regularization loss term given a variable image (x) and its shape
def get_total_variation(x, shape):
    with tf.name_scope('get_total_variation'):
        # Get the dimensions of the variable image
        height = shape[1]
        width = shape[2]
        size = reduce(lambda a, b: a * b, shape) ** 2

        # Disjoin the variable image and evaluate the total variation
        x_cropped = x[:, :height - 1, :width - 1, :]
        left_term = tf.square(x[:, 1:, :width - 1, :] - x_cropped)
        right_term = tf.square(x[:, :height - 1, 1:, :] - x_cropped)
        smoothed_terms = tf.pow(left_term + right_term, TOTAL_VARIATION_SMOOTHING / 2.)
        return tf.reduce_sum(smoothed_terms) / size


# Parse arguments and assign them to their respective global variables
def parse_args():
    global TEXTURE_PATH, OUT_PATH

    parser = argparse.ArgumentParser()
    parser.add_argument("texture", help="path to the image you'd like to resample")
    parser.add_argument("--out", default=OUT_PATH, help="path to where the generated image will be created")
    args = parser.parse_args()

    # Assign image paths from the arg parsing
    TEXTURE_PATH = os.path.realpath(args.texture)
    OUT_PATH = os.path.realpath(args.out)


with tf.Session() as sess:
    parse_args()

    # Initialize and process art image to be used for our texture
    texture, image_shape = utils.load_image(TEXTURE_PATH)

    # Initialize the variable image that will become our final output as random noise
    noise = tf.Variable(tf.truncated_normal(image_shape, mean=.5, stddev=.1))

    with tf.name_scope('vgg_texture'):
        texture_model = vgg19.Vgg19()
        texture_model.build(texture, image_shape[1:])

    with tf.name_scope('vgg_x'):
        x_model = vgg19.Vgg19()
        x_model.build(noise, image_shape[1:])

    # Loss functions
    with tf.name_scope('loss'):
        # Texture
        if TEXTURE_WEIGHT is 0:
            texture_loss = tf.constant(0.)
        else:
            texture_loss = get_texture_loss(x_model, texture_model) * TEXTURE_WEIGHT

        # Norm regularization
        if NORM_WEIGHT is 0:
            norm_loss = tf.constant(0.)
        else:
            norm_loss = (get_l2_norm_loss(noise) ** NORM_TERM) * NORM_WEIGHT

        # Total variation denoising
        if TV_WEIGHT is 0:
            tv_loss = tf.constant(0.)
        else:
            tv_loss = get_total_variation(noise, image_shape) * TV_WEIGHT

        # Total loss
        total_loss = texture_loss + norm_loss + tv_loss

    # Update image
    with tf.name_scope('update_image'):
        optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
        grads = optimizer.compute_gradients(total_loss, [noise])
        clipped_grads = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in grads]
        update_image = optimizer.apply_gradients(clipped_grads)

    # Train
    logging.info("Initializing variables and beginning training..")
    sess.run(tf.global_variables_initializer())
    start_time = time.time()
    for i in range(EPOCHS):
        _, loss = sess.run([update_image, total_loss])
        if PRINT_TRAINING_STATUS and i % PRINT_N == 0:
            logging.info("Epoch %04d | Loss %.03f" % (i, loss))

    # FIN
    elapsed = time.time() - start_time
    logging.info("Training complete. The session took %.2f seconds to complete." % elapsed)
    logging.info("Rendering final image and closing TensorFlow session..")

    # Render the image after making sure the repo's dedicated output dir exists
    out_dir = os.path.dirname(os.path.realpath(__file__)) + '/../output/'
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    utils.render_img(sess, noise, save=True, out_path=OUT_PATH)
