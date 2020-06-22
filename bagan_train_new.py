"""
(C) Copyright IBM Corporation 2018

All rights reserved. This program and the accompanying materials
are made available under the terms of the Eclipse Public License v1.0
which accompanies this distribution, and is available at
http://www.eclipse.org/legal/epl-v10.html
"""

from collections import defaultdict

import numpy as np

from optparse import OptionParser

import balancing_gan as bagan
from rw.batch_generator_from_input import BatchGenerator as BatchGenerator
from utils import save_image_array, save_image_files
import matplotlib.pyplot as plt
import os
# import sys
# sys.setrecursionlimit(10000)


def train_model(X_train, y_train, X_test, y_test, unbalance, target_classes, output_dir, input_dir, gan_epochs, dataset_name='CIFAR10'):

    print("Executing BAGAN.")

    # Read command line parameters
    seed = 0
    np.random.seed(seed)
    gratio_mode = "uniform"
    dratio_mode = "uniform"
    adam_lr = 0.00005
    opt_class = target_classes
    batch_size = 128
    out_dir = output_dir
    input_dir = input_dir

    channels = 3
    print('Using dataset: ', dataset_name)

    # Result directory
    res_dir = "{}/res_{}_dmode_{}_gmode_{}_epochs_{}_lr_{:f}_seed_{}".format(
        out_dir, dataset_name, dratio_mode, gratio_mode, epochs, adam_lr, seed
    )
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    # Read initial data.
    print("read input data...")
    bg_train = BatchGenerator(X_train, y_train, batch_size=batch_size)
    bg_test = BatchGenerator(X_test, y_test, batch_size=batch_size)

    print("input data loaded...")

    shape = bg_train.get_image_shape()
    print('shape here:', shape)

    min_latent_res = shape[-1]
    while min_latent_res > 8:
        min_latent_res = min_latent_res / 2
    min_latent_res = int(min_latent_res)

    classes = bg_train.get_label_table()

    # Initialize statistics information
    gan_train_losses = defaultdict(list)
    gan_test_losses = defaultdict(list)

    img_samples = defaultdict(list)

    # For all possible minority classes.
    target_classes = np.array(range(len(classes)))
    if opt_class is not None:
        min_classes = np.array(opt_class)
    else:
        min_classes = target_classes

    bg_train = BatchGenerator(BatchGenerator.TRAIN, batch_size,
                                  class_to_prune=c, unbalance=unbalance, dataset=dataset_name, input_dir=input_dir, GTSRB_size=g_size)

    # Train the model (or reload it if already available
    if not (
            os.path.exists("{}/class_{}_ratio_{}_score.csv".format(res_dir, min_classes, unbalance)) and
            os.path.exists("{}/class_{}_ratio_{}_discriminator.h5".format(res_dir, min_classes, unbalance)) and
            os.path.exists("{}/class_{}_ratio_{}_generator.h5".format(res_dir, min_classes, unbalance)) and
            os.path.exists(
                "{}/class_{}_ratio_{}_reconstructor.h5".format(res_dir, min_classes, unbalance))
    ):
        # Training required
        print("Required GAN for class {}".format(min_classes))

        print('Class counters: ', bg_train.per_class_count)

        # Train GAN to balance the data
        gan = bagan.BalancingGAN(
            target_classes, min_classes, dratio_mode=dratio_mode, gratio_mode=gratio_mode,
            adam_lr=adam_lr, res_dir=res_dir, image_shape=shape, min_latent_res=min_latent_res
        )
        gan.train(bg_train, bg_test, epochs=gan_epochs)
        gan.save_history(
            res_dir, min_classes
        )

    else:  # GAN pre-trained
        # Unbalance the training.
        print("Loading GAN for class {}".format(c))

        gan = bagan.BalancingGAN(target_classes, c, dratio_mode=dratio_mode, gratio_mode=gratio_mode,
                                    adam_lr=adam_lr, res_dir=res_dir, image_shape=shape, min_latent_res=min_latent_res)

        print('Load trained model')
        gan.load_models(
            "{}/class_{}_ratio_{}_discriminator.h5".format(
                res_dir, min_classes, unbalance),
            "{}/class_{}_ratio_{}_generator.h5".format(
                res_dir, min_classes, unbalance),
            "{}/class_{}_ratio_{}_reconstructor.h5".format(
                res_dir, min_classes, unbalance),
            bg_train=bg_train  # This is required to initialize the per-class mean and covariance matrix
        )

    for i in range(len(min_classes)):
        # Sample and save images
        c = min_classes[i]
        sample_size = 5000*unbalance[i]
        img_samples['class_{}'.format(c)] = gan.generate_samples(
            c=c, samples=sample_size)

        #save_image_array(np.array([img_samples['class_{}'.format(c)]]), '{}/plot_class_{}.png'.format(res_dir, c))
        print(np.array([img_samples['class_{}'.format(c)]])[0].shape)
        #plt.imshow(np.array([img_samples['class_{}'.format(c)]])[0][0])
        save_image_files(np.array([img_samples['class_{}'.format(c)]])[
                            0], c, unbalance[i],res_dir, dataset_name)
