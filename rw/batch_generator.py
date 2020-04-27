"""
(C) Copyright IBM Corporation 2018

All rights reserved. This program and the accompanying materials
are made available under the terms of the Eclipse Public License v1.0
which accompanies this distribution, and is available at
http://www.eclipse.org/legal/epl-v10.html
"""

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import csv

class BatchGenerator:

    TRAIN = 1
    TEST = 0

    def readTrafficSigns(self, rootpath, output_path):
    
        images = []
        labels = []
        # loop over all 42 classes
        for c in range(0,43):
            prefix = rootpath + '/' + format(c, '05d') + '/' # subdirectory for class
            gtFile = open(prefix + 'GT-'+ format(c, '05d') + '.csv') # annotations file
            gtReader = csv.reader(gtFile, delimiter=';') # csv parser for annotations file
            next(gtReader)
            # loop over all images in current annotations file
            for row in gtReader:
                im = Image.open(prefix + row[0])
                im = im.resize((56, 56))
                images.append(np.asarray(im))
                #im.save(os.path.join(output_path, "{}_{}.png".format(c, row[0][:row[0].index('.')])), "PNG")
                labels.append(row[7]) # the 8th column is the label
            gtFile.close()
            
        return np.asarray(images), np.asarray(labels)


    def __init__(self, data_src, batch_size=32, class_to_prune=None, unbalance=0, dataset='MNIST', input_dir=None, 
                 GTSRB_size=None):
        assert dataset in ('MNIST', 'CIFAR10', 'GTSRB'), 'Unknown dataset: ' + dataset
        self.batch_size = batch_size
        self.data_src = data_src

        # Load data
        if dataset == 'MNIST':
            mnist = input_data.read_data_sets("dataset/mnist", one_hot=False)

            assert self.batch_size > 0, 'Batch size has to be a positive integer!'

            if self.data_src == self.TEST:
                self.dataset_x = mnist.test.images
                self.dataset_y = mnist.test.labels
            else:
                self.dataset_x = mnist.train.images
                self.dataset_y = mnist.train.labels

            # Normalize between -1 and 1
            self.dataset_x = (np.reshape(self.dataset_x, (self.dataset_x.shape[0], 28, 28)) - 0.5) * 2

            # Include 1 single color channel
            self.dataset_x = np.expand_dims(self.dataset_x, axis=1)

        elif dataset == 'CIFAR10':
            ((x, y), (x_test, y_test)) = tf.keras.datasets.cifar10.load_data()

            if self.data_src == self.TRAIN:
                self.dataset_x = x
                self.dataset_y = y
            else:
                self.dataset_x = x_test
                self.dataset_y = y_test

            # Arrange x: channel first
            self.dataset_x = np.transpose(self.dataset_x, axes=(0, 3, 1, 2))

            # Normalize between -1 and 1
            self.dataset_x = (self.dataset_x - 127.5) / 127.5

            # Y 1D format
            self.dataset_y = self.dataset_y[:, 0]

        elif dataset == 'GTSRB':
            input_path = 'GTSRB/Training'
            output_path = 'GTSRB/Training_png'

            if not os.path.isdir(output_path):
                os.mkdir(output_path, 0o666)
                
            X_train, labels = self.readTrafficSigns(input_path, output_path)
            y_train = labels.astype("int")
            indices = np.load('/content/drive/My Drive/BAGAN/GTSRB_index.npy').flatten()
            X_train = X_train[indices]
            y_train = y_train[indices]

            self.dataset_x = X_train
            self.dataset_y = y_train

            # Arrange x: channel first
            self.dataset_x = np.transpose(self.dataset_x, axes=(0, 3, 1, 2))

            # Normalize between -1 and 1
            self.dataset_x = (self.dataset_x - 127.5) / 127.5

        assert (self.dataset_x.shape[0] == self.dataset_y.shape[0])

        # Compute per class instance count.
        classes = np.unique(self.dataset_y)
        self.classes = classes
        per_class_count = list()
        for c in classes:
            per_class_count.append(np.sum(np.array(self.dataset_y == c)))

        self.count_0 = per_class_count[c]

        # Prune if needed!
        if class_to_prune is not None:
            all_ids = list(np.arange(len(self.dataset_x)))

            mask = [class_to_prune == lc for lc in self.dataset_y]
            all_ids_c = np.array(all_ids)[mask]
            np.random.shuffle(all_ids_c)

            other_class_count = np.array(per_class_count)
            other_class_count = np.delete(other_class_count, class_to_prune)
            to_keep = int(np.ceil(unbalance * max(
                other_class_count)))

            if dataset=='MNIST':
                to_keep = min([to_keep, 150])
            if dataset=='GTSRB':
                to_keep = GTSRB_size
            to_delete = all_ids_c[to_keep: len(all_ids_c)]

            to_keep = all_ids_c[:to_keep]

            count = 0
            print(len(to_keep))

            for id in to_keep:
                path = '{}/{}_{}_input_{}.png'.format(input_dir, dataset, class_to_prune, count)
                im = np.array(Image.open(path))
                if dataset == 'MNIST':
                    im = ((im[:,:,:1]/255)-0.5)*2
                else:
                    im = (im - 127.5)/127.5
                self.dataset_x[id] = np.transpose(im, (2, 0, 1))
                count += 1

            self.dataset_x = np.delete(self.dataset_x, to_delete, axis=0)
            self.dataset_y = np.delete(self.dataset_y, to_delete, axis=0)


        # Recount after pruning
        per_class_count = list()
        for c in classes:
            per_class_count.append(np.sum(np.array(self.dataset_y == c)))
        self.per_class_count = per_class_count
        self.count_1 = per_class_count[c]
        # List of labels
        self.label_table = [str(c) for c in range(len(np.unique(self.dataset_y)))]

        # Preload all the labels.
        self.labels = self.dataset_y[:]

        # per class ids
        self.per_class_ids = dict()
        ids = np.array(range(len(self.dataset_x)))
        for c in classes:
            self.per_class_ids[c] = ids[self.labels == c]

    def get_samples_for_class(self, c, samples=None):
        if samples is None:
            samples = self.batch_size

        np.random.shuffle(self.per_class_ids[c])
        to_return = self.per_class_ids[c][0:samples]
        return self.dataset_x[to_return]

    def get_label_table(self):
        return self.label_table

    def get_num_classes(self):
        return len( self.label_table )

    def get_class_probability(self):
        return self.per_class_count/sum(self.per_class_count)

    ### ACCESS DATA AND SHAPES ###
    def get_num_samples(self):
        return self.dataset_x.shape[0]

    def get_image_shape(self):
        return [self.dataset_x.shape[1], self.dataset_x.shape[2], self.dataset_x.shape[3]]

    def next_batch(self):
        dataset_x = self.dataset_x
        labels = self.labels

        indices = np.arange(dataset_x.shape[0])

        np.random.shuffle(indices)

        for start_idx in range(0, dataset_x.shape[0] - self.batch_size + 1, self.batch_size):
            access_pattern = indices[start_idx:start_idx + self.batch_size]
            access_pattern = sorted(access_pattern)

            yield dataset_x[access_pattern, :, :, :], labels[access_pattern]

