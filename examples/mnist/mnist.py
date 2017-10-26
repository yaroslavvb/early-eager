# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A deep MNIST classifier using convolutional layers.

Sample usage:
  python mnist.py --help
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import time

import tensorflow as tf

from tensorflow.contrib.eager.python import tfe
from tensorflow.examples.tutorials.mnist import input_data

FLAGS = None


class MNISTModel(tfe.Network):
  """MNIST Network.

  Network structure is equivalent to:
  https://github.com/tensorflow/tensorflow/blob/r1.4/tensorflow/examples/tutorials/mnist/mnist_deep.py
  and
  https://github.com/tensorflow/models/blob/master/tutorials/image/mnist/convolutional.py

  But written using the Keras/tf.layers API.
  """

  def __init__(self, data_format):
    """Creates a model for classifying a hand-written digit.

    Args:
      data_format: Either 'channels_first' or 'channels_last'.
        'channels_first' is typically faster on GPUs while 'channels_last' is
        typically faster on CPUs. See
        https://www.tensorflow.org/performance/performance_guide#data_formats
    """
    super(MNISTModel, self).__init__(name='')
    if data_format == 'channels_first':
      self._input_shape = [-1, 1, 28, 28]
    else:
      assert data_format == 'channels_last'
      self._input_shape = [-1, 28, 28, 1]
    self.conv1 = self.track_layer(
        tf.layers.Conv2D(32, 5, data_format=data_format, activation=tf.nn.relu))
    self.conv2 = self.track_layer(
        tf.layers.Conv2D(64, 5, data_format=data_format, activation=tf.nn.relu))
    self.fc1 = self.track_layer(tf.layers.Dense(1024, activation=tf.nn.relu))
    self.fc2 = self.track_layer(tf.layers.Dense(10))
    self.dropout = self.track_layer(tf.layers.Dropout(0.5))
    self.max_pool2d = self.track_layer(
        tf.layers.MaxPooling2D(
            (2, 2), (2, 2), padding='SAME', data_format=data_format))

  def call(self, inputs, training):
    """Computes labels from inputs.

    Users should invoke __call__ to run the network, which delegates to this
    method (and not call this method directly).

    Args:
      inputs: A batch of images as a Tensor with shape [batch_size, 784].
      training: True if invoked in the context of training (causing dropout to
        be applied).  False otherwise.

    Returns:
      A Tensor with shape [batch_size, 10] containing the predicted logits
      for each image in the batch, for each of the 10 classes.
    """

    x = tf.reshape(inputs, self._input_shape)
    x = self.conv1(x)
    x = self.max_pool2d(x)
    x = self.conv2(x)
    x = self.max_pool2d(x)
    x = tf.layers.flatten(x)
    x = self.fc1(x)
    if training:
      x = self.dropout(x)
    x = self.fc2(x)
    return x


def loss(predictions, labels):
  return tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(
          logits=predictions, labels=labels))


def train_one_epoch(model, optimizer, dataset, log_interval=None):

  for (batch, (images, labels)) in enumerate(tfe.Iterator(dataset)):
    # TODO(ashankar): Remove this: b/67734394
    (images, labels) = (tf.identity(images), tf.identity(labels))

    # TODO(agarwal): switch to functools.partial to avoid this pylint issue.
    # pylint: disable=cell-var-from-loop
    def model_loss():
      return loss(model(images, training=True), labels)
    # pylint: enable=cell-var-from-loop

    optimizer.minimize(model_loss)
    if log_interval and batch % log_interval == 0:
      print('Batch #{}\tLoss: {:.6f}'.format(batch, model_loss().numpy()))


def test(model, dataset):
  (total_loss, total_accuracy, batches) = (0., 0., 0)

  def accuracy(labels, predictions):
    correct_prediction = tf.equal(
        tf.argmax(predictions, axis=1, output_type=tf.int64),
        tf.argmax(labels, axis=1, output_type=tf.int64))
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  for (images, labels) in tfe.Iterator(dataset):
    # TODO(ashankar): Remove this: b/67734394
    (images, labels) = (tf.identity(images), tf.identity(labels))
    predictions = model(images, training=False)
    total_loss += loss(predictions, labels).numpy()
    total_accuracy += accuracy(labels, predictions).numpy()
    batches += 1
  print('Test set: Average loss: {:.4f}, Accuracy: {:4f}%\n'.format(
      total_loss / batches, 100. * total_accuracy / batches))


def load_data(data_dir):
  """Returns training and test tf.data.Dataset objects."""
  data = input_data.read_data_sets(data_dir, one_hot=True)
  train_ds = tf.data.Dataset.from_tensor_slices((data.train.images,
                                                 data.train.labels))
  test_ds = tf.data.Dataset.from_tensors((data.test.images, data.test.labels))
  return (train_ds, test_ds)


def main(_):
  tfe.enable_eager_execution()

  (device, data_format) = ('/gpu:0', 'channels_first')
  if FLAGS.no_gpu or tfe.num_gpus() <= 0:
    (device, data_format) = ('/cpu:0', 'channels_last')
  print('Using device %s, and data format %s.' % (device, data_format))

  # Load the datasets
  (train_ds, test_ds) = load_data(FLAGS.data_dir)
  train_ds = train_ds.shuffle(60000).batch(FLAGS.batch_size)

  # Create the model and optimizer
  model = MNISTModel(data_format)
  optimizer = tf.train.MomentumOptimizer(FLAGS.lr, FLAGS.momentum)

  # TODO(apassos,josh11b): Demonstrate summaries/checkpoints.
  with tf.device(device):
    for epoch in range(1, 11):
      start = time.time()
      train_one_epoch(model, optimizer, train_ds, FLAGS.log_interval)
      end = time.time()
      print('\nTrain time for epoch #%d: %f' % (epoch, end - start))
      test(model, test_ds)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--data-dir',
      type=str,
      default='/tmp/tensorflow/mnist/input_data',
      help='Directory for storing input data')
  parser.add_argument(
      '--batch-size',
      type=int,
      default=64,
      metavar='N',
      help='input batch size for training (default: 64)')
  parser.add_argument(
      '--log-interval',
      type=int,
      default=10,
      metavar='N',
      help='how many batches to wait before logging training status')
  parser.add_argument(
      '--lr',
      type=float,
      default=0.01,
      metavar='LR',
      help='learning rate (default: 0.01)')
  parser.add_argument(
      '--momentum',
      type=float,
      default=0.5,
      metavar='M',
      help='SGD momentum (default: 0.5)')
  parser.add_argument(
      '--no-gpu',
      action='store_true',
      default=False,
      help='disables GPU usage even if a GPU is available')

  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
