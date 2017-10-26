# TensorFlow Eager Execution

## What is this?

Eager execution is a feature that makes TensorFlow execute operations
immediately: concrete values are returned, instead of a computational graph to
be executed later.

As a result, enabling eager execution provides:

-   A [NumPy](http://www.numpy.org/)-like library for numerical computation with
    support for GPU acceleration and automatic differentiation.
-   A flexible platform for machine learning research and experimentation.

Eager execution is under active development. This guide walks through an
alpha/preview release. In particular, not all TensorFlow APIs currently work
with eager execution enabled, and some models may be slow to execute, compared
to models defined without using eager execution.

## Installation

Eager execution is **not** included in the latest release (version 1.4) of
TensorFlow. To use it, you will need to [build TensorFlow from
source](https://www.tensorflow.org/install/install_sources) or install the
nightly builds.

The nightly builds can be installed using `pip`:

-   `pip install tf-nightly` (for CPU-only TensorFlow)
-   `pip install tf-nightly-gpu` (for GPU-enabled TensorFlow)

Or using `docker`:

```sh
# For CPU-only TensorFlow
docker pull tensorflow/tensorflow:nightly
docker run -it -p 8888:8888 tensorflow/tensorflow:nightly

# For GPU-enabled TensorFlow:
# (Requires https://github.com/NVIDIA/nvidia-docker)
nvidia-docker pull tensorflow/tensorflow:nightly-gpu
nvidia-docker run -it -p 8888:8888 tensorflow/tensorflow:nightly-gpu
```

## Getting Started

With TensorFlow installed, eager execution is enabled via a single call:

```python
import tensorflow as tf

from tensorflow.contrib.eager.python import tfe

tfe.enable_eager_execution()
```

Enabling eager execution changes how TensorFlow functions behave (in particular,
`Tensor` objects will reference concrete values instead of being symbolic
handles to nodes in a computational graph). As a result, eager execution should
be enabled at the beginning of a program and cannot be disabled afterwards in
the same program.

Code examples in the rest of this guide assume that eager execution has been
enabled.

## A library for numerical computation

A significant fraction of the [TensorFlow
API](https://www.tensorflow.org/api_docs/python/) consists of numerical
operations: [arithmetic
operations](https://www.tensorflow.org/api_docs/python/tf/matmul), [matrix
operations](https://www.tensorflow.org/api_docs/python/tf/matmul), [linear
algebra operations](https://www.tensorflow.org/api_docs/python/tf/linalg), etc.

With eager execution enabled, these operations consume and return
multi-dimensional arrays as `Tensor` objects, similar to NumPy `ndarray`s. For
example:

```python
# Multiply two 2x2 matrices
x = tf.matmul([[1, 2],
               [3, 4]],
              [[4, 5],
               [6, 7]])
# Add one to each element
# (tf.add supports broadcasting)
y = tf.add(x, 1)

# Create a random random 5x3 matrix
z = tf.random_uniform([5, 3])

print(x)
print(y)
print(z)
```

Output:

```
tf.Tensor(
[[16 19]
 [36 43]], shape=(2, 2), dtype=int32)
tf.Tensor(
[[17 20]
 [37 44]], shape=(2, 2), dtype=int32)
tf.Tensor(
[[ 0.25058532  0.0929395   0.54113817]
 [ 0.3108716   0.93350542  0.84909797]
 [ 0.53081679  0.12788558  0.01767385]
 [ 0.29725885  0.33540785  0.83588314]
 [ 0.38877153  0.39720535  0.78914213]], shape=(5, 3), dtype=float32)
```

For convenience, these operations can also be triggered via operator overloading
of the `Tensor` object. For example, the `+` operator is equivalent to `tf.add`,
`-` to `tf.subtract`, `*` to `tf.multiply`, etc.:

```python
x = (tf.ones([1], dtype=tf.float32) + 1) * 2 - 1
print(x)
```

Output:

```
tf.Tensor([ 3.], shape=(1,), dtype=float32)
```

### Converting to and from NumPy

The operations above automatically convert Python objects (like lists of
numbers) and NumPy arrays to `Tensor` objects. `Tensor` objects can also be used
as NumPy arrays by numpy operations.

```python
import numpy as np

x = tf.add(1, 1)                     # tf.Tensor with a value of 2
y = tf.add(np.array(1), np.array(1)) # tf.Tensor with a value of 2
z = np.multiply(x, y)                # numpy.int64 with a value of 4
```

Alternatively, they can be explicitly converted using
[`tf.constant`](https://www.tensorflow.org/api_docs/python/tf/constant), as
shown in the next example.

Conversely, a NumPy `ndarray` corresponding to a `Tensor` can be obtained using
its `.numpy()` method. For example:

```python
import numpy as np

np_x = np.array(2., dtype=np.float32)
x = tf.constant(np_x)

py_y = 3.
y = tf.constant(py_y)

z = x + y + 1

print(z)
print(z.numpy())
```

Output:

```
tf.Tensor(6.0, shape=(), dtype=float32)
6.0
```

### GPU acceleration

Many TensorFlow operations support GPU acceleration. With eager execution
enabled, [computation is not automatically
offloaded](https://www.tensorflow.org/tutorials/using_gpu) to GPUs. Instead, you
must explicitly specify when GPUs should be used.

The simplest way to do this is to enclose your computation in a `with
tf.device('/gpu:0')` block. Also of interest is the `tfe.num_gpus()` function,
which returns the number of available GPUs.

For example, consider this snippet to measure the time to multiply two 1000x1000
matrices on CPU:

```python
import time

def measure(x):
  # The very first time a GPU is used by TensorFlow, it is initialized.
  # So exclude the first run from timing.
  tf.matmul(x, x)

  start = time.time()
  for i in range(10):
    tf.matmul(x, x)
  end = time.time()

  return "Took %s seconds to multiply a %s matrix by itself 10 times" % (end - start, x.shape)

# Run on CPU:
with tf.device("/cpu:0"):
  print("CPU: %s" % measure(tf.random_normal([1000, 1000])))

# If a GPU is available, run on GPU:
if tfe.num_gpus() > 0:
  with tf.device("/gpu:0"):
    print("GPU: %s" % measure(tf.random_normal([1000, 1000])))
```

Output (exact numbers will depend on the characteristics of the hardware):

```python
CPU: Took 0.145531892776 seconds to multiply a (1000, 1000) matrix by itself 10 times
GPU: Took 0.000458955764771 seconds to multiply a (1000, 1000) matrix by itself 10 times
```

Alternatively, methods on the `Tensor` object can be used to explicitly copy the
`Tensor` to a different device. Operations are typically executed on the device
on which the inputs are placed. For example:

```python
x = tf.random_normal([10, 10])

x_gpu0 = x.gpu()
x_cpu = x.cpu()

_ = tf.matmul(x_cpu, x_cpu)  # Runs on CPU
_ = tf.matmul(x_gpu0, x_gpu0)  # Runs on GPU:0

if tfe.num_gpus() > 1:
  x_gpu1 = x.gpu(1)
  _ = tf.matmul(x_gpu1, x_gpu1)  # Runs on GPU:1
```

### Automatic Differentiation

[Automatic
differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation) is an
integral part of many machine learning algorithms (e.g.,
[backpropagation](https://en.wikipedia.org/wiki/Backpropagation) for training
neural networks). For this purpose, TensorFlow eager execution provides an
[autograd](https://github.com/HIPS/autograd)-style API for automatic
differentiation. Specifically, the functions:

-   `tfe.gradients_function(f)`: Returns a Pyton function that computes the
    derivatives of the Python function `f` with respect to its arguments. `f`
    must return a scalar value. When the returned function is invoked, it
    returns a list of `Tensor` objects (one element for each argument of `f`).
-   `tfe.value_and_gradients_function(f)`: Similar to `tfe.gradients_function`,
    except that when the returned function is invoked, it returns the value of
    `f` in addition to the list of derivatives of `f` with respect to its
    arguments.

These functions naturally apply to higher order differentiation as well. For
example:

```python
def f(x):
  return tf.multiply(x, x) # Or x * x
assert 9 == f(3.).numpy()

df = tfe.gradients_function(f)
assert 6 == df(3.)[0].numpy()

# Second order deriviative.
d2f = tfe.gradients_function(lambda x: df(x)[0])
assert 2 == d2f(3.)[0].numpy()

# Third order derivative.
d3f = tfe.gradients_function(lambda x : d2f(x)[0])
assert 0 == d3f(3.)[0].numpy()
```

These functions can be used to train models. For example, consider the following
simple linear regression model:

```python
def prediction(input, weight, bias):
  return input * weight + bias

# A toy dataset of points around 3 * x + 2
NUM_EXAMPLES = 1000
training_inputs = tf.random_normal([NUM_EXAMPLES])
noise = tf.random_normal([NUM_EXAMPLES])
training_outputs = training_inputs * 3 + 2 + noise

# A loss function: Mean-squared error
def loss(weight, bias):
  error = prediction(training_inputs, weight, bias) - training_outputs
  return tf.reduce_mean(tf.square(error))

# Function that returns the the derivative of loss with respect to
# weight and bias
grad = tfe.gradients_function(loss)

# Train for 200 steps (starting from some random choice for W and B).
W = 5.
B = 10.
learning_rate = 0.01
print("Initial loss: %f" % loss(W, B).numpy())
for i in range(200):
  (dW, dB) = grad(W, B)
  W -= dW * learning_rate
  B -= dB * learning_rate
  if i % 20 == 0:
    print("Loss at step %d: %f" % (i, loss(W, B).numpy()))
print("Final loss: %f" % loss(W, B).numpy())
print("W, B = %f, %f" % (W.numpy(), B.numpy()))
```

Output: (the exact numbers may vary depending on the randomness in noise)

```
Initial loss: 66.730003
Loss at step 0: 64.200096
Loss at step 20: 29.872814
Loss at step 40: 14.233772
Loss at step 60: 7.090570
Loss at step 80: 3.819887
Loss at step 100: 2.318821
Loss at step 120: 1.628385
Loss at step 140: 1.310142
Loss at step 160: 1.163167
Loss at step 180: 1.095162
Final loss: 1.064711
W, B = 3.094944, 2.161383
```

To utilize the GPU, place the code above within a `with tf.device("/gpu:0"):`
block. (However, this particular model, with two floating point parameters, is
unlikely to benefit from any GPU acceleration.)

## Building and training models

In practice, your computation may have many parameters to be optimized (by
computing derivatives), and encapsulating them into re-usable classes/objects
makes the code easier to follow than writing a single top-level function with
many arguments.

In fact, eager execution encourages use of the [Keras](https://keras.io)-style
"Layer" classes in the
[`tf.layers`](https://www.tensorflow.org/api_docs/python/tf/layers) module.

Furthermore, you may want to apply more sophisticated techniques to compute
parameter updates, such as those in
[`tf.train.Optimizer`](https://www.tensorflow.org/api_guides/python/train#Optimizers)
implementations.

This next section walks through using the same `Optimizer` and `Layer` APIs used
to build trainable TensorFlow graphs in an environment where eager execution is
enabled.

### Variables and Optimizers

`tfe.Variable` objects store mutable `Tensor` values that can be accessed during
training, making automatic differentiation easier. In particular, parameters of
a model can be encapsulated in Python classes as variables.

`tfe.gradients_function(f)` introduced earlier computes the derivatives of `f`
with respect to its arguments, which would require all parameters of interest to
be arguments of `f`, which can become cumbersome.

`tfe.implicit_gradients` is an alternative function with some useful properties:

-   It computes the derivatives of `f` with respect to all the `tfe.Variable`s
    used by `f`.
-   When the returned function is invoked, it returns a list of tuples of
    (gradient_value, variable object) pairs.

Representing model parameters as `Variable` objects, along with the use of
`tfe.implicit_gradients` typically results in better encapsulation. For example,
the linear regression model described above can be written into a class:

```python
class Model(object):
  def __init__(self):
    self.W = tfe.Variable(5., name='weight')
    self.B = tfe.Variable(10., name='bias')

  def predict(self, inputs):
    return inputs * self.W + self.B


# The loss function to be optimized
def loss(model, inputs, targets):
  error = model.predict(inputs) - targets
  return tf.reduce_mean(tf.square(error))

# A toy dataset of points around 3 * x + 2
NUM_EXAMPLES = 1000
training_inputs = tf.random_normal([NUM_EXAMPLES])
noise = tf.random_normal([NUM_EXAMPLES])
training_outputs = training_inputs * 3 + 2 + noise

# Define:
# 1. A model
# 2. Derivatives of a loss function with respect to model parameters
# 3. A strategy for updating the variables based on the derivatives
model = Model()
grad = tfe.implicit_gradients(loss)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)

# The training loop
print("Initial loss: %f" %
      loss(model, training_inputs, training_outputs).numpy())
for i in range(201):
  optimizer.apply_gradients(grad(model, training_inputs, training_outputs))
  if i % 20 == 0:
    print("Loss at step %d: %f" %
          (i, loss(model, training_inputs, training_outputs).numpy()))
print("Final loss: %f" % loss(model, training_inputs, training_outputs).numpy())
print("W, B = %s, %s" % (model.W.numpy(), model.B.numpy()))
```

Output:

```
Initial loss: 69.693184
Loss at step 0: 66.987854
Loss at step 20: 30.553387
Loss at step 40: 14.250237
Loss at step 60: 6.955020
Loss at step 80: 3.690550
Loss at step 100: 2.229739
Loss at step 120: 1.576032
Loss at step 140: 1.283496
Loss at step 160: 1.152584
Loss at step 180: 1.093999
Final loss: 1.067780
W, B = 3.0114281, 2.0865183
```

Using `implicit_gradients` avoided the need to provide all the trainable
parameters of the model as arguments to the `loss` function.

### Using Keras and the Layers API

[Keras](https://keras.io) is a popular API for defining model structures. The
[`tf.keras.layers`](https://www.tensorflow.org/api_docs/python/tf/contrib/keras/layers)
module provides a set of building blocks for models and is implemented using the
`tf.layer.Layer` subclasses in the
[`tf.layers`](https://www.tensorflow.org/api_docs/python/tf/layers) module. We
encourage the use of these same building blocks when using TensorFlow's eager
execution feature. For example, the very same linear regression model can be
built using `tf.layers.Dense`:

```python
class Model(object):
  def __init__(self):
    self.layer = tf.layers.Dense(1)

  def predict(self, inputs):
    return self.layer(inputs)
```

The `tf.layers` API makes it more convenient to define more sophisticated
models. For example, the following will train an MNIST model:

```python
class MNISTModel(object):
  def __init__(self, data_format):
    # 'channels_first' is typically faster on GPUs
    # while 'channels_last' is typically faster on CPUs.
    # See: https://www.tensorflow.org/performance/performance_guide#data_formats
    if data_format == 'channels_first':
      self._input_shape = [-1, 1, 28, 28]
    else:
      self._input_shape = [-1, 28, 28, 1]
    self.conv1 = tf.layers.Conv2D(32, 5,
                                  padding='same',
                                  activation=tf.nn.relu,
                                  data_format=data_format)
    self.max_pool2d = tf.layers.MaxPooling2D(
        (2, 2), (2, 2), padding='SAME', data_format=data_format)
    self.conv2 = tf.layers.Conv2D(64, 5,
                                  padding='same',
                                  activation=tf.nn.relu,
                                  data_format=data_format)
    self.dense1 = tf.layers.Dense(1024, activation=tf.nn.relu)
    self.dropout = tf.layers.Dropout(0.5)
    self.dense2 = tf.layers.Dense(10)

  def predict(self, inputs):
    x = tf.reshape(inputs, self._input_shape)
    x = self.max_pool2d(self.conv1(x))
    x = self.max_pool2d(self.conv2(x))
    x = tf.layers.flatten(x)
    x = self.dropout(self.dense1(x))
    return self.dense2(x)

def loss(model, inputs, targets):
  return tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(
          logits=model.predict(inputs), labels=targets))


# Load the training and validation data
from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets("./mnist_data", one_hot=True)

# Train
device = "gpu:0" if tfe.num_gpus() else "cpu:0"
model = MNISTModel('channels_first' if tfe.num_gpus() else 'channels_last')
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
grad = tfe.implicit_gradients(loss)
for i in range(20001):
  with tf.device(device):
    (inputs, targets) = data.train.next_batch(50)
    optimizer.apply_gradients(grad(model, inputs, targets))
    if i % 100 == 0:
      print("Step %d: Loss on training set : %f" %
            (i, loss(model, inputs, targets).numpy()))
print("Loss on test set: %f" % loss(model, data.test.images, data.test.labels).numpy())
```

### Checkpointing trained variables

TODO(ashankar):

### Summaries, metrics and TensorBoard

TODO(ashankar):

### Input Pipelines

The discussion above has been centered around the computation executed by your
model. The
[`tf.data`](https://www.tensorflow.org/versions/r1.4/api_docs/python/tf/data)
module provides APIs to build complex input pipelines from simple, reusable
pieces.

If you're familiar with constructing `tf.data.Dataset` objects when building
TensorFlow graphs, the same API calls are used when eager execution is enabled.
The process of iterating over elements of the dataset differs. When eager
execution is enabled, the discussion on iterator creation using
`make_one_shot_iterator()` and `get_next()` in the [Programmer's
Guide](https://www.tensorflow.org/versions/r1.4/programmers_guide/datasets) is
not relevant and instead a more Pythonic `Iterator` class is available.

For example:

```python
# Create a source Dataset from in-memory numpy arrays.
# For reading from files on disk, you may want to use other Dataset classes
# like the TextLineDataset or the TFRecordDataset.
dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3, 4, 5, 6])

# Apply transformations, shuffling, batching etc.
dataset = dataset.map(tf.square).shuffle(2).batch(2)

# Use tfe.Iterator to iterate over the dataset.
for x in tfe.Iterator(dataset):
  print(x)
```

Output:

```
tf.Tensor([4 9], shape=(2,), dtype=int32)
tf.Tensor([16 25], shape=(2,), dtype=int32)
tf.Tensor([36  1], shape=(2,), dtype=int32)
```

## Interoperating with Graphs

Eager execution improves the process of model development in Python; however,
because it is in its earliest stages, it does not yet support some features
available to [TensorFlow
graphs](https://www.tensorflow.org/get_started/get_started#the_computational_graph)
that are desirable when deploying models in production. In particular, eager
execution does not yet support distributed training, exporting models (to other
[programming languages](https://www.tensorflow.org/api_docs/), [TensorFlow
serving](https://www.tensorflow.org/serving/), and mobile applications), and
various memory and computation optimizations that are applied to TensorFlow's
dataflow graphs.

That said, the APIs used to build modes are exactly the same whether executing
eagerly or constructing graphs. This means that you can iteratively develop your
model with eager execution enabled and later, if needed, use the same code to
reap the benefits of representing models as computational graphs.

For example,
[`mnist.py`](https://www.tensorflow.org/code/tensorflow/contrib/eager/python/examples/mnist/mnist.py)
defines a model that is eagerly executed. That same code is used to construct
and execute a graph in
[`mnist_graph_test.py`](https://www.tensorflow.org/code/tensorflow/contrib/eager/python/examples/mnist/mnist_graph_test.py).

Other models in the [examples
directory](https://www.tensorflow.org/code/tensorflow/contrib/eager/python/examples/)
demonstrate this as well.

Some differences worth noting:

TODO(ashankar): Things to talk about:

-   `tf.gradients` vs. the `tfe` gradient functions
-   Variable naming
-   Use of functional layers API vs. object oriented one
-   tfe.Variable (ResourceVariables) vs. tf.Variable
-   Sessions and placeholders
-   `Tensor` properties like op, name, inputs etc.

TODO(ashankar): Update links (some of the links will currently only be valid
after the website is updated for the 1.4 release)
