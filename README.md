# TensorFlow Eager Execution

> *WARNING*: This is a preview/pre-alpha version. The API and performance
> characteristics are subject to change.


Eager execution is an experimental interface to TensorFlow that provides an
imperative programming style (à la [NumPy](http://www.numpy.org)). When you
enable eager execution, TensorFlow operations execute immediately; you do not
execute a pre-constructed graph with
[`Session.run()`](https://www.tensorflow.org/api_docs/python/tf/Session).

For example, consider a simple computation in TensorFlow:

```python
x = tf.placeholder(tf.float32, shape=[1, 1])
m = tf.matmul(x, x)

with tf.Session() as sess:
  print(sess.run(m, feed_dict={x: [[2.]]}))

# Will print [[4.]]
```

Eager execution makes this much simpler:

```python
x = [[2.]]
m = tf.matmul(x, x)

print(m)
```

## Installation

Since eager execution is not yet part of a TensorFlow release, using it requires
either [building from source](https://www.tensorflow.org/install/install_sources)
or the latest nightly builds. The nightly builds are available as:

- [`pip` packages](https://github.com/tensorflow/tensorflow/blob/master/README.md#installation) and

- [docker](https://hub.docker.com/r/tensorflow/tensorflow/) images.

For example, to run the latest nightly docker image:

```sh
# If you have a GPU, use https://github.com/NVIDIA/nvidia-docker
nvidia-docker pull tensorflow/tensorflow:nightly-gpu
nvidia-docker run -it -p 8888:8888 tensorflow/tensorflow:nightly-gpu

# If you do not have a GPU, use the CPU-only image
docker pull tensorflow/tensorflow:nightly
docker run -it -p 8888:8888 tensorflow/tensorflow:nightly
```

And then visit http://localhost:8888 in your browser for a Jupyter notebook
environment. Try out the notebooks below.

## Documentation

For an introduction to TensorFlow eager execution, see the Jupyter notebooks:

- [Basic Usage](examples/notebooks/1_basics.ipynb)
- [Gradients](examples/notebooks/2_gradients.ipynb)
