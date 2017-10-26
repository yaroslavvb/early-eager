Image classification using the ResNet50 model described in
[Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385).

Contents:

- `resnet50.py`: Model definition
- `resnet50_test.py`: Sanity unittests and benchmarks for using the model with
  eager execution enabled.
- `resnet50_graph_test.py`: Sanity unittests and benchmarks when using the same
  model code to construct a TensorFlow graph.

# Benchmarks

Using a synthetic data.

```
# Using eager execution
bazel run -c opt --config=cuda :resnet50_test -- --benchmarks=.

# Using graph execution
bazel run -c opt --config=cuda :resnet50_graph_test -- --benchmarks=.
```

(Or remove the `--config=cuda` flag for running on CPU instead of GPU).
