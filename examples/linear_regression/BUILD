licenses(["notice"])  # Apache 2.0

load("//tensorflow:tensorflow.bzl", "cuda_py_test")

py_binary(
    name = "linear_regression",
    srcs = ["linear_regression.py"],
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/contrib/eager/python:tfe",
        "@six_archive//:six",
    ],
)

cuda_py_test(
    name = "linear_regression_test",
    size = "small",
    srcs = ["linear_regression_test.py"],
    additional_deps = [
        ":linear_regression",
        "//tensorflow:tensorflow_py:tensorflow",
    ],
)
