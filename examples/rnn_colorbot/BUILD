licenses(["notice"])  # Apache 2.0

load("//tensorflow:tensorflow.bzl", "cuda_py_test")

py_binary(
    name = "rnn_colorbot",
    srcs = ["rnn_colorbot.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/contrib/eager/python:tfe",
        "//third_party/py/matplotlib",
        "@six_archive//:six",
    ],
)

cuda_py_test(
    name = "rnn_colorbot_test",
    srcs = ["rnn_colorbot_test.py"],
    additional_deps = [
        ":rnn_colorbot",
        "//tensorflow/contrib/eager/python:tfe",
        "//tensorflow:tensorflow_py:tensorflow",
    ],
)
