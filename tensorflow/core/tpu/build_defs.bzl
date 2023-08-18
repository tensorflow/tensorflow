"""A helper to include the TF_Status C API implementation or only headers depending on whether building Tensorflow or LibTPU."""

load("//tensorflow:tensorflow.bzl", "if_libtpu")

# Returns the appropriate tf_status C API def based on whether building LibTPU or Tensorflow
def if_libtpu_tf_status():
    return if_libtpu(
        if_true = ["//tensorflow/c:tf_status_headers"],
        if_false = ["//tensorflow/c:tf_status"],
    )
