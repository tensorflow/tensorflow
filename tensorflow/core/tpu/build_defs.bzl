"""A helper to include the TF C API implementations or only headers depending on whether building Tensorflow or LibTPU."""

load("//tensorflow:tensorflow.bzl", "if_libtpu")

# Returns the appropriate TF_Status C API def based on whether building LibTPU or Tensorflow.
def if_libtpu_tf_status():
    return if_libtpu(
        if_true = ["//tensorflow/c:tf_status_headers"],
        if_false = ["//tensorflow/c:tf_status"],
    )

# Returns the appropriate TF_Tensor C API def based on whether building LibTPU or Tensorflow.
def if_libtpu_tf_tensor():
    return if_libtpu(
        if_true = ["//tensorflow/c:tf_tensor_hdrs"],
        if_false = ["//tensorflow/c:tf_tensor"],
    )

# Returns the C API headers or full implementation based on whether building LibTPU or Tensorflow.
def if_libtpu_tf_c_api():
    return if_libtpu(
        if_true = ["//tensorflow/c:c_api_headers"],
        if_false = ["//tensorflow/c:c_api"],
    )
