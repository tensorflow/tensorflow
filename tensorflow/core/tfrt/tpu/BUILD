load("//tensorflow:tensorflow.bzl", "if_google")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

cc_library(
    name = "tpu_resources",
    hdrs = ["tpu_resources.h"],
    visibility = [
        "//tensorflow/core/tfrt/graph_executor:__pkg__",
        "//tensorflow/core/tfrt/saved_model:__pkg__",
    ],
    deps = if_google([
        "@tf_runtime//:hostcontext",
        # TODO(b/178227859): Remove when we have a pluggable device API.
        "//third_party/tf_runtime_google:tpu_model_resource",
    ]),
)
