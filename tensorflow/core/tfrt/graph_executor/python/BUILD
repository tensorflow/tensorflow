load("//tensorflow:tensorflow.default.bzl", "pybind_extension")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
)

licenses(["notice"])

pybind_extension(
    name = "graph_execution_options",
    srcs = ["graph_execution_options.cc"],
    deps = [
        "//tensorflow/core/tfrt/graph_executor",
        "//tensorflow/core/tfrt/graph_executor:graph_execution_options",
        "//tensorflow/core/tfrt/saved_model",
    ],
)
