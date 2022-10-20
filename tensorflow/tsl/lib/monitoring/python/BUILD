# Python bindings for tsl/lib/monitoring.
load("//tensorflow:tensorflow.default.bzl", "pybind_extension")

package(
    licenses = ["notice"],
)

pybind_extension(
    name = "monitoring",
    srcs = [
        "monitoring.cc",
    ],
    pytype_srcs = ["__init__.pyi"],
    visibility = [
        "//third_party/py/jax:__subpackages__",
        "//third_party/py/paxml:__subpackages__",
    ],
    deps = [
        "//tensorflow/tsl/lib/monitoring:sampler",
        "//tensorflow/tsl/platform:env_impl",
        "//tensorflow/tsl/protobuf:histogram_proto_cc_impl",
        "@com_google_absl//absl/memory",
        "@com_google_protobuf//:protobuf",
        "@pybind11",
    ],
)

py_test(
    name = "monitoring_test",
    srcs = ["monitoring_test.py"],
    deps = [
        "//tensorflow/tsl/lib/monitoring/python:monitoring",
        "@absl_py//absl/testing:absltest",
    ],
)
