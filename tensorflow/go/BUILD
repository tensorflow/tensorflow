# Description:
# Go API for TensorFlow.

load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")
load("//tensorflow:tensorflow.bzl", "VERSION_MAJOR", "filegroup")

package(default_visibility = ["//visibility:private"])

licenses(["notice"])  # Apache 2.0

go_library(
    name = "go",
    srcs = [
        "android.go",
        "attrs.go",
        "context.go",
        "doc.go",
        "graph.go",
        "lib.go",
        "operation.go",
        "saved_model.go",
        "session.go",
        "shape.go",
        "signature.go",
        "status.go",
        "tensor.go",
        "tensor_handle.go",
        "version.go",
    ],
    cdeps = [":libtensorflow"],  # keep
    cgo = True,
    clinkopts = select({
        "@io_bazel_rules_go//go/platform:android": [
            "-landroid -llog -lm -lz -ldl",
        ],
        "//conditions:default": [],
    }),  # keep
    copts = [],  # keep
    importpath = "github.com/tensorflow/tensorflow/tensorflow/go",
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/go/core/protobuf:for_core_protos_go_proto",
        "@org_golang_google_protobuf//proto:go_default_library",
    ],
)

go_test(
    name = "go_test",
    size = "small",
    srcs = [
        "attrs_test.go",
        "context_test.go",
        "example_inception_inference_test.go",
        "graph_test.go",
        "operation_test.go",
        "saved_model_test.go",
        "session_test.go",
        "shape_test.go",
        "signature_test.go",
        "tensor_handle_test.go",
        "tensor_test.go",
        "util_test.go",
    ],
    data = glob(["testdata/**/*"]),
    embed = [":go"],
    deps = [
        "//tensorflow/go/core/framework:tensor_shape_go_proto",
        "//tensorflow/go/core/framework:types_go_proto",
        "//tensorflow/go/core/protobuf:for_core_protos_go_proto",
        "//tensorflow/go/op",
    ],
)

cc_library(
    name = "libtensorflow",
    hdrs = [
        "//tensorflow/c:headers",
        "//tensorflow/c/eager:headers",
    ],
    visibility = ["//tensorflow/go:__subpackages__"],
    deps = [
        ":libtensorflow_import_lib",
        "//tensorflow:libtensorflow_framework_import_lib",
    ],
)

cc_import(
    name = "libtensorflow_import_lib",
    shared_library = select({
        "//tensorflow:macos": "//tensorflow:libtensorflow.%s.dylib" % VERSION_MAJOR,
        "//conditions:default": "//tensorflow:libtensorflow.so.%s" % VERSION_MAJOR,
    }),
    visibility = ["//tensorflow/go:__subpackages__"],
)

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
    visibility = ["//visibility:public"],
)
