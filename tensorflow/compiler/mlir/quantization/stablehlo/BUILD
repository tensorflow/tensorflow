load("//tensorflow/core/platform:build_config.bzl", "tf_proto_library")
load("//tensorflow/compiler/mlir/quantization/stablehlo:internal_visibility_allowlist.bzl", "internal_visibility_allowlist")

package_group(
    name = "internal_visibility_allowlist_package",
    packages = [
        "//tensorflow/compiler/mlir/lite/...",
        "//tensorflow/compiler/mlir/quantization/...",
        "//third_party/cloud_tpu/inference_converter/...",  # TPU Inference Converter V1
    ] + internal_visibility_allowlist(),
)

tf_proto_library(
    name = "quantization_options_proto",
    srcs = ["quantization_options.proto"],
    cc_api_version = 2,
    make_default_target_header_only = True,
    visibility = [":internal_visibility_allowlist_package"],
)

# copybara:uncomment_begin(google-only)
# py_proto_library(
#     name = "quantization_options_py_pb2",
#     api_version = 2,
#     visibility = [":internal_visibility_allowlist_package"],
#     deps = [":quantization_options_proto"],
# )
# copybara:uncomment_end
