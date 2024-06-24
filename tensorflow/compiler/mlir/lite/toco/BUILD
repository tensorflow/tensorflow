load(
    "//tensorflow/core/platform:build_config.bzl",
    "tf_proto_library",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# LINT.IfChange

# OSS only: This target is header-only. Link `types_proto_cc_impl` only to
# `libtensorflow_framework.so` via `lib_internal_impl`. Do NOT link `types_flags_proto_cc_impl`
# directly unless the target does not link `libtensorflow_framework.so`.
tf_proto_library(
    name = "types_proto",
    srcs = ["types.proto"],
    cc_api_version = 2,
    make_default_target_header_only = True,
    visibility = ["//visibility:public"],
)

# OSS only: This target is header-only. Link `toco_flags_proto_cc_impl` only to
# `libtensorflow_framework.so` via `lib_internal_impl`. Do NOT link `toco_flags_proto_cc_impl`
# directly unless the target does not link `libtensorflow_framework.so`.
tf_proto_library(
    name = "toco_flags_proto",
    srcs = ["toco_flags.proto"],
    cc_api_version = 2,
    make_default_target_header_only = True,
    protodeps = [
        "//tensorflow/compiler/mlir/quantization/stablehlo:quantization_options_proto",
        "//tensorflow/compiler/mlir/quantization/stablehlo:quantization_config_proto",
        "//tensorflow/compiler/mlir/lite/debug:debug_options_proto",
        ":types_proto",
    ],
    visibility = ["//visibility:public"],
)

# OSS only: This target is header-only. Link `model_flags_proto_cc_impl` only to
# `libtensorflow_framework.so` via `lib_internal_impl`. Do NOT link `model_flags_proto_cc_impl`
# directly unless the target does not link `libtensorflow_framework.so`.
tf_proto_library(
    name = "model_flags_proto",
    srcs = ["model_flags.proto"],
    cc_api_version = 2,
    make_default_target_header_only = True,
    protodeps = [":types_proto"],
    visibility = ["//visibility:public"],
)

# copybara:uncomment_begin(google-only)
# py_proto_library(
#     name = "model_flags_proto_py",
#     api_version = 2,
#     visibility = ["//visibility:public"],
#     deps = [":model_flags_proto"],
# )
#
# py_proto_library(
#     name = "toco_flags_proto_py",
#     api_version = 2,
#     visibility = ["//visibility:public"],
#     deps = [":toco_flags_proto"],
# )
#
# py_proto_library(
#     name = "types_proto_py",
#     api_version = 2,
#     visibility = ["//visibility:public"],
#     deps = [":toco_flags_proto"],
# )
# copybara:uncomment_end

# LINT.ThenChange(//tensorflow/lite/toco/BUILD)
