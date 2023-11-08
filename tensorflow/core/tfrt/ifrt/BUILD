package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        # copybara:uncomment "//learning/brain/experimental/tfrt:__subpackages__",
        # copybara:uncomment "//learning/brain/tfrt:__subpackages__",
        # copybara:uncomment "//learning/infra/mira/distributed:__subpackages__",
        "//tensorflow/compiler/mlir/tfrt:__subpackages__",
        "//tensorflow/compiler/mlir/tfrt/transforms/mlrt:__subpackages__",
        "//tensorflow/compiler/mlir/tfrt/translate/mlrt:__subpackages__",
        "//tensorflow/core/tfrt:__subpackages__",
        "//tensorflow/core/tfrt/mlrt:__subpackages__",
    ],
)

cc_library(
    name = "ifrt_serving_executable",
    srcs = ["ifrt_serving_executable.cc"],
    hdrs = ["ifrt_serving_executable.h"],
    deps = [
        "//tensorflow/compiler/mlir/tfrt/transforms/ifrt:tf2hlo",
        "//tensorflow/compiler/tf2xla:common",
        "//tensorflow/compiler/tf2xla:xla_helpers",
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@llvm-project//mlir:IR",
        "@local_tsl//tsl/concurrency:ref_count",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:statusor",
        "@local_xla//xla/python/ifrt",
        "@local_xla//xla/python/pjrt_ifrt",
        "@local_xla//xla/python/pjrt_ifrt:xla_ifrt",
        "@tf_runtime//:ref_count",
    ],
)

cc_library(
    name = "ifrt_executable_registry",
    srcs = ["ifrt_executable_registry.cc"],
    hdrs = ["ifrt_executable_registry.h"],
    deps = [
        ":ifrt_serving_executable",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_library(
    name = "ifrt_model_context",
    hdrs = ["ifrt_model_context.h"],
    deps = [
        ":ifrt_executable_registry",
        "//tensorflow/compiler/tf2xla:xla_helpers",
        "@com_google_absl//absl/strings",
        "@local_xla//xla/python/ifrt",
    ],
)
