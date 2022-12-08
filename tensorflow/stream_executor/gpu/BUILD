# Description:
#   GPU-platform specific StreamExecutor support code.

load(
    "//tensorflow/compiler/xla/stream_executor:build_defs.bzl",
    "if_gpu_is_configured",
)
load("//tensorflow:tensorflow.bzl", "tf_copts")
load(
    "//tensorflow/core/platform:rules_cc.bzl",
    "cc_library",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//tensorflow/stream_executor:__subpackages__"],
    licenses = ["notice"],
)

cc_library(
    name = "gpu_activation_header",
    hdrs = ["gpu_activation.h"],
    deps = ["//tensorflow/compiler/xla/stream_executor/gpu:gpu_activation_header"],
)

cc_library(
    name = "gpu_activation",
    hdrs = if_gpu_is_configured(["gpu_activation.h"]),
    deps = if_gpu_is_configured([
        "//tensorflow/compiler/xla/stream_executor/gpu:gpu_activation",
    ]),
)

cc_library(
    name = "gpu_diagnostics_header",
    hdrs = if_gpu_is_configured(["gpu_diagnostics.h"]),
    deps = [
        "//tensorflow/compiler/xla/stream_executor/gpu:gpu_diagnostics_header",
    ],
)

cc_library(
    name = "gpu_driver_header",
    hdrs = if_gpu_is_configured(["gpu_driver.h"]),
    deps = [
        "//tensorflow/compiler/xla/stream_executor/gpu:gpu_driver_header",
    ],
)

cc_library(
    name = "gpu_event_header",
    hdrs = if_gpu_is_configured(["gpu_event.h"]),
    deps = if_gpu_is_configured([
        "//tensorflow/compiler/xla/stream_executor/gpu:gpu_event_header",
    ]),
)

cc_library(
    name = "gpu_event",
    hdrs = if_gpu_is_configured(["gpu_event.h"]),
    deps = [
        "//tensorflow/compiler/xla/stream_executor/gpu:gpu_event",
    ],
)

cc_library(
    name = "gpu_executor_header",
    hdrs = if_gpu_is_configured(["gpu_executor.h"]),
    deps = [
        "//tensorflow/compiler/xla/stream_executor/gpu:gpu_executor_header",
    ],
)

cc_library(
    name = "gpu_helpers_header",
    hdrs = if_gpu_is_configured(["gpu_helpers.h"]),
    deps = ["//tensorflow/compiler/xla/stream_executor/gpu:gpu_helpers_header"],
)

cc_library(
    name = "gpu_kernel_header",
    hdrs = if_gpu_is_configured(["gpu_kernel.h"]),
    deps = ["//tensorflow/compiler/xla/stream_executor/gpu:gpu_kernel_header"],
)

cc_library(
    name = "gpu_rng_header",
    hdrs = if_gpu_is_configured(["gpu_rng.h"]),
    deps = ["//tensorflow/compiler/xla/stream_executor/gpu:gpu_rng_header"],
)

cc_library(
    name = "gpu_stream_header",
    hdrs = if_gpu_is_configured(["gpu_stream.h"]),
    deps = ["//tensorflow/compiler/xla/stream_executor/gpu:gpu_stream_header"],
)

cc_library(
    name = "gpu_stream",
    hdrs = if_gpu_is_configured(["gpu_stream.h"]),
    deps = ["//tensorflow/compiler/xla/stream_executor/gpu:gpu_stream"],
)

cc_library(
    name = "gpu_timer_header",
    hdrs = if_gpu_is_configured(["gpu_timer.h"]),
    deps = ["//tensorflow/compiler/xla/stream_executor/gpu:gpu_timer_header"],
)

cc_library(
    name = "gpu_timer",
    hdrs = if_gpu_is_configured(["gpu_timer.h"]),
    deps = ["//tensorflow/compiler/xla/stream_executor/gpu:gpu_timer"],
)

cc_library(
    name = "gpu_types_header",
    hdrs = if_gpu_is_configured(["gpu_types.h"]),
    deps = [
        "//tensorflow/compiler/xla/stream_executor/gpu:gpu_types_header",
    ],
)

cc_library(
    name = "gpu_asm_opts",
    hdrs = ["gpu_asm_opts.h"],
    deps = ["//tensorflow/compiler/xla/stream_executor/gpu:gpu_asm_opts"],
)

cc_library(
    name = "asm_compiler",
    hdrs = if_gpu_is_configured(["asm_compiler.h"]),
    copts = tf_copts(),
    deps = if_gpu_is_configured([
        "//tensorflow/compiler/xla/stream_executor/gpu:asm_compiler",
    ]),
)

cc_library(
    name = "redzone_allocator",
    hdrs = if_gpu_is_configured(["redzone_allocator.h"]),
    copts = tf_copts(),
    deps = if_gpu_is_configured([
        "//tensorflow/compiler/xla/stream_executor/gpu:redzone_allocator",
    ]),
)
