load("//tensorflow/tsl/platform:rules_cc.bzl", "cc_library")
load("//tensorflow/tsl/platform:build_config.bzl", "tf_platform_deps")
load("//tensorflow/compiler/xla/mlir/backends/gpu2:build_config.bzl", "if_gpu2", "if_not_gpu2")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [":friends"],
    licenses = ["notice"],
)

package_group(
    name = "friends",
    includes = ["//tensorflow/compiler/xla:friends"],
)

# copybara:uncomment_begin(not supported in OSS build)
#
# cc_library(
#     name = "compiler",
#     srcs = if_gpu2(["compiler.cc"]),
#     hdrs = if_gpu2(["compiler.h"]),
#     # TODO(ezhulenev): Override cc_library()'s default compatibility because IREE targets are not
#     # compatible with `non_prod` constraint.
#     compatible_with = [],
#     deps = [
#         "@com_google_absl//absl/base",
#         "@llvm-project//llvm:Support",
#         "@llvm-project//mlir:IR",
#         "@llvm-project//mlir:Support",
#         "//tensorflow/compiler/xla:status",
#         "//tensorflow/compiler/xla:statusor",
#         "//tensorflow/compiler/xla:util",
#         "//tensorflow/compiler/xla:xla_proto_cc",
#         "//tensorflow/compiler/xla/service:dump",
#         "//tensorflow/compiler/xla/service/llvm_ir:llvm_util",
#         "//tensorflow/tsl/platform",
#     ] + tf_platform_deps(
#         "compiler",
#         platform_dir = "//tensorflow/compiler/xla/service/gpu/runtime2/",
#     ) + if_gpu2([
#         "//third_party/iree/compiler/bindings/c:headers",
#         "//third_party/iree/compiler/bindings/c:loader",
#         "//third_party/iree/llvm-external-projects/iree-dialects/google_internal:IREEInputDialect",
#     ]),
# )
#
# cc_library(
#     name = "executable",
#     srcs = if_gpu2(["executable.cc"]),
#     hdrs = ["executable.h"],
#     compatible_with = [],
#     defines = if_not_gpu2(["XLA_DISABLE_GPU2_RUNTIME=1"]),
#     deps = [
#         ":compiler",
#         ":module",
#         ":vm",
#         "@com_google_absl//absl/cleanup",
#         "@com_google_absl//absl/log",
#         "@com_google_absl//absl/log:check",
#         "@com_google_absl//absl/strings",
#         "@llvm-project//mlir:IR",
#         "//tensorflow/compiler/xla:status",
#         "//tensorflow/compiler/xla:statusor",
#         "//tensorflow/compiler/xla:util",
#         "//tensorflow/compiler/xla:xla_proto_cc",
#         "//tensorflow/compiler/xla/service:buffer_assignment",
#         "//tensorflow/compiler/xla/service:dump",
#         "//tensorflow/compiler/xla/service:executable",
#         "//tensorflow/compiler/xla/service/gpu:buffer_allocations",
#         "//tensorflow/tsl/platform:path",
#         "//tensorflow/tsl/platform:platform_port",
#     ] + if_gpu2([
#         "//third_party/iree/runtime/src/iree/base",
#         "//third_party/iree/runtime/src/iree/hal",
#         "//third_party/iree/runtime/src/iree/hal/drivers/cuda",
#         "//third_party/iree/runtime/src/iree/modules/hal",
#         "//third_party/iree/runtime/src/iree/modules/hal:types",
#         "//third_party/iree/runtime/src/iree/tooling:numpy_io",
#         "//third_party/iree/runtime/src/iree/vm",
#         "//third_party/iree/runtime/src/iree/vm/bytecode:module",
#     ]),
# )
#
# cc_library(
#     name = "gemm",
#     srcs = if_gpu2(["gemm.cc"]),
#     hdrs = if_gpu2(["gemm.h"]),
#     compatible_with = [],
#     deps = [
#         ":hal",
#         ":vm",
#         "@com_google_absl//absl/container:inlined_vector",
#         "@com_google_absl//absl/log",
#         "//tensorflow/compiler/xla:status",
#         "//tensorflow/compiler/xla:statusor",
#         "//tensorflow/compiler/xla:xla_data_proto_cc",
#         "//tensorflow/compiler/xla/service:executable",
#         "//tensorflow/compiler/xla/service/gpu:matmul_utils",
#         "//tensorflow/tsl/profiler/lib:scoped_annotation",
#     ] + if_gpu2([
#         "//third_party/iree/runtime/src/iree/hal",
#         "//third_party/iree/runtime/src/iree/vm",
#     ]),
# )
#
# cc_library(
#     name = "graph",
#     srcs = if_gpu2(["graph.cc"]),
#     hdrs = if_gpu2(["graph.h"]),
#     compatible_with = [],
#     deps = [
#         ":hal",
#         ":kernel",
#         ":vm",
#         "@com_google_absl//absl/container:inlined_vector",
#         "@com_google_absl//absl/log",
#         "@com_google_absl//absl/synchronization",
#         "@com_google_absl//absl/types:span",
#         "//tensorflow/compiler/xla:status",
#         "//tensorflow/compiler/xla:statusor",
#         "//tensorflow/compiler/xla:xla_data_proto_cc",
#         "//tensorflow/compiler/xla/service:executable",
#         "//tensorflow/compiler/xla/service/gpu:launch_dimensions",
#         "//tensorflow/compiler/xla/service/gpu:stream_executor_util",
#         "//tensorflow/compiler/xla/stream_executor:kernel",
#         "//tensorflow/compiler/xla/stream_executor/gpu:gpu_executor_header",
#         "//tensorflow/compiler/xla/stream_executor/gpu:gpu_graph",
#         "//tensorflow/compiler/xla/stream_executor/gpu:gpu_types_header",
#     ] + if_gpu2([
#         "//third_party/iree/runtime/src/iree/hal",
#         "//third_party/iree/runtime/src/iree/vm",
#     ]),
# )
#
# cc_library(
#     name = "kernel",
#     srcs = if_gpu2(["kernel.cc"]),
#     hdrs = if_gpu2(["kernel.h"]),
#     compatible_with = [],
#     deps = [
#         ":hal",
#         ":vm",
#         "@com_google_absl//absl/container:flat_hash_map",
#         "@com_google_absl//absl/container:inlined_vector",
#         "@com_google_absl//absl/log",
#         "@com_google_absl//absl/synchronization",
#         "//tensorflow/compiler/xla:status",
#         "//tensorflow/compiler/xla:statusor",
#         "//tensorflow/compiler/xla/service:executable",
#         "//tensorflow/compiler/xla/service/gpu:launch_dimensions",
#         "//tensorflow/compiler/xla/service/gpu:stream_executor_util",
#         "//tensorflow/compiler/xla/stream_executor:device_memory",
#         "//tensorflow/compiler/xla/stream_executor:kernel",
#         "//tensorflow/tsl/profiler/lib:scoped_annotation",
#     ] + if_gpu2([
#         "//third_party/iree/runtime/src/iree/hal",
#         "//third_party/iree/runtime/src/iree/vm",
#     ]),
# )
#
# cc_library(
#     name = "memcpy",
#     srcs = if_gpu2(["memcpy.cc"]),
#     hdrs = if_gpu2(["memcpy.h"]),
#     compatible_with = [],
#     deps = [
#         ":hal",
#         ":vm",
#         "@com_google_absl//absl/container:inlined_vector",
#         "@com_google_absl//absl/log",
#         "//tensorflow/compiler/xla:status",
#         "//tensorflow/compiler/xla:statusor",
#         "//tensorflow/compiler/xla/service:executable",
#         "//tensorflow/compiler/xla/service/gpu:launch_dimensions",
#         "//tensorflow/compiler/xla/service/gpu:stream_executor_util",
#         "//tensorflow/compiler/xla/stream_executor:device_memory",
#         "//tensorflow/tsl/profiler/lib:scoped_annotation",
#     ] + if_gpu2([
#         "//third_party/iree/runtime/src/iree/hal",
#         "//third_party/iree/runtime/src/iree/vm",
#     ]),
# )
#
# cc_library(
#     name = "module",
#     srcs = if_gpu2(["module.cc"]),
#     hdrs = if_gpu2(["module.h"]),
#     compatible_with = [],
#     deps = [
#         ":gemm",
#         ":graph",
#         ":hal",
#         ":kernel",
#         ":memcpy",
#         ":vm",
#         "@com_google_absl//absl/log",
#     ] + if_gpu2([
#         "//third_party/iree/runtime/src/iree/base",
#         "//third_party/iree/runtime/src/iree/hal",
#         "//third_party/iree/runtime/src/iree/modules/hal:types",
#         "//third_party/iree/runtime/src/iree/vm",
#         "//third_party/iree/runtime/src/iree/vm:cc",
#     ]),
# )
#
# cc_library(
#     name = "hal",
#     srcs = if_gpu2(["hal.cc"]),
#     hdrs = if_gpu2(["hal.h"]),
#     compatible_with = [],
#     deps = [
#         "@com_google_absl//absl/container:inlined_vector",
#         "@com_google_absl//absl/types:span",
#         "//tensorflow/compiler/xla:shape_util",
#         "//tensorflow/compiler/xla/stream_executor:device_memory",
#     ] + if_gpu2([
#         "//third_party/iree/runtime/src/iree/hal",
#     ]),
# )
#
# cc_library(
#     name = "vm",
#     srcs = if_gpu2(["vm.cc"]),
#     hdrs = if_gpu2(["vm.h"]),
#     compatible_with = [],
#     deps = [
#         "@com_google_absl//absl/container:inlined_vector",
#         "@com_google_absl//absl/strings:str_format",
#         "//tensorflow/compiler/xla:status",
#         "//tensorflow/compiler/xla:statusor",
#     ] + if_gpu2([
#         "//third_party/iree/runtime/src/iree/vm",
#         "//third_party/iree/runtime/src/iree/vm:cc",
#         "//third_party/iree/runtime/src/iree/hal",
#         "//third_party/iree/runtime/src/iree/modules/hal:types",
#     ]),
# )
#
# copybara:uncomment_end_and_comment_begin
cc_library(
    name = "executable",
    hdrs = ["executable.h"],
    defines = ["XLA_DISABLE_GPU2_RUNTIME=1"],
    deps = [
        "//tensorflow/compiler/xla:status",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla/service:executable",
        "//tensorflow/compiler/xla/service/gpu:buffer_allocations",
        "@llvm-project//mlir:IR",
    ],
)
# copybara:comment_end
