load("@local_tsl//tsl/platform:rules_cc.bzl", "cc_library")
load("@local_tsl//tsl/platform:build_config.bzl", "tf_platform_deps")
load("//xla/mlir/backends/gpu2:build_config.bzl", "if_gpu2", "if_not_gpu2")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

package_group(
    name = "friends",
    includes = ["//xla:friends"],
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
#         "//xla:status",
#         "//xla:statusor",
#         "//xla:util",
#         "//xla:xla_proto_cc",
#         "//xla/service:dump",
#         "//xla/service/llvm_ir:llvm_util",
#         "@local_tsl//tsl/platform",
#         "@llvm-project//llvm:Support",
#         "@llvm-project//mlir:IR",
#         "@llvm-project//mlir:Support",
#     ] + tf_platform_deps(
#         "compiler",
#         platform_dir = "//xla/service/gpu/runtime2/",
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
#         "//xla:status",
#         "//xla:statusor",
#         "//xla:util",
#         "//xla:xla_proto_cc",
#         "//xla/service:buffer_assignment",
#         "//xla/service:dump",
#         "//xla/service:executable",
#         "//xla/service/gpu:buffer_allocations",
#         "@local_tsl//tsl/platform:path",
#         "@local_tsl//tsl/platform:platform_port",
#         "@llvm-project//mlir:IR",
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
#         "//xla:status",
#         "//xla:statusor",
#         "//xla:xla_data_proto_cc",
#         "//xla/service:executable",
#         "//xla/service/gpu:matmul_utils",
#         "@local_tsl//tsl/profiler/lib:scoped_annotation",
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
#         "//xla:status",
#         "//xla:statusor",
#         "//xla/service:executable",
#         "//xla/service/gpu:launch_dimensions",
#         "//xla/service/gpu:stream_executor_util",
#         "//xla/stream_executor:device_memory",
#         "//xla/stream_executor:kernel",
#         "@local_tsl//tsl/profiler/lib:scoped_annotation",
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
#         "//xla:status",
#         "//xla:statusor",
#         "//xla/service:executable",
#         "//xla/service/gpu:launch_dimensions",
#         "//xla/service/gpu:stream_executor_util",
#         "//xla/stream_executor:device_memory",
#         "@local_tsl//tsl/profiler/lib:scoped_annotation",
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
#         "//xla:shape_util",
#         "//xla/stream_executor:device_memory",
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
#         "//xla:status",
#         "//xla:statusor",
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
    visibility = ["//visibility:public"],
    deps = [
        "//xla:status",
        "//xla:statusor",
        "//xla/service:executable",
        "//xla/service/gpu:buffer_allocations",
        "@llvm-project//mlir:IR",
    ],
)
# copybara:comment_end
