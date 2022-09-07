load("//tensorflow:tensorflow.bzl", "tf_cc_test")
load("//tensorflow:tensorflow.bzl", "get_compatible_with_cloud")
load("//tensorflow/core/platform:build_config.bzl", "tf_platform_deps")
load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")

package(
    default_visibility = [
        "//tensorflow:internal",
        "@tf_runtime//:friends",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "arguments",
    srcs = ["arguments.cc"],
    hdrs = ["arguments.h"],
    compatible_with = get_compatible_with_cloud(),
    deps = [
        ":errors",
        ":types",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla/mlir/transforms/runtime:type_converter",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
    ],
)

tf_cc_test(
    name = "arguments_test",
    srcs = ["arguments_test.cc"],
    deps = [
        ":arguments",
        ":types",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "async_runtime",
    srcs = ["async_runtime.cc"],
    hdrs = ["async_runtime.h"],
    compatible_with = get_compatible_with_cloud(),
    deps = [
        "//tensorflow/tsl/platform:platform_port",
        "//third_party/eigen3",
        "@com_google_absl//absl/base:dynamic_annotations",
        "@com_google_absl//absl/status",
        "@llvm-project//llvm:Support",
        "@tf_runtime//:hostcontext",
    ],
)

cc_library(
    name = "async_values_cache",
    hdrs = ["async_values_cache.h"],
    compatible_with = get_compatible_with_cloud(),
    deps = [
        "//tensorflow/tsl/platform:platform",
    ] + tf_platform_deps(
        "async_values_cache",
        platform_dir = "//tensorflow/compiler/xla/runtime/",
    ),
)

cc_library(
    name = "constraints",
    srcs = ["constraints.cc"],
    hdrs = ["constraints.h"],
    compatible_with = get_compatible_with_cloud(),
    deps = [
        ":errors",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "custom_call",
    srcs = ["custom_call.cc"],
    hdrs = ["custom_call.h"],
    compatible_with = get_compatible_with_cloud(),
    deps = [
        ":diagnostics",
        ":errors",
        ":logical_result",
        ":map_by_type",
        ":type_id",
        "//tensorflow/compiler/xla:shape_util",
        "//third_party/eigen3",
        "@com_google_absl//absl/base:dynamic_annotations",
        "@llvm-project//llvm:Support",
    ],
)

tf_cc_test(
    name = "custom_call_test",
    srcs = ["custom_call_test.cc"],
    deps = [
        ":arguments",
        ":async_runtime",
        ":custom_call",
        ":custom_call_registry",
        ":diagnostics",
        ":executable",
        ":jit_executable",
        "//tensorflow/compiler/xla/mlir/ir/runtime/tests:testlib",
        "//tensorflow/compiler/xla/mlir/transforms/runtime:compilation_pipeline",
        "//tensorflow/compiler/xla/mlir/transforms/runtime:custom_call_encoding",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

cc_library(
    name = "custom_call_registry",
    srcs = ["custom_call_registry.cc"],
    hdrs = ["custom_call_registry.h"],
    compatible_with = get_compatible_with_cloud(),
    deps = [
        ":custom_call",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "diagnostics",
    srcs = ["diagnostics.cc"],
    hdrs = ["diagnostics.h"],
    compatible_with = get_compatible_with_cloud(),
    deps = [
        ":logical_result",
        "//tensorflow/tsl/platform:logging",
        "@com_google_absl//absl/status",
        "@llvm-project//llvm:Support",
    ],
)

tf_cc_test(
    name = "diagnostics_test",
    srcs = ["diagnostics_test.cc"],
    deps = [
        ":diagnostics",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "@com_google_absl//absl/status",
    ],
)

cc_library(
    name = "errors",
    hdrs = ["errors.h"],
    compatible_with = get_compatible_with_cloud(),
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:str_format",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "executable",
    srcs = ["executable.cc"],
    hdrs = ["executable.h"],
    compatible_with = get_compatible_with_cloud(),
    deps = [
        ":arguments",
        ":async_runtime",
        ":custom_call",
        ":custom_call_registry",
        ":diagnostics",
        ":errors",
        ":execution_engine",
        ":logical_result",
        ":memory_mapper",
        ":results",
        ":runtime",
        ":type_id",
        ":types",
        "//tensorflow/compiler/xla/mlir/utils/runtime:async_runtime_api",
        "//tensorflow/compiler/xla/mlir/utils/runtime:c_runner_utils",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@llvm-project//llvm:OrcJIT",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "execution_engine",
    srcs = ["execution_engine.cc"],
    hdrs = ["execution_engine.h"],
    compatible_with = get_compatible_with_cloud(),
    deps = [
        ":errors",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@llvm-project//llvm:ARMAsmParser",
        "@llvm-project//llvm:ARMCodeGen",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:ExecutionEngine",
        "@llvm-project//llvm:OrcJIT",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:X86AsmParser",
        "@llvm-project//llvm:X86CodeGen",
    ],
)

cc_library(
    name = "jit_executable",
    srcs = ["jit_executable.cc"],
    hdrs = ["jit_executable.h"],
    compatible_with = get_compatible_with_cloud(),
    deps = [
        ":arguments",
        ":async_values_cache",
        ":constraints",
        ":errors",
        "//tensorflow/compiler/xla/mlir/transforms/runtime:jit_compiler",
        "//tensorflow/compiler/xla/mlir/utils/runtime:constraints",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@llvm-project//llvm:Support",
        "@tf_runtime//:hostcontext",
    ],
)

cc_library(
    name = "logical_result",
    hdrs = ["logical_result.h"],
    compatible_with = get_compatible_with_cloud(),
    deps = ["@llvm-project//mlir:Support"],
)

cc_library(
    name = "map_by_type",
    hdrs = ["map_by_type.h"],
    compatible_with = get_compatible_with_cloud(),
    deps = [
        ":type_id",
        "@llvm-project//llvm:Support",
    ],
)

tf_cc_test(
    name = "map_by_type_test",
    srcs = ["map_by_type_test.cc"],
    deps = [
        ":map_by_type",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

cc_library(
    name = "memory_mapper",
    srcs = ["memory_mapper.cc"],
    hdrs = ["memory_mapper.h"],
    compatible_with = get_compatible_with_cloud(),
    deps = [
        "//tensorflow/tsl/platform:platform",
        "@llvm-project//llvm:ExecutionEngine",
        "@llvm-project//llvm:Support",
    ] + tf_platform_deps(
        "memory_mapper",
        platform_dir = "//tensorflow/compiler/xla/runtime/",
    ),
)

cc_library(
    name = "results",
    hdrs = ["results.h"],
    compatible_with = get_compatible_with_cloud(),
    deps = [
        ":logical_result",
        ":types",
    ],
)

cc_library(
    name = "runtime",
    hdrs = ["runtime.h"],
    compatible_with = get_compatible_with_cloud(),
)

cc_library(
    name = "symbolic_shape",
    srcs = ["symbolic_shape.cc"],
    hdrs = ["symbolic_shape.h"],
    compatible_with = get_compatible_with_cloud(),
    deps = [
        ":arguments",
        ":constraints",
        ":logical_result",
        ":types",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@llvm-project//llvm:Support",
    ],
)

tf_cc_test(
    name = "symbolic_shape_test",
    srcs = ["symbolic_shape_test.cc"],
    deps = [
        ":arguments",
        ":constraints",
        ":symbolic_shape",
        ":types",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "types",
    srcs = ["types.cc"],
    hdrs = ["types.h"],
    compatible_with = get_compatible_with_cloud(),
    deps = [
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "type_id",
    srcs = ["type_id.cc"],
    hdrs = ["type_id.h"],
    compatible_with = get_compatible_with_cloud(),
    deps = [
        "@com_google_absl//absl/container:flat_hash_map",
        "@llvm-project//mlir:Support",
    ],
)

tf_cc_test(
    name = "type_id_test",
    srcs = ["type_id_test.cc"],
    deps = [
        ":type_id",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)
