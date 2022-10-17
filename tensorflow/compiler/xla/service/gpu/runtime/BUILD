package(
    default_visibility = [":friends"],
    licenses = ["notice"],
)

package_group(
    name = "friends",
    includes = [
        "//tensorflow/compiler/xla:friends",
    ],
)

cc_library(
    name = "graph_launch",
    srcs = ["graph_launch.cc"],
    hdrs = ["graph_launch.h"],
    deps = [
        ":kernel_launch",
        ":support",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla/runtime:custom_call",
        "//tensorflow/compiler/xla/runtime:custom_call_registry",
        "//tensorflow/compiler/xla/runtime:executable",
        "//tensorflow/compiler/xla/service:executable",
        "//tensorflow/compiler/xla/service/gpu:launch_dimensions",
        "//tensorflow/compiler/xla/service/gpu:stream_executor_util",
        "//tensorflow/compiler/xla/stream_executor",
        "//tensorflow/compiler/xla/stream_executor/gpu:gpu_stream",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/synchronization",
        "@local_config_cuda//cuda:cuda_headers",
    ],
)

cc_library(
    name = "kernel_launch",
    srcs = ["kernel_launch.cc"],
    hdrs = ["kernel_launch.h"],
    deps = [
        ":support",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla/runtime:custom_call",
        "//tensorflow/compiler/xla/runtime:custom_call_registry",
        "//tensorflow/compiler/xla/runtime:executable",
        "//tensorflow/compiler/xla/service:executable",
        "//tensorflow/compiler/xla/service/gpu:launch_dimensions",
        "//tensorflow/compiler/xla/service/gpu:stream_executor_util",
        "//tensorflow/compiler/xla/stream_executor",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_library(
    name = "support",
    hdrs = ["support.h"],
    deps = [
        "//tensorflow/compiler/xla/runtime:custom_call",
        "//tensorflow/compiler/xla/stream_executor:device_memory",
    ],
)

cc_library(
    name = "tracing",
    srcs = ["tracing.cc"],
    hdrs = ["tracing.h"],
    deps = [
        ":support",
        "//tensorflow/compiler/xla/runtime:custom_call",
        "//tensorflow/compiler/xla/runtime:custom_call_registry",
        "//tensorflow/compiler/xla/runtime:executable",
        "//tensorflow/compiler/xla/runtime:tracing",
        "//tensorflow/compiler/xla/runtime:type_id",
        "//tensorflow/tsl/profiler/lib:scoped_annotation_stack",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
    ],
)
