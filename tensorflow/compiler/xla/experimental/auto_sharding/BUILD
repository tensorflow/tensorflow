# Automatic sharding annotation

load("//tensorflow:tensorflow.bzl", "tf_cc_binary")

package(default_visibility = [":friends"])

package_group(
    name = "friends",
    packages = [
        "//platforms/...",
    ],
)

cc_library(
    name = "auto_sharding",
    srcs = [
        "auto_sharding.cc",
        "auto_sharding_dot_handler.cc",
        "auto_sharding_util.cc",
    ],
    hdrs = [
        "auto_sharding.h",
        "auto_sharding_cost_graph.h",
        "auto_sharding_strategy.h",
        "auto_sharding_util.h",
    ],
    deps = [
        "//tensorflow/compiler/xla:array",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla/service:dump",
        "//tensorflow/compiler/xla/service:heap_simulator",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_live_range",
        "//tensorflow/compiler/xla/service:hlo_memory_scheduler",
        "//tensorflow/compiler/xla/service:hlo_ordering",
        "//tensorflow/compiler/xla/service:hlo_pass",
        "//tensorflow/compiler/xla/service:hlo_sharding_util",
        "//tensorflow/compiler/xla/service:sharding_propagation",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/protobuf:error_codes_proto_impl_cc",
        "//tensorflow/tsl/platform:status",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@com_google_ortools//ortools/linear_solver",
        "@com_google_ortools//ortools/linear_solver:linear_solver_cc_proto",
    ],
)

tf_cc_binary(
    name = "auto_sharding_runner",
    srcs = ["auto_sharding_runner.cc"],
    deps = [
        ":auto_sharding",
        "//tensorflow/compiler/xla:status",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_parser",
        "//tensorflow/compiler/xla/tools:hlo_module_loader",
        "//tensorflow/tsl/platform:platform_port",
    ],
)
