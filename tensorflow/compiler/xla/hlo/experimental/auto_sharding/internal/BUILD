package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [":friends"],
)

package_group(
    name = "friends",
    packages = [
        "//tensorflow/compiler/xla/hlo/experimental/auto_sharding/...",
    ],
)

cc_library(
    name = "auto_sharding_impl_internal",
    srcs = ["auto_sharding_impl_internal.cc"],
    deps = [
        "//tensorflow/compiler/xla/hlo/experimental/auto_sharding:auto_sharding_cost_graph",
        "//tensorflow/compiler/xla/hlo/experimental/auto_sharding:auto_sharding_option",
        "//tensorflow/compiler/xla/hlo/experimental/auto_sharding:auto_sharding_solver",
        "//tensorflow/compiler/xla/hlo/experimental/auto_sharding:auto_sharding_strategy",
        "//tensorflow/compiler/xla/hlo/experimental/auto_sharding:auto_sharding_wrapper",
        "//tensorflow/compiler/xla/hlo/utils:hlo_live_range",
    ],
)
