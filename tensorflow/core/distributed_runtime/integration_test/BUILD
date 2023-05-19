load(
    "//tensorflow:tensorflow.bzl",
    "tf_cc_test",
    "tf_cuda_cc_test",
    "tf_cuda_library",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//tensorflow:internal",
    ],
    licenses = ["notice"],
)

tf_cuda_library(
    name = "coordination_test_opkernel_registration",
    testonly = 1,
    srcs = ["coordination_test_opkernel_registration.cc"],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/tsl/distributed_runtime/coordination:coordination_service_agent",
        "//tensorflow/tsl/distributed_runtime/coordination:coordination_service_error_util",
    ],
    alwayslink = 1,
)

tf_cuda_cc_test(
    name = "c_api_coordination_test",
    size = "medium",
    srcs = ["c_api_coordination_test.cc"],
    # copybara:uncomment extra_copts = ["-Wthread-safety-analysis"],
    tags = [
        "no_cuda_asan",  # TODO(b/193450885)
        "no_windows",  # TODO(b/207281588)
    ],
    deps = [
        ":coordination_test_opkernel_registration",
        "//tensorflow/c:c_api_experimental",
        "//tensorflow/c/eager:c_api",
        "//tensorflow/c/eager:c_api_experimental",
        "//tensorflow/c/eager:c_api_internal",
        "//tensorflow/c/eager:c_api_test_util",
        "//tensorflow/c/eager:tfe_tensorhandle_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/distributed_runtime:server_lib",
        "//tensorflow/core/platform:blocking_counter",
        "//tensorflow/core/platform:env",
        "//tensorflow/tsl/lib/core:status_test_util",
        "@com_google_absl//absl/time",
    ],
)

tf_cuda_cc_test(
    name = "c_api_session_coordination_test",
    size = "small",
    srcs = ["c_api_session_coordination_test.cc"],
    # copybara:uncomment extra_copts = ["-Wthread-safety-analysis"],
    tags = [
        "no_windows",  # TODO(b/207281588)
    ],
    deps = [
        ":coordination_test_opkernel_registration",
        "//tensorflow/c:c_api",
        "//tensorflow/c:c_api_experimental",
        "//tensorflow/c:c_test_util",
        "//tensorflow/c/eager:c_api_test_util",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/distributed_runtime:server_lib",
        "//tensorflow/core/distributed_runtime/rpc:grpc_session",
        "//tensorflow/core/platform:env",
    ],
)

tf_cuda_cc_test(
    name = "c_api_multi_client_test",
    size = "medium",
    srcs = ["c_api_multi_client_test.cc"],
    # copybara:uncomment extra_copts = ["-Wthread-safety-analysis"],
    tags = [
        "no_windows",  # TODO(b/207281588)
    ],
    deps = [
        "//tensorflow/c:c_api_experimental",
        "//tensorflow/c:c_test_util",
        "//tensorflow/c/eager:c_api",
        "//tensorflow/c/eager:c_api_experimental",
        "//tensorflow/c/eager:c_api_internal",
        "//tensorflow/c/eager:c_api_test_util",
        "//tensorflow/c/eager:tfe_context_internal",
        "//tensorflow/c/eager:tfe_tensorhandle_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/common_runtime/eager:context",
        "//tensorflow/core/distributed_runtime:server_lib",
        "//tensorflow/core/platform:env",
        "//tensorflow/tsl/distributed_runtime/coordination:coordination_service",
        "//tensorflow/tsl/distributed_runtime/coordination:coordination_service_agent",
    ],
)

tf_cc_test(
    name = "c_api_multi_client_function_test",
    size = "small",
    srcs = ["c_api_multi_client_function_test.cc"],
    # copybara:uncomment extra_copts = ["-Wthread-safety-analysis"],
    tags = [
        "no_oss",  # test uses TFRT
    ],
    deps = [
        "//tensorflow/c:c_api_experimental",
        "//tensorflow/c:c_test_util",
        "//tensorflow/c/eager:c_api",
        "//tensorflow/c/eager:c_api_experimental",
        "//tensorflow/c/eager:c_api_internal",
        "//tensorflow/c/eager:c_api_test_util",
        "//tensorflow/c/eager:tfe_context_internal",
        "//tensorflow/c/eager:tfe_op_internal",
        "//tensorflow/c/eager:tfe_tensorhandle_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/common_runtime/eager:context",
        "//tensorflow/core/common_runtime/eager:eager_operation",
        "//tensorflow/core/common_runtime/eager:kernel_and_device",
        "//tensorflow/core/distributed_runtime:server_lib",
        "//tensorflow/core/platform:blocking_counter",
        "//tensorflow/core/platform:env",
        "//tensorflow/tsl/distributed_runtime/coordination:coordination_service",
        "//tensorflow/tsl/distributed_runtime/coordination:coordination_service_agent",
    ],
)

tf_cuda_cc_test(
    name = "c_api_recoverable_jobs_test",
    size = "small",
    srcs = ["c_api_recoverable_jobs_test.cc"],
    # copybara:uncomment extra_copts = ["-Wthread-safety-analysis"],
    tags = [
        "no_cuda_asan",  # TODO(b/193450885)
        "no_windows",  # TODO(b/207281588)
    ],
    deps = [
        ":coordination_test_opkernel_registration",
        "//tensorflow/c:c_api_experimental",
        "//tensorflow/c/eager:c_api",
        "//tensorflow/c/eager:c_api_experimental",
        "//tensorflow/c/eager:c_api_internal",
        "//tensorflow/c/eager:c_api_test_util",
        "//tensorflow/c/eager:tfe_tensorhandle_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/distributed_runtime:server_lib",
        "//tensorflow/core/platform:env",
        "//tensorflow/tsl/lib/core:status_test_util",
        "@com_google_absl//absl/time",
    ],
)
