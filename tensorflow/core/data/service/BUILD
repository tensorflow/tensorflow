load("@com_github_grpc_grpc//bazel:cc_grpc_library.bzl", "cc_grpc_library")
load(
    "//tensorflow/core/platform:build_config.bzl",
    "tf_additional_all_protos",
    "tf_proto_library",
)
load(
    "//tensorflow:tensorflow.bzl",
    "tf_cc_test",
)

package(
    default_visibility = [
        "//tensorflow:internal",
    ],
    licenses = ["notice"],  # Apache 2.0
)

exports_files(["LICENSE"])

tf_proto_library(
    name = "common_proto",
    srcs = ["common.proto"],
    cc_api_version = 2,
    protodeps = tf_additional_all_protos(),
)

tf_proto_library(
    name = "master_proto",
    srcs = ["master.proto"],
    has_services = 1,
    cc_api_version = 2,
    protodeps = tf_additional_all_protos() + [
        ":common_proto",
    ],
)

tf_proto_library(
    name = "worker_proto",
    srcs = ["worker.proto"],
    has_services = 1,
    cc_api_version = 2,
    protodeps = tf_additional_all_protos() + [
        ":common_proto",
    ],
)

cc_library(
    name = "master_impl",
    srcs = ["master_impl.cc"],
    hdrs = [
        "master_impl.h",
    ],
    deps = [
        ":common_proto_cc",
        ":credentials_factory",
        ":grpc_util",
        ":master_proto_cc",
        ":worker_cc_grpc_proto",
        ":worker_proto_cc",
        "//tensorflow:grpc++",
        "//tensorflow/c:c_api_internal",
        "//tensorflow/c:tf_status_helper",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/kernels/data:dataset_utils",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/memory",
    ],
)

cc_library(
    name = "worker_impl",
    srcs = ["worker_impl.cc"],
    hdrs = [
        "worker_impl.h",
    ],
    deps = [
        ":common_proto_cc",
        ":compression_utils",
        ":credentials_factory",
        ":grpc_util",
        ":master_cc_grpc_proto",
        ":master_proto_cc",
        ":worker_proto_cc",
        "//tensorflow:grpc++",
        "//tensorflow/c:c_api_internal",
        "//tensorflow/c:tf_status_helper",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/data:standalone",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/memory",
    ],
)

cc_library(
    name = "grpc_util",
    srcs = ["grpc_util.cc"],
    hdrs = [
        "grpc_util.h",
    ],
    deps = [
        "//tensorflow:grpc++",
        "//tensorflow/core:lib",
    ],
)

tf_cc_test(
    name = "grpc_util_test",
    srcs = ["grpc_util_test.cc"],
    deps = [
        ":grpc_util",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

cc_library(
    name = "compression_utils",
    srcs = ["compression_utils.cc"],
    hdrs = [
        "compression_utils.h",
    ],
    deps = [
        ":common_proto_cc",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/profiler/lib:traceme",
        "@com_google_absl//absl/memory",
    ],
)

tf_cc_test(
    name = "compression_utils_test",
    srcs = ["compression_utils_test.cc"],
    deps = [
        ":compression_utils",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/kernels/data:dataset_test_base",
    ],
)

cc_library(
    name = "credentials_factory",
    srcs = ["credentials_factory.cc"],
    hdrs = ["credentials_factory.h"],
    deps = [
        "//tensorflow:grpc++",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "credentials_factory_test",
    srcs = ["credentials_factory_test.cc"],
    deps = [
        ":credentials_factory",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

# Link this target to enable LOCAL credentials for the dataset service.
cc_library(
    name = "local_credentials_factory",
    srcs = ["local_credentials_factory.cc"],
    deps = [
        ":credentials_factory",
        "//tensorflow:grpc++",
    ],
    alwayslink = 1,
)

cc_library(
    name = "test_util",
    testonly = True,
    srcs = ["test_util.cc"],
    hdrs = [
        "test_util.h",
    ],
    data = glob(["testdata/*.pbtxt"]),
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core/framework:protos_all_cc",
        "//tensorflow/core/kernels/data:dataset_test_base",
    ],
)

tf_cc_test(
    name = "test_util_test",
    srcs = ["test_util_test.cc"],
    deps = [
        ":test_util",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/data:standalone",
        "//tensorflow/core/kernels/data:dataset_test_base",
    ],
)

cc_library(
    name = "grpc_master_impl",
    srcs = ["grpc_master_impl.cc"],
    hdrs = ["grpc_master_impl.h"],
    deps = [
        ":master_cc_grpc_proto",
        ":master_impl",
        "//tensorflow:grpc++",
        "//tensorflow/core/distributed_runtime/rpc:grpc_util",
    ],
)

cc_library(
    name = "grpc_worker_impl",
    srcs = ["grpc_worker_impl.cc"],
    hdrs = ["grpc_worker_impl.h"],
    deps = [
        ":worker_cc_grpc_proto",
        ":worker_impl",
        "//tensorflow:grpc++",
        "//tensorflow/core/distributed_runtime/rpc:grpc_util",
    ],
)

cc_library(
    name = "server_lib",
    srcs = ["server_lib.cc"],
    hdrs = ["server_lib.h"],
    deps = [
        ":credentials_factory",
        ":grpc_master_impl",
        ":grpc_worker_impl",
        "//tensorflow:grpc++",
        "//tensorflow/core:lib",
        "//tensorflow/core:tensorflow",
    ],
)

tf_cc_test(
    name = "data_service_test",
    srcs = ["data_service_test.cc"],
    deps = [
        ":compression_utils",
        ":grpc_master_impl",
        ":grpc_util",
        ":grpc_worker_impl",
        ":local_credentials_factory",
        ":master_cc_grpc_proto",
        ":master_proto_cc",
        ":server_lib",
        ":test_util",
        ":worker_cc_grpc_proto",
        ":worker_proto_cc",
        "//tensorflow:grpc++",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/kernels/data:dataset_test_base",
        "@com_google_absl//absl/strings",
    ],
)

cc_grpc_library(
    name = "master_cc_grpc_proto",
    srcs = [":master_proto"],
    generate_mocks = True,
    grpc_only = True,
    deps = [":master_proto_cc"],
)

cc_grpc_library(
    name = "worker_cc_grpc_proto",
    srcs = [":worker_proto"],
    generate_mocks = True,
    grpc_only = True,
    deps = [":worker_proto_cc"],
)
