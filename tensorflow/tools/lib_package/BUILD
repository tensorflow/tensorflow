# Packaging for TensorFlow artifacts other than the Python API (pip whl).
# This includes the C API, Java API, and protocol buffer srcs.

package(default_visibility = ["//visibility:private"])

load("@bazel_tools//tools/build_defs/pkg:pkg.bzl", "pkg_tar")
load("@local_config_syslibs//:build_defs.bzl", "if_not_system_lib")
load("//tensorflow:tensorflow.bzl", "tf_binary_additional_srcs")
load("//tensorflow:tensorflow.bzl", "if_cuda")
load("//tensorflow:tensorflow.bzl", "if_not_windows")
load("//third_party/mkl:build_defs.bzl", "if_mkl")

genrule(
    name = "libtensorflow_proto",
    srcs = ["//tensorflow/core:protos_all_proto_srcs"],
    outs = ["libtensorflow_proto.zip"],
    cmd = "zip $@ $(SRCS)",
)

pkg_tar(
    name = "libtensorflow",
    extension = "tar.gz",
    # Mark as "manual" till
    # https://github.com/bazelbuild/bazel/issues/2352
    # and https://github.com/bazelbuild/bazel/issues/1580
    # are resolved, otherwise these rules break when built
    # with Python 3.
    tags = ["manual"],
    deps = [
        ":cheaders",
        ":clib",
        ":clicenses",
        ":eager_cheaders",
    ],
)

pkg_tar(
    name = "libtensorflow_cc",
    extension = "tar.gz",
    # Mark as "manual" till
    # https://github.com/bazelbuild/bazel/issues/2352
    # and https://github.com/bazelbuild/bazel/issues/1580
    # are resolved, otherwise these rules break when built
    # with Python 3.
    tags = ["manual"],
    deps = [
        ":cheaders",
        ":eager_cheaders",
        ":ccheaders",
        ":coreheaders",
        ":clib",
        ":cclib",
        ":clicenses",
        ":cclicenses",
    ],
)

pkg_tar(
    name = "libtensorflow_jni",
    extension = "tar.gz",
    srcs = [
        "include/tensorflow/jni/LICENSE",
        "//:LICENSE",
        "//tensorflow/java:libtensorflow_jni",
    ],
    # Mark as "manual" till
    # https://github.com/bazelbuild/bazel/issues/2352
    # and https://github.com/bazelbuild/bazel/issues/1580
    # are resolved, otherwise these rules break when built
    # with Python 3.
    tags = ["manual"],
    deps = [":common_deps"],
)

# Shared objects that all TensorFlow libraries depend on.
pkg_tar(
    name = "common_deps",
    srcs = tf_binary_additional_srcs(),
    tags = ["manual"],
)

pkg_tar(
    name = "cheaders",
    srcs = [
        "//tensorflow/c:headers",
    ],
    package_dir = "include/tensorflow/c",
    # Mark as "manual" till
    # https://github.com/bazelbuild/bazel/issues/2352
    # and https://github.com/bazelbuild/bazel/issues/1580
    # are resolved, otherwise these rules break when built
    # with Python 3.
    tags = ["manual"],
)

pkg_tar(
    name = "eager_cheaders",
    srcs = [
        "//tensorflow/c/eager:headers",
    ],
    package_dir = "include/tensorflow/c/eager",
    # Mark as "manual" till
    # https://github.com/bazelbuild/bazel/issues/2352
    # and https://github.com/bazelbuild/bazel/issues/1580
    # are resolved, otherwise these rules break when built
    # with Python 3.
    tags = ["manual"],
)

pkg_tar(
    name = "ccheaders",
    package_dir = "include/tensorflow/cc",
    # Mark as "manual" till
    # https://github.com/bazelbuild/bazel/issues/2352
    # and https://github.com/bazelbuild/bazel/issues/1580
    # are resolved, otherwise these rules break when built
    # with Python 3.
    tags = ["manual"],
    deps = [
        ":client_ccheaders",
        ":framework_ccheaders",
        ":gradient_ccheaders",
        ":ops_ccheaders",
        ":profiler_ccheaders",
        ":saved_model_ccheaders",
        ":tools_ccheaders",
        ":training_ccheaders",
    ],
)

pkg_tar(
    name = "client_ccheaders",
    srcs = [
        "//tensorflow/cc:client_headers",
    ],
    package_dir = "client",
    tags = ["manual"],
)

pkg_tar(
    name = "framework_ccheaders",
    srcs = [
        "//tensorflow/cc:framework_headers",
    ],
    package_dir = "framework",
    tags = ["manual"],
)

pkg_tar(
    name = "gradient_ccheaders",
    srcs = [
        "//tensorflow/cc:gradient_headers",
    ],
    package_dir = "gradient",
    tags = ["manual"],
)

pkg_tar(
    name = "ops_ccheaders",
    srcs = [
        "//tensorflow/cc:ops_headers",
    ],
    package_dir = "ops",
    tags = ["manual"],
)

pkg_tar(
    name = "profiler_ccheaders",
    srcs = [
        "//tensorflow/cc/profiler:headers",
    ],
    package_dir = "profiler",
    tags = ["manual"],
)

pkg_tar(
    name = "saved_model_ccheaders",
    srcs = [
        "//tensorflow/cc/saved_model:headers",
    ],
    package_dir = "saved_model",
    tags = ["manual"],
)

pkg_tar(
    name = "tools_ccheaders",
    srcs = [
        "//tensorflow/cc/tools:headers",
    ],
    package_dir = "tools",
    tags = ["manual"],
)

pkg_tar(
    name = "training_ccheaders",
    srcs = [
        "//tensorflow/cc:training_headers",
    ],
    package_dir = "training",
    tags = ["manual"],
)

# to be completed
pkg_tar(
    name = "coreheaders",
    package_dir = "include/tensorflow/core",
    # Mark as "manual" till
    # https://github.com/bazelbuild/bazel/issues/2352
    # and https://github.com/bazelbuild/bazel/issues/1580
    # are resolved, otherwise these rules break when built
    # with Python 3.
    tags = ["manual"],
    deps = [
        ":common_runtime_coreheaders",
        ":debug_coreheaders",
        ":distributed_runtime_coreheaders",
        ":framework_coreheaders",
        ":graph_coreheaders",
        ":grappler_coreheaders",
        ":kernels_coreheaders",
        ":lib_coreheaders",
        ":nccl_coreheaders",
        ":ops_coreheaders",
        ":platform_coreheaders",
        ":protobuf_coreheaders",
        ":profiler_coreheaders",
        ":public_coreheaders",
        ":summary_coreheaders",
        ":tpu_coreheaders",
        ":util_coreheaders",
    ],
)

pkg_tar(
    name = "common_runtime_coreheaders",
    srcs = [
        "//tensorflow/core:common_runtime_headers",
    ],
    package_dir = "common_runtime",
    tags = ["manual"],
    deps = [
        "common_runtime_data_coreheaders",
        "common_runtime_eager_coreheaders",
        "common_runtime_gpu_coreheaders",
        "common_runtime_sycl_coreheaders",
    ],
)

pkg_tar(
    name = "common_runtime_data_coreheaders",
    srcs = [
        "//tensorflow/core/common_runtime/data:headers",
    ],
    package_dir = "data",
    tags = ["manual"],
)

pkg_tar(
    name = "common_runtime_eager_coreheaders",
    srcs = [
        "//tensorflow/core/common_runtime/eager:headers",
    ],
    package_dir = "eager",
    tags = ["manual"],
)

pkg_tar(
    name = "common_runtime_gpu_coreheaders",
    srcs = [
        "//tensorflow/core:common_runtime_gpu_headers",
    ],
    package_dir = "gpu",
    tags = ["manual"],
)

pkg_tar(
    name = "common_runtime_sycl_coreheaders",
    srcs = [
        "//tensorflow/core:common_runtime_sycl_headers",
    ],
    package_dir = "sycl",
    tags = ["manual"],
)

pkg_tar(
    name = "debug_coreheaders",
    srcs = [
        "//tensorflow/core/debug:headers",
    ],
    package_dir = "common_runtime",
    tags = ["manual"],
)

pkg_tar(
    name = "distributed_runtime_coreheaders",
    srcs = [
        "//tensorflow/core/distributed_runtime:headers",
    ],
    package_dir = "distributed_runtime",
    tags = ["manual"],
    deps = [
        "distributed_runtime_eager_headers",
        "distributed_runtime_rpc_headers",
        "distributed_runtime_rpc_eager_headers",
    ],
)

pkg_tar(
    name = "distributed_runtime_eager_headers",
    srcs = [
        "//tensorflow/core/distributed_runtime/eager:headers",
    ],
    package_dir = "eager",
    tags = ["manual"],
)

pkg_tar(
    name = "distributed_runtime_rpc_headers",
    srcs = [
        "//tensorflow/core/distributed_runtime/rpc:headers",
    ],
    package_dir = "rpc",
    tags = ["manual"],
)

pkg_tar(
    name = "distributed_runtime_rpc_eager_headers",
    srcs = [
        "//tensorflow/core/distributed_runtime/rpc/eager:headers",
    ],
    package_dir = "rpc/eager",
    tags = ["manual"],
)

pkg_tar(
    name = "framework_coreheaders",
    srcs = [
        "//tensorflow/core:framework_headers",
    ],
    package_dir = "framework",
    tags = ["manual"],
)

pkg_tar(
    name = "graph_coreheaders",
    srcs = [
        "//tensorflow/core:graph_headers",
    ],
    package_dir = "graph",
    tags = ["manual"],
)

pkg_tar(
    name = "grappler_coreheaders",
    srcs = [
        "//tensorflow/core/grappler:headers",
    ],
    package_dir = "grappler",
    tags = ["manual"],
    deps = [
        ":grappler_clusters_coreheaders",
        ":grappler_costs_coreheaders",
        ":grappler_graph_analyzer_coreheaders",
        ":grappler_inputs_coreheaders",
        ":grappler_optimizers_coreheaders",
        ":grappler_utils_coreheaders",
        ":grappler_verifiers_coreheaders",
    ],
)

pkg_tar(
    name = "grappler_clusters_coreheaders",
    srcs = [
        "//tensorflow/core/grappler/clusters:headers",
    ],
    package_dir = "clusters",
    tags = ["manual"],
)

pkg_tar(
    name = "grappler_costs_coreheaders",
    srcs = [
        "//tensorflow/core/grappler/costs:headers",
    ],
    package_dir = "costs",
    tags = ["manual"],
)

pkg_tar(
    name = "grappler_graph_analyzer_coreheaders",
    srcs = [
        "//tensorflow/core/grappler/graph_analyzer:headers",
    ],
    package_dir = "graph_analyzer",
    tags = ["manual"],
)

pkg_tar(
    name = "grappler_inputs_coreheaders",
    srcs = [
        "//tensorflow/core/grappler/inputs:headers",
    ],
    package_dir = "graph_analyzer",
    tags = ["manual"],
)

pkg_tar(
    name = "grappler_utils_coreheaders",
    srcs = [
        "//tensorflow/core/grappler/utils:headers",
    ],
    package_dir = "utils",
    tags = ["manual"],
)

pkg_tar(
    name = "grappler_verifiers_coreheaders",
    srcs = [
        "//tensorflow/core/grappler/verifiers:headers",
    ],
    package_dir = "verifiers",
    tags = ["manual"],
)

pkg_tar(
    name = "grappler_optimizers_coreheaders",
    srcs = [
        "//tensorflow/core/grappler/optimizers:headers",
    ],
    package_dir = "optimizers",
    tags = ["manual"],
)

pkg_tar(
    name = "kernels_coreheaders",
    srcs = [
        "//tensorflow/core/kernels:headers",
    ],
    package_dir = "kernels",
    tags = ["manual"],
    deps = [
        ":kernels_batching_util_coreheaders",
        ":kernels_boosted_trees_coreheaders",
        ":kernels_boosted_trees_quantiles_coreheaders",
        ":kernels_data_coreheaders",
        ":kernels_data_experimental_sql_coreheaders",
        ":kernels_fuzzing_coreheaders",
        ":kernels_hexagon_coreheaders",
        ":kernels_lookup_tables_coreheaders",
        ":kernels_neon_coreheaders",
        ":kernels_tensor_forest_coreheaders",
    ],
)

pkg_tar(
    name = "kernels_batching_util_coreheaders",
    srcs = [
        "//tensorflow/core/kernels/batching_util:headers",
    ],
    package_dir = "batching_util",
    tags = ["manual"],
)

pkg_tar(
    name = "kernels_boosted_trees_coreheaders",
    srcs = [
        "//tensorflow/core/kernels/boosted_trees:headers",
    ],
    package_dir = "boosted_trees",
    tags = ["manual"],
)

pkg_tar(
    name = "kernels_boosted_trees_quantiles_coreheaders",
    srcs = [
        "//tensorflow/core/kernels/boosted_trees/quantiles:headers",
    ],
    package_dir = "boosted_trees/quantiles",
    tags = ["manual"],
)

pkg_tar(
    name = "kernels_data_coreheaders",
    srcs = [
        "//tensorflow/core/kernels/data:headers",
    ],
    package_dir = "data",
    tags = ["manual"],
)

pkg_tar(
    name = "kernels_data_experimental_sql_coreheaders",
    srcs = [
        "//tensorflow/core/kernels/data/experimental/sql:headers",
    ],
    package_dir = "data/experimental/sql",
    tags = ["manual"],
)

pkg_tar(
    name = "kernels_fuzzing_coreheaders",
    srcs = [
        "//tensorflow/core/kernels/fuzzing:headers",
    ],
    package_dir = "fuzzing",
    tags = ["manual"],
)

pkg_tar(
    name = "kernels_hexagon_coreheaders",
    srcs = [
        "//tensorflow/core/kernels/hexagon:headers",
    ],
    package_dir = "hexagon",
    tags = ["manual"],
)

pkg_tar(
    name = "kernels_lookup_tables_coreheaders",
    srcs = [
        "//tensorflow/core/kernels/lookup_tables:headers",
    ],
    package_dir = "lookup_tables",
    tags = ["manual"],
)

pkg_tar(
    name = "kernels_neon_coreheaders",
    srcs = [
        "//tensorflow/core/kernels/neon:headers",
    ],
    package_dir = "neon",
    tags = ["manual"],
)

pkg_tar(
    name = "kernels_tensor_forest_coreheaders",
    srcs = [
        "//tensorflow/core/kernels/tensor_forest:headers",
    ],
    package_dir = "tensor_forest",
    tags = ["manual"],
)

pkg_tar(
    name = "lib_coreheaders",
    package_dir = "lib",
    tags = ["manual"],
    deps = [
        ":lib_bfloat16_coreheaders",
        ":lib_core_coreheaders",
        ":lib_db_coreheaders",
        ":lib_gif_coreheaders",
        ":lib_gtl_coreheaders",
        ":lib_hash_coreheaders",
        ":lib_histogram_coreheaders",
        ":lib_io_coreheaders",
        ":lib_jpeg_coreheaders",
        ":lib_math_coreheaders",
        ":lib_monitoring_coreheaders",
        ":lib_png_coreheaders",
        ":lib_random_coreheaders",
        ":lib_strings_coreheaders",
        ":lib_wav_coreheaders",
    ],
)

pkg_tar(
    name = "lib_bfloat16_coreheaders",
    srcs = [
        "//tensorflow/core:lib_bfloat16_headers",
    ],
    package_dir = "bfloat16",
    tags = ["manual"],
)

pkg_tar(
    name = "lib_core_coreheaders",
    srcs = [
        "//tensorflow/core:lib_core_headers",
    ],
    package_dir = "core",
    tags = ["manual"],
)

pkg_tar(
    name = "lib_db_coreheaders",
    srcs = [
        "//tensorflow/core/lib/db:headers",
    ],
    package_dir = "db",
    tags = ["manual"],
)

pkg_tar(
    name = "lib_gif_coreheaders",
    srcs = [
        "//tensorflow/core:lib_gif_headers",
    ],
    package_dir = "gif",
    tags = ["manual"],
)

pkg_tar(
    name = "lib_gtl_coreheaders",
    srcs = [
        "//tensorflow/core:lib_gtl_headers",
    ],
    package_dir = "gtl",
    tags = ["manual"],
)

pkg_tar(
    name = "lib_hash_coreheaders",
    srcs = [
        "//tensorflow/core:lib_hash_headers",
    ],
    package_dir = "hash",
    tags = ["manual"],
)

pkg_tar(
    name = "lib_histogram_coreheaders",
    srcs = [
        "//tensorflow/core:lib_histogram_headers",
    ],
    package_dir = "histogram",
    tags = ["manual"],
)

pkg_tar(
    name = "lib_io_coreheaders",
    srcs = [
        "//tensorflow/core:lib_io_headers",
    ],
    package_dir = "io",
    tags = ["manual"],
    deps = [
        ":lib_io_snappy_coreheaders",
    ],
)

pkg_tar(
    name = "lib_io_snappy_coreheaders",
    srcs = [
        "//tensorflow/core:lib_io_snappy_headers",
    ],
    package_dir = "snappy",
    tags = ["manual"],
)

pkg_tar(
    name = "lib_jpeg_coreheaders",
    srcs = [
        "//tensorflow/core:lib_jpeg_headers",
    ],
    package_dir = "jpeg",
    tags = ["manual"],
)

pkg_tar(
    name = "lib_math_coreheaders",
    srcs = [
        "//tensorflow/core:lib_math_headers",
    ],
    package_dir = "math",
    tags = ["manual"],
)

pkg_tar(
    name = "lib_monitoring_coreheaders",
    srcs = [
        "//tensorflow/core:lib_monitoring_headers",
    ],
    package_dir = "monitoring",
    tags = ["manual"],
)

pkg_tar(
    name = "lib_png_coreheaders",
    srcs = [
        "//tensorflow/core:lib_png_headers",
    ],
    package_dir = "png",
    tags = ["manual"],
)

pkg_tar(
    name = "lib_random_coreheaders",
    srcs = [
        "//tensorflow/core:lib_random_headers",
    ],
    package_dir = "random",
    tags = ["manual"],
)

pkg_tar(
    name = "lib_strings_coreheaders",
    srcs = [
        "//tensorflow/core:lib_strings_headers",
    ],
    package_dir = "strings",
    tags = ["manual"],
)

pkg_tar(
    name = "lib_wav_coreheaders",
    srcs = [
        "//tensorflow/core:lib_strings_headers",
    ],
    package_dir = "wav",
    tags = ["manual"],
)

pkg_tar(
    name = "nccl_coreheaders",
    srcs = [
        "//tensorflow/core/nccl:headers",
    ],
    package_dir = "nccl",
    tags = ["manual"],
)

pkg_tar(
    name = "ops_coreheaders",
    package_dir = "ops",
    tags = ["manual"],
    deps = [
        ":ops_compat_headers",
    ],
)

pkg_tar(
    name = "ops_compat_headers",
    srcs = [
        "//tensorflow/core/ops/compat:headers",
    ],
    package_dir = "compat",
    tags = ["manual"],
)

pkg_tar(
    name = "platform_coreheaders",
    srcs = [
        "//tensorflow/core:platform_headers",
    ],
    package_dir = "platform",
    tags = ["manual"],
    deps = [
        ":platform_cloud_coreheaders",
        ":platform_default_coreheaders",
        ":platform_hadoop_coreheaders",
        ":platform_posix_coreheaders",
        ":platform_profile_utils_coreheaders",
        ":platform_s3_coreheaders",
        ":platform_windows_coreheaders",
    ],
)

pkg_tar(
    name = "platform_cloud_coreheaders",
    srcs = [
        "//tensorflow/core/platform/cloud:headers",
    ],
    package_dir = "cloud",
    tags = ["manual"],
)

pkg_tar(
    name = "platform_default_coreheaders",
    srcs = [
        "//tensorflow/core:platform_default_headers",
    ],
    package_dir = "default",
    tags = ["manual"],
)

pkg_tar(
    name = "platform_hadoop_coreheaders",
    srcs = [
        "//tensorflow/core/platform/hadoop:headers",
    ],
    package_dir = "hadoop",
    tags = ["manual"],
)

pkg_tar(
    name = "platform_posix_coreheaders",
    srcs = [
        "//tensorflow/core:platform_posix_headers",
    ],
    package_dir = "posix",
    tags = ["manual"],
)

pkg_tar(
    name = "platform_profile_utils_coreheaders",
    srcs = [
        "//tensorflow/core:platform_profile_utils_headers",
    ],
    package_dir = "profile_utils",
    tags = ["manual"],
)

pkg_tar(
    name = "platform_s3_coreheaders",
    srcs = [
        "//tensorflow/core/platform/s3:headers",
    ],
    package_dir = "s3",
    tags = ["manual"],
)

pkg_tar(
    name = "platform_windows_coreheaders",
    srcs = [
        "//tensorflow/core:platform_windows_headers",
    ],
    package_dir = "windows",
    tags = ["manual"],
)

pkg_tar(
    name = "protobuf_coreheaders",
    srcs = [
        "//tensorflow/core:protobuf_headers",
    ],
    package_dir = "protobuf",
    tags = ["manual"],
)

pkg_tar(
    name = "protobuf_tpu_coreheaders",
    srcs = [
        "//tensorflow/core/protobuf/tpu:headers",
    ],
    package_dir = "tpu",
    tags = ["manual"],
)

pkg_tar(
    name = "profiler_coreheaders",
    srcs = [
        "//tensorflow/core/profiler:headers",
    ],
    package_dir = "profiler",
    tags = ["manual"],
    deps = [
        ":profiler_internal_coreheaders",
        ":profiler_lib_coreheaders",
        ":profiler_rpc_coreheaders",
    ],
)

pkg_tar(
    name = "profiler_internal_coreheaders",
    srcs = [
        "//tensorflow/core/profiler/internal:headers",
    ],
    package_dir = "internal",
    tags = ["manual"],
    deps = [
        ":profiler_internal_advisor_coreheaders",
        ":profiler_internal_cpu_coreheaders",
        ":profiler_internal_gpu_coreheaders",
        ":profiler_internal_runtime_coreheaders",
    ]
)

pkg_tar(
    name = "profiler_internal_advisor_coreheaders",
    srcs = [
        "//tensorflow/core/profiler/internal/advisor:headers",
    ],
    package_dir = "advisor",
    tags = ["manual"],
)

pkg_tar(
    name = "profiler_internal_cpu_coreheaders",
    srcs = [
        "//tensorflow/core/profiler/internal/cpu:headers",
    ],
    package_dir = "cpu",
    tags = ["manual"],
)

pkg_tar(
    name = "profiler_internal_gpu_coreheaders",
    srcs = [
        "//tensorflow/core/profiler/internal/gpu:headers",
    ],
    package_dir = "gpu",
    tags = ["manual"],
)

pkg_tar(
    name = "profiler_internal_runtime_coreheaders",
    srcs = [
        "//tensorflow/core/profiler/internal/runtime:headers",
    ],
    package_dir = "runtime",
    tags = ["manual"],
)

pkg_tar(
    name = "profiler_lib_coreheaders",
    srcs = [
        "//tensorflow/core/profiler/lib:headers",
    ],
    package_dir = "lib",
    tags = ["manual"],
)

pkg_tar(
    name = "profiler_rpc_coreheaders",
    srcs = [
        "//tensorflow/core/profiler/rpc:headers",
    ],
    package_dir = "rpc",
    tags = ["manual"],
    deps = [
        ":profiler_rpc_client_coreheaders",
    ],
)

pkg_tar(
    name = "profiler_rpc_client_coreheaders",
    srcs = [
        "//tensorflow/core/profiler/rpc/client:headers",
    ],
    package_dir = "client",
    tags = ["manual"],
)

pkg_tar(
    name = "public_coreheaders",
    srcs = [
        "//tensorflow/core:public_headers",
    ],
    package_dir = "public",
    tags = ["manual"],
)

pkg_tar(
    name = "summary_coreheaders",
    srcs = [
        "//tensorflow/core/summary:headers",
    ],
    package_dir = "summary",
    tags = ["manual"],
)

pkg_tar(
    name = "tpu_coreheaders",
    srcs = [
        "//tensorflow/core/tpu:headers",
    ],
    package_dir = "tpu",
    tags = ["manual"],
)

pkg_tar(
    name = "util_coreheaders",
    srcs = [
        "//tensorflow/core:util_headers",
    ],
    package_dir = "util",
    tags = ["manual"],
    deps = [
        ":util_ctc_coreheaders",
        ":util_proto_coreheaders",
        ":util_sparse_coreheaders",
        ":util_tensor_bundle_coreheaders",
    ],
)

pkg_tar(
    name = "util_ctc_coreheaders",
    srcs = [
        "//tensorflow/core/util/ctc:headers",
    ],
    package_dir = "ctc",
    tags = ["manual"],
)

pkg_tar(
    name = "util_proto_coreheaders",
    srcs = [
        "//tensorflow/core/util/proto:headers",
    ],
    package_dir = "proto",
    tags = ["manual"],
)

pkg_tar(
    name = "util_sparse_coreheaders",
    srcs = [
        "//tensorflow/core:util_sparse_headers",
    ],
    package_dir = "sparse",
    tags = ["manual"],
)

pkg_tar(
    name = "util_tensor_bundle_coreheaders",
    srcs = [
        "//tensorflow/core/util/tensor_bundle:headers",
    ],
    package_dir = "tensor_bundle",
    tags = ["manual"],
)

pkg_tar(
    name = "clib",
    files = select({
        "//tensorflow:windows": [
            "//tensorflow:tensorflow.dll",
            "//tensorflow:tensorflow_dll_import_lib",
        ],
        "//conditions:default": ["//tensorflow:libtensorflow.so"],
    }),
    package_dir = "lib",
    # Mark as "manual" till
    # https://github.com/bazelbuild/bazel/issues/2352
    # and https://github.com/bazelbuild/bazel/issues/1580
    # are resolved, otherwise these rules break when built
    # with Python 3.
    tags = ["manual"],
    deps = [":common_deps"],
)

pkg_tar(
    name = "cclib",
    srcs = select({
        "//tensorflow:windows": [
            "//tensorflow:tensorflow_cc.dll",
            "//tensorflow:tensorflow_cc_dll_import_lib",
        ],
        "//conditions:default": ["//tensorflow:libtensorflow_cc.so"],
    }),
    package_dir = "lib",
    # Mark as "manual" till
    # https://github.com/bazelbuild/bazel/issues/2352
    # and https://github.com/bazelbuild/bazel/issues/1580
    # are resolved, otherwise these rules break when built
    # with Python 3.
    tags = ["manual"],
    deps = [":common_deps"],
)

pkg_tar(
    name = "clicenses",
    srcs = [
        ":include/tensorflow/c/LICENSE",
        "//:LICENSE",
    ],
    package_dir = "include/tensorflow/c",
    # Mark as "manual" till
    # https://github.com/bazelbuild/bazel/issues/2352
    # and https://github.com/bazelbuild/bazel/issues/1580
    # are resolved, otherwise these rules break when built
    # with Python 3.
    tags = ["manual"],
)

pkg_tar(
    name = "cclicenses",
    srcs = [":include/tensorflow/cc/LICENSE"],
    package_dir = "include/tensorflow/cc",
    # Mark as "manual" till
    # https://github.com/bazelbuild/bazel/issues/2352
    # and https://github.com/bazelbuild/bazel/issues/1580
    # are resolved, otherwise these rules break when built
    # with Python 3.
    tags = ["manual"],
)

genrule(
    name = "clicenses_generate",
    srcs = [
        "//third_party/icu/data:LICENSE",
        "//third_party/hadoop:LICENSE.txt",
        "//third_party/eigen3:LICENSE",
        "//third_party/fft2d:LICENSE",
        "@boringssl//:LICENSE",
        "@com_googlesource_code_re2//:LICENSE",
        "@curl//:COPYING",
        "@double_conversion//:LICENSE",
        "@eigen_archive//:COPYING.MPL2",
        "@farmhash_archive//:COPYING",
        "@fft2d//:fft/readme.txt",
        "@gemmlowp//:LICENSE",
        "@gif_archive//:COPYING",
        "@highwayhash//:LICENSE",
        "@hwloc//:COPYING",
        "@icu//:icu4c/LICENSE",
        "@jpeg//:LICENSE.md",
        "@lmdb//:LICENSE",
        "@local_config_sycl//sycl:LICENSE.text",
        "@nasm//:LICENSE",
        "@nsync//:LICENSE",
        "@png_archive//:LICENSE",
        "@protobuf_archive//:LICENSE",
        "@snappy//:COPYING",
        "@zlib_archive//:zlib.h",
    ] + if_not_windows([
        "@llvm//:LICENSE.TXT",
    ]) + select({
        "//tensorflow:android": [],
        "//tensorflow:ios": [],
        "//tensorflow:linux_s390x": [],
        "//tensorflow:windows": [],
        "//tensorflow:no_aws_support": [],
        "//conditions:default": [
            "@aws//:LICENSE",
        ],
    }) + select({
        "//tensorflow:android": [],
        "//tensorflow:ios": [],
        "//tensorflow:linux_s390x": [],
        "//tensorflow:windows": [],
        "//tensorflow:no_gcp_support": [],
        "//conditions:default": [
            "@com_github_googlecloudplatform_google_cloud_cpp//:LICENSE",
        ],
    }) + select({
        "//tensorflow/core/kernels:xsmm": [
            "@libxsmm_archive//:LICENSE.md",
        ],
        "//conditions:default": [],
    }) + if_cuda([
        "@cub_archive//:LICENSE.TXT",
        "@local_config_nccl//:LICENSE",
    ]) + if_mkl([
        "//third_party/mkl:LICENSE",
        "//third_party/mkl_dnn:LICENSE",
    ]) + if_not_system_lib(
        "grpc",
        [
            "@grpc//:LICENSE",
            "@grpc//third_party/address_sorting:LICENSE",
        ],
    ),
    outs = ["include/tensorflow/c/LICENSE"],
    cmd = "$(location :concat_licenses.sh) $(SRCS) >$@",
    tools = [":concat_licenses.sh"],
)

genrule(
    name = "cclicenses_generate",
    srcs = [
        "//third_party/icu/data:LICENSE",
        "//third_party/hadoop:LICENSE.txt",
        "//third_party/eigen3:LICENSE",
        "//third_party/fft2d:LICENSE",
        "@boringssl//:LICENSE",
        "@com_googlesource_code_re2//:LICENSE",
        "@curl//:COPYING",
        "@double_conversion//:LICENSE",
        "@eigen_archive//:COPYING.MPL2",
        "@farmhash_archive//:COPYING",
        "@fft2d//:fft/readme.txt",
        "@gemmlowp//:LICENSE",
        "@gif_archive//:COPYING",
        "@highwayhash//:LICENSE",
        "@icu//:icu4c/LICENSE",
        "@jpeg//:LICENSE.md",
        "@lmdb//:LICENSE",
        "@local_config_sycl//sycl:LICENSE.text",
        "@nasm//:LICENSE",
        "@nsync//:LICENSE",
        "@png_archive//:LICENSE",
        "@protobuf_archive//:LICENSE",
        "@snappy//:COPYING",
        "@zlib_archive//:zlib.h",
    ] + if_not_windows([
        "@llvm//:LICENSE.TXT",
    ]) + select({
        "//tensorflow:android": [],
        "//tensorflow:ios": [],
        "//tensorflow:linux_s390x": [],
        "//tensorflow:windows": [],
        "//tensorflow:no_aws_support": [],
        "//conditions:default": [
            "@aws//:LICENSE",
        ],
    }) + select({
        "//tensorflow:android": [],
        "//tensorflow:ios": [],
        "//tensorflow:linux_s390x": [],
        "//tensorflow:windows": [],
        "//tensorflow:no_gcp_support": [],
        "//conditions:default": [
            "@com_github_googlecloudplatform_google_cloud_cpp//:LICENSE",
        ],
    }) + select({
        "//tensorflow/core/kernels:xsmm": [
            "@libxsmm_archive//:LICENSE.md",
        ],
        "//conditions:default": [],
    }) + if_cuda([
        "@cub_archive//:LICENSE.TXT",
        "@local_config_nccl//:LICENSE",
    ]) + if_mkl([
        "//third_party/mkl:LICENSE",
        "//third_party/mkl_dnn:LICENSE",
    ]) + if_not_system_lib(
        "grpc",
        [
            "@grpc//:LICENSE",
            "@grpc//third_party/address_sorting:LICENSE",
        ],
    ),
    outs = ["include/tensorflow/cc/LICENSE"],
    cmd = "$(location :concat_licenses.sh) $(SRCS) >$@",
    tools = [":concat_licenses.sh"],
)

genrule(
    name = "jnilicenses_generate",
    srcs = [
        "//third_party/icu/data:LICENSE",
        "//third_party/hadoop:LICENSE.txt",
        "//third_party/eigen3:LICENSE",
        "//third_party/fft2d:LICENSE",
        "@boringssl//:LICENSE",
        "@com_googlesource_code_re2//:LICENSE",
        "@curl//:COPYING",
        "@double_conversion//:LICENSE",
        "@eigen_archive//:COPYING.MPL2",
        "@farmhash_archive//:COPYING",
        "@fft2d//:fft/readme.txt",
        "@gemmlowp//:LICENSE",
        "@gif_archive//:COPYING",
        "@highwayhash//:LICENSE",
        "@hwloc//:COPYING",
        "@icu//:icu4j/main/shared/licenses/LICENSE",
        "@jpeg//:LICENSE.md",
        "@lmdb//:LICENSE",
        "@local_config_sycl//sycl:LICENSE.text",
        "@nasm//:LICENSE",
        "@nsync//:LICENSE",
        "@png_archive//:LICENSE",
        "@protobuf_archive//:LICENSE",
        "@snappy//:COPYING",
        "@zlib_archive//:zlib.h",
        "@grpc//:LICENSE",
        "@grpc//third_party/address_sorting:LICENSE",
    ] + if_not_windows([
        "@llvm//:LICENSE.TXT",
    ]) + select({
        "//tensorflow:android": [],
        "//tensorflow:ios": [],
        "//tensorflow:linux_s390x": [],
        "//tensorflow:windows": [],
        "//tensorflow:no_aws_support": [],
        "//conditions:default": [
            "@aws//:LICENSE",
        ],
    }) + select({
        "//tensorflow:android": [],
        "//tensorflow:ios": [],
        "//tensorflow:linux_s390x": [],
        "//tensorflow:windows": [],
        "//tensorflow:no_gcp_support": [],
        "//conditions:default": [
            "@com_github_googlecloudplatform_google_cloud_cpp//:LICENSE",
        ],
    }) + select({
        "//tensorflow/core/kernels:xsmm": [
            "@libxsmm_archive//:LICENSE.md",
        ],
        "//conditions:default": [],
    }) + if_cuda([
        "@cub_archive//:LICENSE.TXT",
        "@local_config_nccl//:LICENSE",
    ]) + if_mkl([
        "//third_party/mkl:LICENSE",
        "//third_party/mkl_dnn:LICENSE",
    ]),
    outs = ["include/tensorflow/jni/LICENSE"],
    cmd = "$(location :concat_licenses.sh) $(SRCS) >$@",
    tools = [":concat_licenses.sh"],
)

sh_test(
    name = "libtensorflow_test",
    size = "small",
    srcs = ["libtensorflow_test.sh"],
    data = [
        "libtensorflow_test.c",
        ":libtensorflow.tar.gz",
    ],
    # Mark as "manual" till
    # https://github.com/bazelbuild/bazel/issues/2352
    # and https://github.com/bazelbuild/bazel/issues/1580
    # are resolved, otherwise these rules break when built
    # with Python 3.
    # Till then, this test is explicitly executed when building
    # the release by tensorflow/tools/ci_build/builds/libtensorflow.sh
    tags = ["manual"],
)

sh_test(
    name = "libtensorflow_java_test",
    size = "small",
    srcs = ["libtensorflow_java_test.sh"],
    data = [
        ":LibTensorFlowTest.java",
        ":libtensorflow_jni.tar.gz",
        "//tensorflow/java:libtensorflow.jar",
    ],
    # Mark as "manual" till
    # https://github.com/bazelbuild/bazel/issues/2352
    # and https://github.com/bazelbuild/bazel/issues/1580
    # are resolved, otherwise these rules break when built
    # with Python 3.
    # Till then, this test is explicitly executed when building
    # the release by tensorflow/tools/ci_build/builds/libtensorflow.sh
    tags = ["manual"],
)
