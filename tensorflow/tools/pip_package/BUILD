# Description:
#  Tools for building the TensorFlow pip package.

package(default_visibility = ["//visibility:private"])

load(
    "//tensorflow:tensorflow.bzl",
    "if_not_windows",
    "if_windows",
    "transitive_hdrs",
)
load("//third_party/mkl:build_defs.bzl", "if_mkl")
load("//tensorflow:tensorflow.bzl", "if_cuda")
load("@local_config_syslibs//:build_defs.bzl", "if_not_system_lib")
load("//tensorflow/core:platform/default/build_config_root.bzl", "tf_additional_license_deps")

# This returns a list of headers of all public header libraries (e.g.,
# framework, lib), and all of the transitive dependencies of those
# public headers.  Not all of the headers returned by the filegroup
# are public (e.g., internal headers that are included by public
# headers), but the internal headers need to be packaged in the
# pip_package for the public headers to be properly included.
#
# Public headers are therefore defined by those that are both:
#
# 1) "publicly visible" as defined by bazel
# 2) Have documentation.
#
# This matches the policy of "public" for our python API.
transitive_hdrs(
    name = "included_headers",
    deps = [
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:stream_executor",
        "//third_party/eigen3",
    ] + if_cuda([
        "@local_config_cuda//cuda:cuda_headers",
    ]),
)

py_binary(
    name = "simple_console",
    srcs = ["simple_console.py"],
    srcs_version = "PY2AND3",
    deps = ["//tensorflow:tensorflow_py"],
)

COMMON_PIP_DEPS = [
    ":licenses",
    "MANIFEST.in",
    "README",
    "setup.py",
    ":included_headers",
    "//tensorflow:tensorflow_py",
    "//tensorflow/contrib/autograph:autograph",
    "//tensorflow/contrib/autograph/converters:converters",
    "//tensorflow/contrib/autograph/core:core",
    "//tensorflow/contrib/autograph/core:test_lib",
    "//tensorflow/contrib/autograph/impl:impl",
    "//tensorflow/contrib/autograph/lang:lang",
    "//tensorflow/contrib/autograph/operators:operators",
    "//tensorflow/contrib/autograph/pyct:pyct",
    "//tensorflow/contrib/autograph/pyct/static_analysis:static_analysis",
    "//tensorflow/contrib/autograph/pyct/common_transformers:common_transformers",
    "//tensorflow/contrib/boosted_trees:boosted_trees_pip",
    "//tensorflow/contrib/cluster_resolver:cluster_resolver_pip",
    "//tensorflow/contrib/constrained_optimization:constrained_optimization_pip",
    "//tensorflow/contrib/data/python/kernel_tests/serialization:dataset_serialization_test_base",
    "//tensorflow/contrib/data/python/ops:contrib_op_loader",
    "//tensorflow/contrib/eager/python/examples:examples_pip",
    "//tensorflow/contrib/eager/python:evaluator",
    "//tensorflow/contrib/gan:gan",
    "//tensorflow/contrib/graph_editor:graph_editor_pip",
    "//tensorflow/contrib/keras:keras",
    "//tensorflow/contrib/labeled_tensor:labeled_tensor_pip",
    "//tensorflow/contrib/nn:nn_py",
    "//tensorflow/contrib/predictor:predictor_pip",
    "//tensorflow/contrib/proto:proto_pip",
    "//tensorflow/contrib/receptive_field:receptive_field_pip",
    "//tensorflow/contrib/rpc:rpc_pip",
    "//tensorflow/contrib/session_bundle:session_bundle_pip",
    "//tensorflow/contrib/signal:signal_py",
    "//tensorflow/contrib/signal:test_util",
    "//tensorflow/contrib/slim:slim",
    "//tensorflow/contrib/slim/python/slim/data:data_pip",
    "//tensorflow/contrib/slim/python/slim/nets:nets_pip",
    "//tensorflow/contrib/specs:specs",
    "//tensorflow/contrib/summary:summary_test_util",
    "//tensorflow/contrib/tensor_forest:init_py",
    "//tensorflow/contrib/tensor_forest/hybrid:hybrid_pip",
    "//tensorflow/contrib/timeseries:timeseries_pip",
    "//tensorflow/contrib/tpu",
    "//tensorflow/examples/tutorials/mnist:package",
    "//tensorflow/python:cond_v2",
    "//tensorflow/python:distributed_framework_test_lib",
    "//tensorflow/python:meta_graph_testdata",
    "//tensorflow/python:spectral_ops_test_util",
    "//tensorflow/python:util_example_parser_configuration",
    "//tensorflow/python/debug:debug_pip",
    "//tensorflow/python/eager:eager_pip",
    "//tensorflow/python/kernel_tests/testdata:self_adjoint_eig_op_test_files",
    "//tensorflow/python/saved_model:saved_model",
    "//tensorflow/python/tools:tools_pip",
    "//tensorflow/python/tools/api/generator:create_python_api",
    "//tensorflow/python:test_ops",
    "//tensorflow/tools/dist_test/server:grpc_tensorflow_server",
]

# On Windows, python binary is a zip file of runfiles tree.
# Add everything to its data dependency for generating a runfiles tree
# for building the pip package on Windows.
py_binary(
    name = "simple_console_for_windows",
    srcs = ["simple_console_for_windows.py"],
    data = COMMON_PIP_DEPS,
    srcs_version = "PY2AND3",
    deps = ["//tensorflow:tensorflow_py"],
)

filegroup(
    name = "licenses",
    data = [
        "//third_party/eigen3:LICENSE",
        "//third_party/fft2d:LICENSE",
        "//third_party/hadoop:LICENSE.txt",
        "@absl_py//absl/flags:LICENSE",
        "@arm_neon_2_x86_sse//:LICENSE",
        "@astor_archive//:LICENSE",
        "@aws//:LICENSE",
        "@boringssl//:LICENSE",
        "@com_github_googleapis_googleapis//:LICENSE",
        "@com_github_googlecloudplatform_google_cloud_cpp//:LICENSE",
        "@com_google_absl//:LICENSE",
        "@com_googlesource_code_re2//:LICENSE",
        "@cub_archive//:LICENSE.TXT",
        "@curl//:COPYING",
        "@double_conversion//:LICENSE",
        "@eigen_archive//:COPYING.MPL2",
        "@farmhash_archive//:COPYING",
        "@fft2d//:fft/readme.txt",
        "@flatbuffers//:LICENSE.txt",
        "@gast_archive//:PKG-INFO",
        "@gemmlowp//:LICENSE",
        "@gif_archive//:COPYING",
        "@highwayhash//:LICENSE",
        "@jemalloc//:COPYING",
        "@jpeg//:LICENSE.md",
        "@kafka//:LICENSE",
        "@libxsmm_archive//:LICENSE.md",
        "@lmdb//:LICENSE",
        "@local_config_nccl//:LICENSE",
        "@local_config_sycl//sycl:LICENSE.text",
        "@nasm//:LICENSE",
        "@nsync//:LICENSE",
        "@pcre//:LICENCE",
        "@png_archive//:LICENSE",
        "@protobuf_archive//:LICENSE",
        "@six_archive//:LICENSE",
        "@snappy//:COPYING",
        "@swig//:LICENSE",
        "@termcolor_archive//:COPYING.txt",
        "@zlib_archive//:zlib.h",
        "@org_python_pypi_backports_weakref//:LICENSE",
    ] + if_mkl([
        "//third_party/mkl:LICENSE",
        "//third_party/mkl_dnn:LICENSE",
    ]) + if_not_system_lib(
        "grpc",
        [
            "@grpc//:LICENSE",
            "@grpc//third_party/nanopb:LICENSE.txt",
            "@grpc//third_party/address_sorting:LICENSE",
        ],
    ) + tf_additional_license_deps(),
)

sh_binary(
    name = "build_pip_package",
    srcs = ["build_pip_package.sh"],
    data = select({
        "//tensorflow:windows": [":simple_console_for_windows"],
        "//tensorflow:windows_msvc": [":simple_console_for_windows"],
        "//conditions:default": COMMON_PIP_DEPS + [
            ":simple_console",
            "//tensorflow/contrib/lite/python:interpreter_test_data",
            "//tensorflow/contrib/lite/python:tflite_convert",
            "//tensorflow/contrib/lite/toco/python:toco_from_protos",
        ],
    }) + if_mkl(["//third_party/mkl:intel_binary_blob"]),
)

# A genrule for generating a marker file for the pip package on Windows
#
# This only works on Windows, because :simple_console_for_windows is a
# python zip file containing everything we need for building the pip package.
# However, on other platforms, due to https://github.com/bazelbuild/bazel/issues/4223,
# when C++ extensions change, this generule doesn't rebuild.
genrule(
    name = "win_pip_package_marker",
    srcs = if_windows([
        ":build_pip_package",
        ":simple_console_for_windows",
    ]),
    outs = ["win_pip_package_marker_file"],
    cmd = select({
        "//conditions:default": "touch $@",
        "//tensorflow:windows": "md5sum $(locations :build_pip_package) $(locations :simple_console_for_windows) > $@",
    }),
    visibility = ["//visibility:public"],
)
