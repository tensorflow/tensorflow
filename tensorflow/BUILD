# Description:
# TensorFlow is a computational framework, primarily for use in machine
# learning applications.

package(default_visibility = [":internal"])

licenses(["notice"])  # Apache 2.0

exports_files([
    "LICENSE",
    "ACKNOWLEDGMENTS",
    # The leakr files are used by //third_party/cloud_tpu.
    "leakr_badwords.dic",
    "leakr_badfiles.dic",
])

load("//tensorflow:tensorflow.bzl", "tf_cc_shared_object")
load(
    "//tensorflow/core:platform/default/build_config.bzl",
    "tf_additional_binary_deps",
)

# Config setting for determining if we are building for Android.
config_setting(
    name = "android",
    values = {"crosstool_top": "//external:android/crosstool"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "android_x86",
    values = {
        "crosstool_top": "//external:android/crosstool",
        "cpu": "x86",
    },
    visibility = ["//visibility:public"],
)

config_setting(
    name = "android_x86_64",
    values = {
        "crosstool_top": "//external:android/crosstool",
        "cpu": "x86_64",
    },
    visibility = ["//visibility:public"],
)

config_setting(
    name = "android_armeabi",
    values = {
        "crosstool_top": "//external:android/crosstool",
        "cpu": "armeabi",
    },
    visibility = ["//visibility:public"],
)

config_setting(
    name = "raspberry_pi_armeabi",
    values = {
        "crosstool_top": "@local_config_arm_compiler//:toolchain",
        "cpu": "armeabi",
    },
    visibility = ["//visibility:public"],
)

config_setting(
    name = "android_arm",
    values = {
        "crosstool_top": "//external:android/crosstool",
        "cpu": "armeabi-v7a",
    },
    visibility = ["//visibility:public"],
)

config_setting(
    name = "android_arm64",
    values = {
        "crosstool_top": "//external:android/crosstool",
        "cpu": "arm64-v8a",
    },
    visibility = ["//visibility:public"],
)

config_setting(
    name = "android_mips",
    values = {
        "crosstool_top": "//external:android/crosstool",
        "cpu": "mips",
    },
    visibility = ["//visibility:public"],
)

config_setting(
    name = "android_mips64",
    values = {
        "crosstool_top": "//external:android/crosstool",
        "cpu": "mips64",
    },
    visibility = ["//visibility:public"],
)

config_setting(
    name = "darwin",
    values = {"cpu": "darwin"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "windows",
    values = {"cpu": "x64_windows"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "windows_msvc",
    values = {"cpu": "x64_windows_msvc"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "no_tensorflow_py_deps",
    define_values = {"no_tensorflow_py_deps": "true"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "ios",
    values = {"crosstool_top": "//tools/osx/crosstool:crosstool"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "ios_x86_64",
    values = {
        "crosstool_top": "//tools/osx/crosstool:crosstool",
        "cpu": "ios_x86_64",
    },
    visibility = ["//visibility:public"],
)

config_setting(
    name = "linux_x86_64",
    values = {"cpu": "k8"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "linux_ppc64le",
    values = {"cpu": "ppc"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "debug",
    values = {
        "compilation_mode": "dbg",
    },
    visibility = ["//visibility:public"],
)

config_setting(
    name = "optimized",
    values = {
        "compilation_mode": "opt",
    },
    visibility = ["//visibility:public"],
)

config_setting(
    name = "freebsd",
    values = {"cpu": "freebsd"},
    visibility = ["//visibility:public"],
)

# TODO(jhseu): Enable on other platforms other than Linux.
config_setting(
    name = "with_jemalloc_linux_x86_64",
    define_values = {"with_jemalloc": "true"},
    values = {"cpu": "k8"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "with_jemalloc_linux_ppc64le",
    define_values = {"with_jemalloc": "true"},
    values = {"cpu": "ppc"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "with_default_optimizations",
    define_values = {"with_default_optimizations": "true"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "with_gcp_support",
    define_values = {"with_gcp_support": "true"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "with_hdfs_support",
    define_values = {"with_hdfs_support": "true"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "with_s3_support",
    define_values = {"with_s3_support": "true"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "with_kafka_support",
    define_values = {"with_kafka_support": "true"},
    visibility = ["//visibility:public"],
)

# Crosses between platforms and file system libraries not supported on those
# platforms due to limitations in nested select() statements.
config_setting(
    name = "with_gcp_support_windows_override",
    define_values = {"with_gcp_support": "true"},
    values = {"cpu": "x64_windows"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "with_hdfs_support_windows_override",
    define_values = {"with_hdfs_support": "true"},
    values = {"cpu": "x64_windows"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "with_s3_support_windows_override",
    define_values = {"with_s3_support": "true"},
    values = {"cpu": "x64_windows"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "with_gcp_support_android_override",
    define_values = {"with_gcp_support": "true"},
    values = {"crosstool_top": "//external:android/crosstool"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "with_hdfs_support_android_override",
    define_values = {"with_hdfs_support": "true"},
    values = {"crosstool_top": "//external:android/crosstool"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "with_s3_support_android_override",
    define_values = {"with_s3_support": "true"},
    values = {"crosstool_top": "//external:android/crosstool"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "with_gcp_support_ios_override",
    define_values = {"with_gcp_support": "true"},
    values = {"crosstool_top": "//tools/osx/crosstool:crosstool"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "with_hdfs_support_ios_override",
    define_values = {"with_hdfs_support": "true"},
    values = {"crosstool_top": "//tools/osx/crosstool:crosstool"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "with_s3_support_ios_override",
    define_values = {"with_s3_support": "true"},
    values = {"crosstool_top": "//tools/osx/crosstool:crosstool"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "with_xla_support",
    define_values = {"with_xla_support": "true"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "with_gdr_support",
    define_values = {"with_gdr_support": "true"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "with_verbs_support",
    define_values = {"with_verbs_support": "true"},
    visibility = ["//visibility:public"],
)

# Crosses between framework_shared_object and a bunch of other configurations
# due to limitations in nested select() statements.
config_setting(
    name = "framework_shared_object",
    define_values = {
        "framework_shared_object": "true",
    },
    visibility = ["//visibility:public"],
)

config_setting(
    name = "with_jemalloc_linux_x86_64_dynamic",
    define_values = {
        "with_jemalloc": "true",
        "framework_shared_object": "true",
    },
    values = {
        "cpu": "k8",
    },
    visibility = ["//visibility:public"],
)

config_setting(
    name = "with_jemalloc_linux_ppc64le_dynamic",
    define_values = {
        "with_jemalloc": "true",
        "framework_shared_object": "true",
    },
    values = {
        "cpu": "ppc",
    },
    visibility = ["//visibility:public"],
)

config_setting(
    name = "using_cuda_clang",
    define_values = {
        "using_cuda_clang": "true",
    },
)

config_setting(
    name = "using_cuda_clang_with_dynamic_build",
    define_values = {
        "using_cuda_clang": "true",
        "framework_shared_object": "true",
    },
)

config_setting(
    name = "using_cuda_nvcc",
    define_values = {
        "using_cuda_nvcc": "true",
    },
)

config_setting(
    name = "using_cuda_nvcc_with_dynamic_build",
    define_values = {
        "using_cuda_nvcc": "true",
        "framework_shared_object": "true",
    },
)

config_setting(
    name = "with_mpi_support",
    values = {"define": "with_mpi_support=true"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "override_eigen_strong_inline",
    values = {"define": "override_eigen_strong_inline=true"},
    visibility = ["//visibility:public"],
)

# TODO(laigd): consider removing this option and make TensorRT enabled
# automatically when CUDA is enabled.
config_setting(
    name = "with_tensorrt_support",
    values = {"define": "with_tensorrt_support=true"},
    visibility = ["//visibility:public"],
)

package_group(
    name = "internal",
    packages = [
        "//learning/meta_rank/...",
        "//tensorflow/...",
        "//tensorflow_fold/llgtm/...",
        "//third_party/py/tensor2tensor/...",
    ],
)

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
            "g3doc/sitemap.md",
        ],
    ),
    visibility = ["//tensorflow:__subpackages__"],
)

py_library(
    name = "tensorflow_py",
    srcs = ["__init__.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = ["//tensorflow/python"],
)

filegroup(
    name = "all_opensource_files",
    data = [
        ":all_files",
        "//tensorflow/c:all_files",
        "//tensorflow/cc:all_files",
        "//tensorflow/cc/saved_model:all_files",
        "//tensorflow/cc/saved_model/python:all_files",
        "//tensorflow/cc/tools:all_files",
        "//tensorflow/compiler/aot:all_files",
        "//tensorflow/compiler/aot/tests:all_files",
        "//tensorflow/compiler/jit:all_files",
        "//tensorflow/compiler/jit/graphcycles:all_files",
        "//tensorflow/compiler/jit/kernels:all_files",
        "//tensorflow/compiler/jit/legacy_flags:all_files",
        "//tensorflow/compiler/jit/ops:all_files",
        "//tensorflow/compiler/plugin:all_files",
        "//tensorflow/compiler/tests:all_files",
        "//tensorflow/compiler/tf2xla:all_files",
        "//tensorflow/compiler/tf2xla/cc:all_files",
        "//tensorflow/compiler/tf2xla/kernels:all_files",
        "//tensorflow/compiler/tf2xla/lib:all_files",
        "//tensorflow/compiler/tf2xla/ops:all_files",
        "//tensorflow/compiler/xla:all_files",
        "//tensorflow/compiler/xla/client:all_files",
        "//tensorflow/compiler/xla/client/lib:all_files",
        "//tensorflow/compiler/xla/legacy_flags:all_files",
        "//tensorflow/compiler/xla/python:all_files",
        "//tensorflow/compiler/xla/service:all_files",
        "//tensorflow/compiler/xla/service/cpu:all_files",
        "//tensorflow/compiler/xla/service/gpu:all_files",
        "//tensorflow/compiler/xla/service/gpu/llvm_gpu_backend:all_files",
        "//tensorflow/compiler/xla/service/interpreter:all_files",
        "//tensorflow/compiler/xla/service/llvm_ir:all_files",
        "//tensorflow/compiler/xla/tests:all_files",
        "//tensorflow/compiler/xla/tools:all_files",
        "//tensorflow/compiler/xla/tools/parser:all_files",
        "//tensorflow/contrib:all_files",
        "//tensorflow/contrib/all_reduce:all_files",
        "//tensorflow/contrib/android:all_files",
        "//tensorflow/contrib/batching:all_files",
        "//tensorflow/contrib/bayesflow:all_files",
        "//tensorflow/contrib/boosted_trees:all_files",
        "//tensorflow/contrib/boosted_trees/estimator_batch:all_files",
        "//tensorflow/contrib/boosted_trees/lib:all_files",
        "//tensorflow/contrib/boosted_trees/proto:all_files",
        "//tensorflow/contrib/boosted_trees/resources:all_files",
        "//tensorflow/contrib/cloud:all_files",
        "//tensorflow/contrib/cloud/kernels:all_files",
        "//tensorflow/contrib/cluster_resolver:all_files",
        "//tensorflow/contrib/coder:all_files",
        "//tensorflow/contrib/compiler:all_files",
        "//tensorflow/contrib/copy_graph:all_files",
        "//tensorflow/contrib/crf:all_files",
        "//tensorflow/contrib/cudnn_rnn:all_files",
        "//tensorflow/contrib/data:all_files",
        "//tensorflow/contrib/data/kernels:all_files",
        "//tensorflow/contrib/data/python/kernel_tests:all_files",
        "//tensorflow/contrib/data/python/ops:all_files",
        "//tensorflow/contrib/decision_trees/proto:all_files",
        "//tensorflow/contrib/deprecated:all_files",
        "//tensorflow/contrib/distributions:all_files",
        "//tensorflow/contrib/eager/proto:all_files",
        "//tensorflow/contrib/eager/python:all_files",
        "//tensorflow/contrib/estimator:all_files",
        "//tensorflow/contrib/factorization:all_files",
        "//tensorflow/contrib/factorization/examples:all_files",
        "//tensorflow/contrib/factorization/kernels:all_files",
        "//tensorflow/contrib/ffmpeg:all_files",
        "//tensorflow/contrib/ffmpeg/default:all_files",
        "//tensorflow/contrib/framework:all_files",
        "//tensorflow/contrib/fused_conv:all_files",
        "//tensorflow/contrib/gan:all_files",
        "//tensorflow/contrib/gdr:all_files",
        "//tensorflow/contrib/graph_editor:all_files",
        "//tensorflow/contrib/grid_rnn:all_files",
        "//tensorflow/contrib/hooks:all_files",
        "//tensorflow/contrib/hvx/clock_cycle_profiling:all_files",
        "//tensorflow/contrib/hvx/hvx_ops_support_checker:all_files",
        "//tensorflow/contrib/image:all_files",
        "//tensorflow/contrib/input_pipeline:all_files",
        "//tensorflow/contrib/input_pipeline/kernels:all_files",
        "//tensorflow/contrib/integrate:all_files",
        "//tensorflow/contrib/keras:all_files",
        "//tensorflow/contrib/kernel_methods:all_files",
        "//tensorflow/contrib/kfac:all_files",
        "//tensorflow/contrib/kfac/examples:all_files",
        "//tensorflow/contrib/kfac/examples/tests:all_files",
        "//tensorflow/contrib/kfac/python/kernel_tests:all_files",
        "//tensorflow/contrib/kfac/python/ops:all_files",
        "//tensorflow/contrib/labeled_tensor:all_files",
        "//tensorflow/contrib/layers:all_files",
        "//tensorflow/contrib/layers/kernels:all_files",
        "//tensorflow/contrib/learn:all_files",
        "//tensorflow/contrib/learn/python/learn/datasets:all_files",
        "//tensorflow/contrib/legacy_seq2seq:all_files",
        "//tensorflow/contrib/libsvm:all_files",
        "//tensorflow/contrib/linalg:all_files",
        "//tensorflow/contrib/linear_optimizer:all_files",
        "//tensorflow/contrib/lite:all_files",
        "//tensorflow/contrib/lite/java:all_files",
        "//tensorflow/contrib/lite/java/demo/app/src/main:all_files",
        "//tensorflow/contrib/lite/java/demo/app/src/main/assets:all_files",
        "//tensorflow/contrib/lite/java/src/main/native:all_files",
        "//tensorflow/contrib/lite/java/src/testhelper/java/org/tensorflow/lite:all_files",
        "//tensorflow/contrib/lite/kernels:all_files",
        "//tensorflow/contrib/lite/kernels/internal:all_files",
        "//tensorflow/contrib/lite/models/smartreply:all_files",
        "//tensorflow/contrib/lite/nnapi:all_files",
        "//tensorflow/contrib/lite/python:all_files",
        "//tensorflow/contrib/lite/schema:all_files",
        "//tensorflow/contrib/lite/testing:all_files",
        "//tensorflow/contrib/lite/toco:all_files",
        "//tensorflow/contrib/lite/toco/graph_transformations/tests:all_files",
        "//tensorflow/contrib/lite/toco/python:all_files",
        "//tensorflow/contrib/lite/toco/tensorflow_graph_matching:all_files",
        "//tensorflow/contrib/lite/toco/tflite:all_files",
        "//tensorflow/contrib/lite/tools:all_files",
        "//tensorflow/contrib/lookup:all_files",
        "//tensorflow/contrib/losses:all_files",
        "//tensorflow/contrib/makefile:all_files",
        "//tensorflow/contrib/memory_stats:all_files",
        "//tensorflow/contrib/meta_graph_transform:all_files",
        "//tensorflow/contrib/metrics:all_files",
        "//tensorflow/contrib/model_pruning:all_files",
        "//tensorflow/contrib/model_pruning/examples/cifar10:all_files",
        "//tensorflow/contrib/nccl:all_files",
        "//tensorflow/contrib/ndlstm:all_files",
        "//tensorflow/contrib/nearest_neighbor:all_files",
        "//tensorflow/contrib/nn:all_files",
        "//tensorflow/contrib/opt:all_files",
        "//tensorflow/contrib/periodic_resample:all_files",
        "//tensorflow/contrib/predictor:all_files",
        "//tensorflow/contrib/py2tf:all_files",
        "//tensorflow/contrib/py2tf/converters:all_files",
        "//tensorflow/contrib/py2tf/impl:all_files",
        "//tensorflow/contrib/py2tf/pyct:all_files",
        "//tensorflow/contrib/py2tf/pyct/static_analysis:all_files",
        "//tensorflow/contrib/quantize:all_files",
        "//tensorflow/contrib/receptive_field:all_files",
        "//tensorflow/contrib/reduce_slice_ops:all_files",
        "//tensorflow/contrib/remote_fused_graph/pylib:all_files",
        "//tensorflow/contrib/resampler:all_files",
        "//tensorflow/contrib/rnn:all_files",
        "//tensorflow/contrib/saved_model:all_files",
        "//tensorflow/contrib/saved_model/cc/saved_model:all_files",
        "//tensorflow/contrib/seq2seq:all_files",
        "//tensorflow/contrib/session_bundle:all_files",
        "//tensorflow/contrib/session_bundle/example:all_files",
        "//tensorflow/contrib/signal:all_files",
        "//tensorflow/contrib/slim:all_files",
        "//tensorflow/contrib/slim/python/slim/data:all_files",
        "//tensorflow/contrib/slim/python/slim/nets:all_files",
        "//tensorflow/contrib/solvers:all_files",
        "//tensorflow/contrib/sparsemax:all_files",
        "//tensorflow/contrib/specs:all_files",
        "//tensorflow/contrib/staging:all_files",
        "//tensorflow/contrib/stat_summarizer:all_files",
        "//tensorflow/contrib/stateless:all_files",
        "//tensorflow/contrib/summary:all_files",
        "//tensorflow/contrib/tensor_forest:all_files",
        "//tensorflow/contrib/tensor_forest/hybrid:all_files",
        "//tensorflow/contrib/tensor_forest/kernels/v4:all_files",
        "//tensorflow/contrib/tensor_forest/proto:all_files",
        "//tensorflow/contrib/tensorboard:all_files",
        "//tensorflow/contrib/tensorboard/db:all_files",
        "//tensorflow/contrib/tensorrt:all_files",
        "//tensorflow/contrib/testing:all_files",
        "//tensorflow/contrib/text:all_files",
        "//tensorflow/contrib/tfprof:all_files",
        "//tensorflow/contrib/timeseries:all_files",
        "//tensorflow/contrib/timeseries/examples:all_files",
        "//tensorflow/contrib/timeseries/python/timeseries:all_files",
        "//tensorflow/contrib/timeseries/python/timeseries/state_space_models:all_files",
        "//tensorflow/contrib/tpu:all_files",
        "//tensorflow/contrib/tpu/profiler:all_files",
        "//tensorflow/contrib/tpu/proto:all_files",
        "//tensorflow/contrib/training:all_files",
        "//tensorflow/contrib/util:all_files",
        "//tensorflow/contrib/verbs:all_files",
        "//tensorflow/core:all_files",
        "//tensorflow/core/api_def:all_files",
        "//tensorflow/core/debug:all_files",
        "//tensorflow/core/distributed_runtime:all_files",
        "//tensorflow/core/distributed_runtime/rpc:all_files",
        "//tensorflow/core/grappler:all_files",
        "//tensorflow/core/grappler/clusters:all_files",
        "//tensorflow/core/grappler/costs:all_files",
        "//tensorflow/core/grappler/inputs:all_files",
        "//tensorflow/core/grappler/optimizers:all_files",
        "//tensorflow/core/grappler/utils:all_files",
        "//tensorflow/core/kernels:all_files",
        "//tensorflow/core/kernels/batching_util:all_files",
        "//tensorflow/core/kernels/data:all_files",
        "//tensorflow/core/kernels/data/sql:all_files",
        "//tensorflow/core/kernels/fuzzing:all_files",
        "//tensorflow/core/kernels/hexagon:all_files",
        "//tensorflow/core/kernels/neon:all_files",
        "//tensorflow/core/lib/db:all_files",
        "//tensorflow/core/ops/compat:all_files",
        "//tensorflow/core/platform/cloud:all_files",
        "//tensorflow/core/platform/default/build_config:all_files",
        "//tensorflow/core/platform/hadoop:all_files",
        "//tensorflow/core/platform/s3:all_files",
        "//tensorflow/core/profiler:all_files",
        "//tensorflow/core/profiler/internal:all_files",
        "//tensorflow/core/profiler/internal/advisor:all_files",
        "//tensorflow/core/util/ctc:all_files",
        "//tensorflow/core/util/tensor_bundle:all_files",
        "//tensorflow/examples/adding_an_op:all_files",
        "//tensorflow/examples/android:all_files",
        "//tensorflow/examples/benchmark:all_files",
        "//tensorflow/examples/get_started/regression:all_files",
        "//tensorflow/examples/how_tos/reading_data:all_files",
        "//tensorflow/examples/image_retraining:all_files",
        "//tensorflow/examples/label_image:all_files",
        "//tensorflow/examples/learn:all_files",
        "//tensorflow/examples/multibox_detector:all_files",
        "//tensorflow/examples/saved_model:all_files",
        "//tensorflow/examples/speech_commands:all_files",
        "//tensorflow/examples/tutorials/estimators:all_files",
        "//tensorflow/examples/tutorials/layers:all_files",
        "//tensorflow/examples/tutorials/mnist:all_files",
        "//tensorflow/examples/tutorials/monitors:all_files",
        "//tensorflow/examples/tutorials/word2vec:all_files",
        "//tensorflow/examples/wav_to_spectrogram:all_files",
        "//tensorflow/go:all_files",
        "//tensorflow/java:all_files",
        "//tensorflow/java/src/main/java/org/tensorflow/examples:all_files",
        "//tensorflow/java/src/main/native:all_files",
        "//tensorflow/python:all_files",
        "//tensorflow/python/data:all_files",
        "//tensorflow/python/data/kernel_tests:all_files",
        "//tensorflow/python/data/ops:all_files",
        "//tensorflow/python/data/util:all_files",
        "//tensorflow/python/debug:all_files",
        "//tensorflow/python/eager:all_files",
        "//tensorflow/python/estimator:all_files",
        "//tensorflow/python/feature_column:all_files",
        "//tensorflow/python/keras:all_files",
        "//tensorflow/python/kernel_tests:all_files",
        "//tensorflow/python/kernel_tests/distributions:all_files",
        "//tensorflow/python/kernel_tests/linalg:all_files",
        "//tensorflow/python/kernel_tests/random:all_files",
        "//tensorflow/python/ops/distributions:all_files",
        "//tensorflow/python/ops/linalg:all_files",
        "//tensorflow/python/ops/losses:all_files",
        "//tensorflow/python/profiler:all_files",
        "//tensorflow/python/profiler/internal:all_files",
        "//tensorflow/python/saved_model:all_files",
        "//tensorflow/python/tools:all_files",
        "//tensorflow/tools/api/generator:all_files",
        "//tensorflow/tools/api/golden:all_files",
        "//tensorflow/tools/api/lib:all_files",
        "//tensorflow/tools/api/tests:all_files",
        "//tensorflow/tools/benchmark:all_files",
        "//tensorflow/tools/build_info:all_files",
        "//tensorflow/tools/ci_build/gpu_build:all_files",
        "//tensorflow/tools/common:all_files",
        "//tensorflow/tools/compatibility:all_files",
        "//tensorflow/tools/dist_test/server:all_files",
        "//tensorflow/tools/docker:all_files",
        "//tensorflow/tools/docker/notebooks:all_files",
        "//tensorflow/tools/docs:all_files",
        "//tensorflow/tools/git:all_files",
        "//tensorflow/tools/graph_transforms:all_files",
        "//tensorflow/tools/mlpbtxt:all_files",
        "//tensorflow/tools/proto_text:all_files",
        "//tensorflow/tools/quantization:all_files",
        "//tensorflow/tools/test:all_files",
        "//tensorflow/user_ops:all_files",
        "//third_party/eigen3:all_files",
        "//third_party/fft2d:all_files",
        "//third_party/flatbuffers:all_files",
        "//third_party/hadoop:all_files",
        "//third_party/sycl:all_files",
        "//third_party/sycl/sycl:all_files",
    ],
    visibility = ["//visibility:public"],
)

load(
    "//third_party/mkl:build_defs.bzl",
    "if_mkl",
)

filegroup(
    name = "intel_binary_blob",
    data = if_mkl(
        [
            "//third_party/mkl:intel_binary_blob",
        ],
    ),
)

filegroup(
    name = "docs_src",
    data = glob(["docs_src/**/*.md"]),
)

# A shared object which includes registration mechanisms for ops and
# kernels. Does not include the implementations of any ops or kernels. Instead,
# the library which loads libtensorflow_framework.so
# (e.g. _pywrap_tensorflow_internal.so for Python, libtensorflow.so for the C
# API) is responsible for registering ops with libtensorflow_framework.so. In
# addition to this core set of ops, user libraries which are loaded (via
# TF_LoadLibrary/tf.load_op_library) register their ops and kernels with this
# shared object directly.
#
# For example, from Python tf.load_op_library loads a custom op library (via
# dlopen() on Linux), the library finds libtensorflow_framework.so (no
# filesystem search takes place, since libtensorflow_framework.so has already
# been loaded by pywrap_tensorflow) and registers its ops and kernels via
# REGISTER_OP and REGISTER_KERNEL_BUILDER (which use symbols from
# libtensorflow_framework.so), and pywrap_tensorflow can then use these
# ops. Since other languages use the same libtensorflow_framework.so, op
# libraries are language agnostic.
#
# This shared object is not used unless framework_shared_object=true (set in the
# configure script unconditionally); otherwise if it is false or undefined, the
# build is static and TensorFlow symbols (in Python only) are loaded into the
# global symbol table in order to support op registration. This means that
# projects building with Bazel and importing TensorFlow as a dependency will not
# depend on libtensorflow_framework.so unless they opt in.
tf_cc_shared_object(
    name = "libtensorflow_framework.so",
    framework_so = [],
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:framework_internal_impl",
        "//tensorflow/core:lib_internal_impl",
        "//tensorflow/core:core_cpu_impl",
        "//tensorflow/stream_executor:stream_executor_impl",
        "//tensorflow/core:gpu_runtime_impl",
    ] + tf_additional_binary_deps(),
)

# -------------------------------------------
# New rules should be added above this target.
# -------------------------------------------

# TensorFlow uses several libraries that may also be used by applications
# linking against the C and C++ APIs (such as libjpeg).  When we create
# the shared library, only export the core TF API functions to avoid
# causing library conflicts (e.g., those reported in github issue 1924).
# On Linux, tell the linker (-Wl,<option>) to use a version script that
# excludes all but a subset of function names.
# On MacOS, the linker does not support version_script, but has an
# an "-exported_symbols_list" command.  -z defs disallows undefined
# symbols in object files and -s strips the output.

tf_cc_shared_object(
    name = "libtensorflow.so",
    linkopts = select({
        "//tensorflow:darwin": [
            "-Wl,-exported_symbols_list",  # This line must be directly followed by the exported_symbols.lds file
            "//tensorflow/c:exported_symbols.lds",
            "-Wl,-install_name,@rpath/libtensorflow.so",
        ],
        "//tensorflow:windows": [],
        "//tensorflow:windows_msvc": [],
        "//conditions:default": [
            "-z defs",
            "-s",
            "-Wl,--version-script",  #  This line must be directly followed by the version_script.lds file
            "//tensorflow/c:version_script.lds",
        ],
    }),
    deps = [
        "//tensorflow/c:c_api",
        "//tensorflow/c:exported_symbols.lds",
        "//tensorflow/c:version_script.lds",
        "//tensorflow/c/eager:c_api",
        "//tensorflow/core:tensorflow",
    ],
)

tf_cc_shared_object(
    name = "libtensorflow_cc.so",
    linkopts = select({
        "//tensorflow:darwin": [
            "-Wl,-exported_symbols_list",  # This line must be directly followed by the exported_symbols.lds file
            "//tensorflow:tf_exported_symbols.lds",
        ],
        "//tensorflow:windows": [],
        "//tensorflow:windows_msvc": [],
        "//conditions:default": [
            "-z defs",
            "-s",
            "-Wl,--version-script",  #  This line must be directly followed by the version_script.lds file
            "//tensorflow:tf_version_script.lds",
        ],
    }),
    deps = [
        "//tensorflow:tf_exported_symbols.lds",
        "//tensorflow:tf_version_script.lds",
        "//tensorflow/c:c_api",
        "//tensorflow/c/eager:c_api",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:client_session",
        "//tensorflow/cc:scope",
        "//tensorflow/cc/profiler",
        "//tensorflow/core:tensorflow",
    ],
)

exports_files(
    [
        "tf_version_script.lds",
        "tf_exported_symbols.lds",
    ],
)
