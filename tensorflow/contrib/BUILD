# Description:
#   contains parts of TensorFlow that are experimental or unstable and which are not supported.

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package(default_visibility = ["//tensorflow:__subpackages__"])

py_library(
    name = "contrib_py",
    srcs = glob(["**/*.py"]),
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/contrib/bayesflow:bayesflow_py",
        "//tensorflow/contrib/compiler:compiler_py",
        "//tensorflow/contrib/copy_graph:copy_graph_py",
        "//tensorflow/contrib/crf:crf_py",
        "//tensorflow/contrib/cudnn_rnn:cudnn_rnn_py",
        "//tensorflow/contrib/deprecated:deprecated_py",
        "//tensorflow/contrib/distributions:distributions_py",
        "//tensorflow/contrib/factorization:factorization_py",
        "//tensorflow/contrib/ffmpeg:ffmpeg_ops_py",
        "//tensorflow/contrib/framework:framework_py",
        "//tensorflow/contrib/graph_editor:graph_editor_py",
        "//tensorflow/contrib/grid_rnn:grid_rnn_py",
        "//tensorflow/contrib/image:image_py",
        "//tensorflow/contrib/input_pipeline:input_pipeline_py",
        "//tensorflow/contrib/integrate:integrate_py",
        "//tensorflow/contrib/labeled_tensor",
        "//tensorflow/contrib/layers:layers_py",
        "//tensorflow/contrib/learn",
        "//tensorflow/contrib/legacy_seq2seq:seq2seq_py",
        "//tensorflow/contrib/linalg:linalg_py",
        "//tensorflow/contrib/linear_optimizer:sdca_ops_py",
        "//tensorflow/contrib/lookup:lookup_py",
        "//tensorflow/contrib/losses:losses_py",
        "//tensorflow/contrib/metrics:metrics_py",
        "//tensorflow/contrib/ndlstm",
        "//tensorflow/contrib/opt:opt_py",
        "//tensorflow/contrib/quantization:quantization_py",
        "//tensorflow/contrib/rnn:rnn_py",
        "//tensorflow/contrib/seq2seq:seq2seq_py",
        "//tensorflow/contrib/slim",
        "//tensorflow/contrib/slim:nets",
        "//tensorflow/contrib/solvers:solvers_py",
        "//tensorflow/contrib/specs",
        "//tensorflow/contrib/stat_summarizer:stat_summarizer_py",
        "//tensorflow/contrib/tensor_forest:tensor_forest_py",
        "//tensorflow/contrib/tensor_forest/hybrid:ops_lib",
        "//tensorflow/contrib/tensorboard",
        "//tensorflow/contrib/testing:testing_py",
        "//tensorflow/contrib/tfprof",
        "//tensorflow/contrib/training:training_py",
        "//tensorflow/contrib/util:util_py",
    ],
)

cc_library(
    name = "contrib_kernels",
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/contrib/factorization/kernels:all_kernels",
        "//tensorflow/contrib/input_pipeline:input_pipeline_ops_kernels",
        "//tensorflow/contrib/layers:bucketization_op_kernel",
        "//tensorflow/contrib/layers:sparse_feature_cross_op_kernel",
    ],
)

cc_library(
    name = "contrib_ops_op_lib",
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/contrib/factorization:all_ops",
        "//tensorflow/contrib/framework:all_ops",
        "//tensorflow/contrib/input_pipeline:input_pipeline_ops_op_lib",
        "//tensorflow/contrib/layers:bucketization_op_op_lib",
        "//tensorflow/contrib/layers:sparse_feature_cross_op_op_lib",
    ],
)

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
    visibility = ["//tensorflow:__subpackages__"],
)
