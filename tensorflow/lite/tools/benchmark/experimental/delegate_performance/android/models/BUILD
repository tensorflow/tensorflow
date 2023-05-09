# Description:
#  Holds model-specific files. The app will bundle the files into assets.

load("//tensorflow/lite/tools/benchmark/experimental/delegate_performance/android:proto.bzl", "proto_data")
load("//tensorflow/lite/tools/benchmark/experimental/delegate_performance/android:build_defs.bzl", "accuracy_benchmark_extra_models", "latency_benchmark_extra_models")
load("//tensorflow/lite/experimental/acceleration/mini_benchmark:build_defs.bzl", "validation_model")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//tensorflow/lite/tools/benchmark/experimental/delegate_performance/android:__subpackages__"],
    licenses = ["notice"],
)

# Embedded models for accuracy benchmarking.
validation_model(
    name = "mobilenet_v1_1.0_224_with_validation.tflite",
    jpegs = "//tensorflow/lite/experimental/acceleration/mini_benchmark:odt_classifier_testfiles",
    main_model = "//tensorflow/lite/experimental/acceleration/mini_benchmark/models:mobilenet_v1_1.0_224.tflite",
    metrics_model = "//tensorflow/lite/experimental/acceleration/mini_benchmark/metrics:mobilenet_metrics_tflite",
)

validation_model(
    name = "mobilenet_v1_1.0_224_quant_with_validation.tflite",
    jpegs = "//tensorflow/lite/experimental/acceleration/mini_benchmark:odt_classifier_testfiles",
    main_model = "//tensorflow/lite/experimental/acceleration/mini_benchmark/models:mobilenet_v1_1.0_224_quant.tflite",
    metrics_model = "//tensorflow/lite/experimental/acceleration/mini_benchmark/metrics:mobilenet_metrics_tflite",
)

# Migrate the models into assets folder.
ACCURACY_MODELS = [
    (
        "mobilenet_v1_1.0_224_with_validation.tflite",
        ":mobilenet_v1_1.0_224_with_validation.tflite",
    ),
    (
        "mobilenet_v1_1.0_224_quant_with_validation.tflite",
        ":mobilenet_v1_1.0_224_quant_with_validation.tflite",
    ),
] + accuracy_benchmark_extra_models()

LATENCY_MODELS = [
    (
        "mobilenet_v1_1.0_224.tflite",
        "@tflite_mobilenet_float//:mobilenet_v1_1.0_224.tflite",
    ),
    (
        "mobilenet_v1_1.0_224_quant.tflite",
        "@tflite_mobilenet_quant//:mobilenet_v1_1.0_224_quant.tflite",
    ),
] + latency_benchmark_extra_models()

COPY_CMD = """
  srcs=($(SRCS))
  outs=($(OUTS))
  for ((i = 0; i < $${#srcs[@]}; ++i)); do
    echo $${srcs[$$i]};
    cp $${srcs[$$i]} $${outs[$$i]};
  done
"""

genrule(
    name = "accuracy_models",
    srcs = [target for _, target in ACCURACY_MODELS],
    outs = ["assets/accuracy/%s" % name for name, _ in ACCURACY_MODELS],
    cmd = COPY_CMD,
)

genrule(
    name = "latency_models",
    srcs = [target for _, target in LATENCY_MODELS],
    outs = ["assets/latency/%s" % name for name, _ in LATENCY_MODELS],
    cmd = COPY_CMD,
)

# Latency criteria for latency benchmarking.
filegroup(
    name = "latency_criteria_files",
    srcs = [
        ":mobilenet_v1_1_0_224_latency_criteria",
        ":mobilenet_v1_1_0_224_quant_latency_criteria",
    ],
)

proto_data(
    name = "mobilenet_v1_1_0_224_latency_criteria",
    src = "mobilenet_v1_1.0_224.textproto",
    out = "assets/proto/mobilenet_v1_1.0_224.binarypb",
    proto_deps = ["//tensorflow/lite/tools/benchmark/experimental/delegate_performance/android/proto:delegate_performance_proto"],
    proto_name = "tflite.proto.benchmark.LatencyCriteria",
)

proto_data(
    name = "mobilenet_v1_1_0_224_quant_latency_criteria",
    src = "mobilenet_v1_1.0_224_quant.textproto",
    out = "assets/proto/mobilenet_v1_1.0_224_quant.binarypb",
    proto_deps = ["//tensorflow/lite/tools/benchmark/experimental/delegate_performance/android/proto:delegate_performance_proto"],
    proto_name = "tflite.proto.benchmark.LatencyCriteria",
)
