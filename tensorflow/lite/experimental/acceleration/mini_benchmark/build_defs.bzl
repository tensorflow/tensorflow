# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Helpers for mini-benchmark build rules."""

load(
    "//tensorflow:tensorflow.bzl",
    "clean_dep",
)

def embedded_binary(name, binary, array_variable_name, testonly = False):
    """Create a cc_library that embeds a binary as constant data.

    Args:
        name: name for the generated cc_library target, and the base name for
              generated header file
        binary: binary file to be embedded
        array_variable_name: name of the constant array for the data.
    """
    cc_name = "%s.cc" % name
    h_name = "%s.h" % name
    native.genrule(
        name = name + "_src",
        srcs = [binary],
        outs = [
            cc_name,
            h_name,
        ],
        cmd = """
      $(location //tensorflow/lite/experimental/acceleration/compatibility:convert_binary_to_cc_source) \
          --input_binary_file $(location %s) \
          --output_header_file $(location :%s) \
          --output_source_file $(location :%s) \
          --array_variable_name %s
      """ % (binary, h_name, cc_name, array_variable_name),
        tools = ["//tensorflow/lite/experimental/acceleration/compatibility:convert_binary_to_cc_source"],
        testonly = testonly,
    )

    native.cc_library(
        name = name,
        srcs = [cc_name],
        hdrs = [h_name],
        testonly = testonly,
    )

def validation_model(
        name,
        main_model,
        metrics_model,
        jpegs,
        scale = "",
        zeropoint = "",
        use_ondevice_cpu_for_golden = False,
        testonly = 0):
    """Create a tflite model with embedded validation.

    Args:
        name: name of the target. A file called 'name'.tflite is generated
        main_model: main tflite model target
        metrics_model: metrics tflite model target
        jpegs: target with 1 or more jpeg files
        scale: the input (de)quantization scale parameter for float models
        zeropoint: the input (de)quantization zeropoint parameter for float models
        use_ondevice_cpu_for_golden: use on-device CPU for golden data (rather than embedding)
        testonly: whether target is marked testonly
    """
    if use_ondevice_cpu_for_golden:
        use_ondevice_cpu_for_golden = "true"
    else:
        use_ondevice_cpu_for_golden = "false"
    scale_arg = ""
    zeropoint_arg = ""
    if scale:
        scale_arg = "--scale=" + scale
        zeropoint_arg = "--zero_point=" + zeropoint
    native.genrule(
        name = name,
        testonly = testonly,
        srcs = [
            main_model,
            jpegs,
            "//tensorflow/lite/schema:schema.fbs",
            metrics_model,
        ],
        outs = [name + ".tflite"],
        cmd = """
          JPEGS='$(locations %s)'
          JPEGS=$${JPEGS// /,}
          $(location //tensorflow/lite/experimental/acceleration/mini_benchmark/model_modifier:embedder_cmdline) \
              --schema=$(location //tensorflow/lite/schema:schema.fbs) \
              --main_model=$(location %s) \
              --metrics_model=$(location %s) \
              %s %s \
              --jpegs=$$JPEGS \
              --use_ondevice_cpu_for_golden=%s \
              --output=$(@D)/tmp
          $(location //tensorflow/lite/experimental/acceleration/mini_benchmark:copy_associated_files) \
              $(@D)/tmp \
              $(location %s) \
              $(location %s.tflite)
          rm $(@D)/tmp
        """ % (jpegs, main_model, metrics_model, scale_arg, zeropoint_arg, use_ondevice_cpu_for_golden, main_model, name),
        tools = [
            "//tensorflow/lite/experimental/acceleration/mini_benchmark/model_modifier:embedder_cmdline",
            "//tensorflow/lite/experimental/acceleration/mini_benchmark:copy_associated_files",
        ],
    )

def validation_test(name, validation_model, tags = [], copts = [], deps = []):
    """Create a test binary for the given model with validation.

    Args:
        name: name of the target.
        validation_model: tflite model with validation target.
        tags: to be passed to cc_test.
        copts: to be passed to cc_test.
        deps: to be passed to cc_test.
    """
    embed_name = name + "_embed_model"
    embedded_binary(
        embed_name,
        binary = validation_model,
        array_variable_name = "g_tflite_acceleration_" + name + "_model",
    )
    native.cc_test(
        name = name,
        srcs = ["//tensorflow/lite/experimental/acceleration/mini_benchmark:model_validation_test.cc"],
        tags = tags + ["no_mac", "no_windows", "tflite_not_portable_ios"],
        copts = copts + [
            "-DTENSORFLOW_ACCELERATION_MODEL_DATA_VARIABLE=\"g_tflite_acceleration_%s_model\"" % name,
            "-DTENSORFLOW_ACCELERATION_MODEL_LENGTH_VARIABLE=\"g_tflite_acceleration_%s_model_len\"" % name,
        ],
        deps = deps + [
            embed_name,
            "@com_google_googletest//:gtest_main",
            "@flatbuffers",
            "//tensorflow/lite/experimental/acceleration/compatibility:android_info",
            "//tensorflow/lite/experimental/acceleration/configuration:configuration_fbs",
            "//tensorflow/lite/experimental/acceleration/configuration:nnapi_plugin",
            "//tensorflow/lite/experimental/acceleration/mini_benchmark:big_little_affinity",
            "//tensorflow/lite/experimental/acceleration/mini_benchmark:model_loader",
            "//tensorflow/lite/experimental/acceleration/mini_benchmark:status_codes",
            "//tensorflow/lite/experimental/acceleration/mini_benchmark:validator",
        ] + select({
            clean_dep("//tensorflow:android"): [
                "//tensorflow/lite/experimental/acceleration/configuration:gpu_plugin",
            ],
            "//conditions:default": [],
        }),
    )
