# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Runs sample models with TensorRT and analyzes numerics and timing information."""

import os

from absl import app
from absl import flags
from absl import logging

from tensorflow.python.compiler.tensorrt import trt_convert as trt
from tensorflow.python.compiler.tensorrt.model_tests import model_handler
from tensorflow.python.framework import ops as framework_ops
from tensorflow.python.platform import test as platform_test

FLAGS = flags.FLAGS
flags.DEFINE_boolean("use_tf2", True,
                     "Whether to test with TF2 behavior or not (TF1).")

DEFAUL_TRT_CONVERT_PARAMS = trt.DEFAULT_TRT_CONVERSION_PARAMS


def _get_mean_latency(result: model_handler.TestResult):
  return (sum(result.latency) / len(result.latency)) * 1000.0


def run_all_tests():
  """Runs all sample model with TensorRT FP32/FP16 and reports latency."""
  model_configs = (model_handler.ModelConfig(
      saved_model_dir=platform_test.test_src_dir_path(
          "python/compiler/tensorrt/model_tests/sample_model"),
      default_batch_size=128),)
  if FLAGS.use_tf2:
    model_handler_cls = model_handler.ModelHandlerV2
    trt_model_handeler_cls = model_handler.TrtModelHandlerV2
    default_trt_convert_params = DEFAUL_TRT_CONVERT_PARAMS._replace(
        is_dynamic_op=True)
  else:
    model_handler_cls = model_handler.ModelHandlerV1
    trt_model_handeler_cls = model_handler.TrtModelHandlerV1
    default_trt_convert_params = DEFAUL_TRT_CONVERT_PARAMS._replace(
        is_dynamic_op=False)
  for model_config in model_configs:
    trt_convert_params = default_trt_convert_params._replace(
        max_batch_size=model_config.default_batch_size)
    base_model = model_handler_cls(model_config)
    random_inputs = base_model.generate_random_inputs()
    base_model_result = base_model.run(random_inputs)
    trt_fp32_model_result = trt_model_handeler_cls(
        model_config=model_config,
        trt_convert_params=trt_convert_params._replace(
            precision_mode=trt.TrtPrecisionMode.FP32)).run(random_inputs)
    trt_fp16_model_result = trt_model_handeler_cls(
        model_config=model_config,
        trt_convert_params=trt_convert_params._replace(
            precision_mode=trt.TrtPrecisionMode.FP16)).run(random_inputs)

    logging.info("Base model latency: %f ms",
                 _get_mean_latency(base_model_result))
    logging.info("TensorRT FP32 model latency: %f ms",
                 _get_mean_latency(trt_fp32_model_result))
    logging.info("TensorRT FP16 model latency: %f ms",
                 _get_mean_latency(trt_fp16_model_result))


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  os.environ["TF_TRT_ALLOW_ENGINE_NATIVE_SEGMENT_EXECUTION"] = "False"

  if FLAGS.use_tf2:
    logging.info("Running in TF2 mode. Eager execution is enabled.")
    framework_ops.enable_eager_execution()
  else:
    logging.info("Running in TF1 mode. Eager execution is disabled.")
    framework_ops.disable_eager_execution()

  run_all_tests()


if __name__ == "__main__":
  app.run(main)
