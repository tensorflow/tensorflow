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
from typing import Callable, Iterable, Sequence

from absl import app
from absl import flags
from absl import logging

from tensorflow.python.compiler.tensorrt import trt_convert as trt
from tensorflow.python.compiler.tensorrt.model_tests import model_handler
from tensorflow.python.framework import ops as framework_ops
from tensorflow.python.platform import test as platform_test
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "saved_model_dir",
    platform_test.test_src_dir_path(
        "python/compiler/tensorrt/model_tests/sample_model"),
    "The directory to the testing SavedModel.")

flags.DEFINE_string("saved_model_signature_key",
                    signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY,
                    "The signature key of the testing SavedModel being used.")

flags.DEFINE_multi_string("saved_model_tags", (tag_constants.SERVING,),
                          "The tags of the testing SavedModel being used.")

flags.DEFINE_integer("batch_size", 128,
                     "The batch size used to run the testing model with.")

flags.DEFINE_boolean("use_tf2", True,
                     "Whether to test with TF2 behavior or not (TF1).")
DEFAUL_TRT_CONVERT_PARAMS = trt.DEFAULT_TRT_CONVERSION_PARAMS


def _get_mean_latency(result: model_handler.TestResult):
  return (sum(result.latency) / len(result.latency)) * 1000.0


class SampleRunner(object):
  """The driver to run all sample models in all specified configurations."""

  def __init__(self,
               saved_model_dir: str,
               saved_model_tags: Sequence[str],
               saved_model_signature_key: str,
               batch_size: int,
               use_tf2=True):
    # The model_configs contains (saved_model_dir, saved_model_signature_key,
    # batch_size) for each model
    self._configs = (model_handler.ModelConfig(
        saved_model_dir=saved_model_dir,
        saved_model_tags=tuple(saved_model_tags),
        saved_model_signature_key=saved_model_signature_key,
        default_batch_size=batch_size),)
    self._model_handler_manager_cls = (
        model_handler.ModelHandlerManagerV2
        if use_tf2 else model_handler.ModelHandlerManagerV1)
    self._default_trt_convert_params = (
        DEFAUL_TRT_CONVERT_PARAMS._replace(is_dynamic_op=True)
        if use_tf2 else DEFAUL_TRT_CONVERT_PARAMS._replace(is_dynamic_op=False))

  def _run_impl(
      self,
      default_trt_converter_params: trt.TrtConversionParams,
      trt_converter_params_updater: Callable[[trt.TrtConversionParams],
                                             Iterable[trt.TrtConversionParams]],
  ):
    """Runs all sample models based on a key varying parameter."""
    for model_config in self._configs:
      trt_convert_params = default_trt_converter_params._replace(
          max_batch_size=model_config.default_batch_size)
      # Load, compile and runs the models.
      manager = self._model_handler_manager_cls(
          model_config=model_config,
          default_trt_convert_params=trt_convert_params,
          trt_convert_params_updater=trt_converter_params_updater)
      inputs = manager.generate_random_inputs()
      # As all the data are randomly generated, directly use inference data as
      # calibration data to produce reliable dynamic ranges.
      manager.convert(inputs)
      result_collection = manager.run(inputs)

      logging.info("Model information: %s", repr(manager))
      for result in result_collection.results:
        logging.info("TensorRT parameters: %s", result.trt_convert_params or
                     "Not a TensorRT Model")
        logging.info("Mean latency: %f ms", _get_mean_latency(result))

  def run_trt_precision_tests(self) -> None:
    """Runs tests for all TensorRT precisions."""

    def trt_converter_params_updater(params: trt.TrtConversionParams):
      for precision_mode in [
          trt.TrtPrecisionMode.FP32, trt.TrtPrecisionMode.FP16,
          trt.TrtPrecisionMode.INT8
      ]:
        yield params._replace(
            precision_mode=precision_mode,
            use_calibration=(precision_mode == trt.TrtPrecisionMode.INT8))

    self._run_impl(
        default_trt_converter_params=self._default_trt_convert_params,
        trt_converter_params_updater=trt_converter_params_updater)

  def run_all_tests(self) -> None:
    """Runs all tests available."""
    self.run_trt_precision_tests()


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

  SampleRunner(
      saved_model_dir=FLAGS.saved_model_dir,
      saved_model_tags=FLAGS.saved_model_tags,
      saved_model_signature_key=FLAGS.saved_model_signature_key,
      batch_size=FLAGS.batch_size,
      use_tf2=FLAGS.use_tf2).run_all_tests()


if __name__ == "__main__":
  app.run(main)
