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
"""Runs sample models with TensorRT and analyzes latency and numerics information."""

import functools
import os
import tempfile
from typing import Callable, Iterable, Sequence

from absl import app
from absl import flags

from tensorflow.python.compiler.tensorrt import trt_convert as trt
from tensorflow.python.compiler.tensorrt.model_tests import model_handler
from tensorflow.python.compiler.tensorrt.model_tests import result_analyzer
from tensorflow.python.eager import context
from tensorflow.python.framework import config as framework_config
from tensorflow.python.framework import ops as framework_ops
from tensorflow.python.platform import gfile
from tensorflow.python.platform import test as platform_test
from tensorflow.python.platform import tf_logging as logging
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

flags.DEFINE_boolean("use_int8", True,
                     "Whether to convert with INT8 precision.")

flags.DEFINE_enum("latency_baseline", "GPU", ["CPU", "GPU"],
                  "The baseline version for latency improvement analysis.")

flags.DEFINE_enum("numerics_baseline", "CPU", ["CPU", "GPU"],
                  "The baseline version for numerical difference analysis.")

flags.DEFINE_float(
    "speedup_tolerance", 0.95,
    "Log errors whenever mean TensorRT speedup is lower than the tolerance.")

flags.DEFINE_float(
    "diff_tolerance", 0.05,
    "Log errors whenever mean TensorRT relative difference is larger than "
    "the tolerance.")

flags.DEFINE_integer(
    "gpu_memory_limit_mb", None,
    "Limitation on the device memory being used during TensorRT compilation "
    "and inference.")

flags.DEFINE_string("output_dir", None, "Output directory of analysis results.")

flags.DEFINE_enum("output_format", "CSV", ["CSV", "JSON"],
                  "Output format of analysis results.")

DEFAUL_TRT_CONVERT_PARAMS = trt.DEFAULT_TRT_CONVERSION_PARAMS


# pylint: disable=bad-whitespace
def set_up_gpu_memory_limit(memory_limit_mb: int) -> None:
  gpus = framework_config.list_physical_devices("GPU")
  virtual_device_config = context.LogicalDeviceConfiguration(
      memory_limit=memory_limit_mb)
  for gpu in gpus:
    framework_config.set_logical_device_configuration(gpu,
                                                      [virtual_device_config])


class SampleRunner(object):
  """The driver to run all sample models in all specified configurations."""

  def __init__(self, saved_model_dir: str, saved_model_tags: Sequence[str],
               saved_model_signature_key: str, batch_size: int, output_dir: str,
               output_format: str, use_tf2: bool, use_int8: bool,
               analyzer: result_analyzer.ResultAnalyzer):
    self._output_dir = output_dir or tempfile.mkdtemp(
        prefix="tf2trt_model_tests")
    logging.info("Use output directory as: %s", self._output_dir)
    self._output_format = output_format
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

    if use_int8:
      self._precision_modes = [
          trt.TrtPrecisionMode.FP32, trt.TrtPrecisionMode.FP16,
          trt.TrtPrecisionMode.INT8]
    else:
      self._precision_modes = [
          trt.TrtPrecisionMode.FP32, trt.TrtPrecisionMode.FP16]

    self._analyzer = analyzer

  def _write_analysis_result(self, df: result_analyzer.DataFrame,
                             path: str) -> None:
    if self._output_format == "CSV":
      df.to_csv(os.path.join(path, "result.csv"))
    elif self._output_format == "JSON":
      df.to_json(os.path.join(path, "result.json"))
    else:
      raise NotImplementedError("Unsupported output format: {}".format(
          self._output_format))

  def _run_impl(
      self, test_name: str,
      default_trt_converter_params: trt.TrtConversionParams,
      trt_converter_params_updater: Callable[[trt.TrtConversionParams],
                                             Iterable[trt.TrtConversionParams]]
  ) -> None:
    """Runs all sample models based on a key varying parameter."""
    for model_config in self._configs:
      # Loads, compiles, calibrates and runs models.
      manager = self._model_handler_manager_cls(
          name=test_name,
          model_config=model_config,
          default_trt_convert_params=default_trt_converter_params,
          trt_convert_params_updater=trt_converter_params_updater)
      inputs = manager.generate_random_inputs()
      # As all the data are randomly generated, directly use inference data as
      # calibration data to produce reliable dynamic ranges.
      manager.convert(inputs)
      test_results = manager.run(inputs)

      # Analyzes the latency and numerical results.
      analysis_result_df, _ = self._analyzer.analysis(test_results)

      # Outputs the analysis results
      model_name = os.path.split(manager.model_config.saved_model_dir)[-1]
      model_dir = os.path.join(self._output_dir, model_name)
      gfile.MkDir(model_dir)
      test_dir = os.path.join(model_dir, test_name)
      gfile.MkDir(test_dir)
      with gfile.Open(
          os.path.join(test_dir, "default_tensorrt_params.txt"), "w") as f:
        f.write(repr(default_trt_converter_params))
      self._write_analysis_result(analysis_result_df, test_dir)

  def run_trt_precision_tests(self) -> None:
    """Runs tests for all TensorRT precisions."""

    def trt_converter_params_updater(params: trt.TrtConversionParams):
      for precision_mode in self._precision_modes:
        yield params._replace(
            precision_mode=precision_mode,
            use_calibration=(precision_mode == trt.TrtPrecisionMode.INT8))

    self._run_impl(
        test_name="precision_mode_test",
        default_trt_converter_params=DEFAUL_TRT_CONVERT_PARAMS,
        trt_converter_params_updater=trt_converter_params_updater)

  def run_all_tests(self) -> None:
    """Runs all tests available."""
    self.run_trt_precision_tests()
    logging.info("Check analysis result at: %s", self._output_dir)


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

  if FLAGS.use_int8:
    logging.info("Will try converting with INT8 precision.")
  else:
    logging.info("Will not try converting with INT8 precision.")

  if FLAGS.gpu_memory_limit_mb:
    set_up_gpu_memory_limit(FLAGS.gpu_memory_limit_mb)

  analyzer = result_analyzer.ResultAnalyzer(
      use_cpu_latency_baseline=FLAGS.latency_baseline == "CPU",
      use_cpu_numerics_baseline=FLAGS.numerics_baseline == "CPU",
      checkers=[
          functools.partial(
              result_analyzer.check_column,
              name="speedup",
              fn=lambda x: x > FLAGS.speedup_tolerance),
          functools.partial(
              result_analyzer.check_column,
              name="rel_diff_mean",
              fn=lambda x: all(v < FLAGS.diff_tolerance for v in x.values()))
      ])
  runner = SampleRunner(
      saved_model_dir=FLAGS.saved_model_dir,
      saved_model_tags=FLAGS.saved_model_tags,
      saved_model_signature_key=FLAGS.saved_model_signature_key,
      batch_size=FLAGS.batch_size,
      output_dir=FLAGS.output_dir,
      output_format=FLAGS.output_format,
      use_tf2=FLAGS.use_tf2,
      use_int8=FLAGS.use_int8,
      analyzer=analyzer)

  runner.run_all_tests()


if __name__ == "__main__":
  app.run(main)
