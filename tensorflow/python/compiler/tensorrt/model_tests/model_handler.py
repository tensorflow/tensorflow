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
"""Loads, converts, calibrates, and runs sample models."""

import abc
import collections
import functools
import itertools
import tempfile
import time
from typing import Callable, Iterable, List, Mapping, Optional, Sequence, Union

from absl import logging
import numpy as np

from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import tensor_shape_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.python.client import session
from tensorflow.python.compiler.tensorrt import trt_convert as trt
from tensorflow.python.framework import convert_to_constants
from tensorflow.python.framework import dtypes as tf_dtypes
from tensorflow.python.framework import importer
from tensorflow.python.framework import ops as framework_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.saved_model import load as saved_model_load
from tensorflow.python.saved_model import loader as saved_model_loader
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants


# pylint: disable=bad-whitespace
### Helper Functions
def _remove_graph_sequence_number(name: str) -> str:
  return name.split(":")[0]


def _get_concrete_tensor_shape(
    tensor_shape: tensor_shape_pb2.TensorShapeProto,
    batch_size: Optional[int] = None) -> Sequence[int]:
  """Gets a concrete tensor shape without dynamic dimensions."""
  if tensor_shape.unknown_rank:
    raise ValueError("Cannot generates random tensors for unknown rank!")
  shape = [dim.size for dim in tensor_shape.dim]
  if not shape:
    raise ValueError("The tensor cannot have a rank of 0!")
  if shape[0] < 0:
    if batch_size is None or batch_size <= 0:
      raise ValueError("Must provide a valid batch size "
                       "as the tensor has a dynamic batch size!")
    shape[0] = batch_size
  if any(filter(lambda x: x < 0, shape)):
    raise ValueError("Cannot have dynamic dimensions except for batch size!")
  return shape


def _generate_random_tensor_ops(shape: Sequence[int], dtype: tf_dtypes.DType,
                                name: str) -> framework_ops.Tensor:
  # Need to generate a random tensor in float32/int32 and cast to a different
  # datatype as random_ops doesn't suppprt all the datatypes.
  random_dtype = tf_dtypes.float32 if dtype.is_floating else tf_dtypes.int32
  # tf.bool doesn't have `max` attribute
  dtype_max = 1 if dtype == tf_dtypes.bool else dtype.max
  return math_ops.cast(
      random_ops.random_uniform(
          shape=shape,
          dtype=random_dtype,
          # Limits maximum value as 255 to simulate pixel values, avoid
          # generating large numbers and casuing overflows.
          maxval=min(dtype_max, random_dtype.max, 255)),
      dtype=dtype,
      name=name)


def _generate_random_tensor_v1(tensor_info: meta_graph_pb2.TensorInfo,
                               batch_size: Optional[int] = None) -> np.ndarray:
  """Generates a random tensor based on the data type and tensor shape."""
  dtype = tf_dtypes.as_dtype(tensor_info.dtype)
  shape = _get_concrete_tensor_shape(tensor_info.tensor_shape, batch_size)
  with framework_ops.Graph().as_default() as graph, session.Session(
      graph=graph):
    return _generate_random_tensor_ops(
        shape=shape,
        dtype=dtype,
        name=_remove_graph_sequence_number(tensor_info.name)).eval()


def _generate_random_tensor_v2(
    tensor: framework_ops.Tensor,
    batch_size: Optional[int] = None) -> framework_ops.Tensor:
  """Generates a random tensor based on the data type and tensor shape."""
  shape = _get_concrete_tensor_shape(tensor.shape.as_proto(), batch_size)
  return _generate_random_tensor_ops(
      shape=shape, dtype=tensor.dtype, name=tensor.name)


# Models are repeatedly loaded for different TensorRT conversion settings.
# Using cache can reduce I/O.
@functools.lru_cache()
def load_meta_graph(
    saved_model_dir: str, saved_model_tags: str,
    saved_model_signature_key: str) -> meta_graph_pb2.MetaGraphDef:
  """Loads a `tf.MetaGraphDef` in TF1."""
  with framework_ops.Graph().as_default() as graph, session.Session(
      graph=graph) as sess:
    meta_graph = saved_model_loader.load(
        sess=sess,
        export_dir=saved_model_dir,
        tags=saved_model_tags,
    )
    output_node_names = [
        _remove_graph_sequence_number(tensor.name) for tensor in
        meta_graph.signature_def[saved_model_signature_key].outputs.values()
    ]
    graph_def = (
        convert_to_constants.convert_variables_to_constants_from_session_graph(
            sess, meta_graph.graph_def, output_node_names))
    meta_graph.graph_def.CopyFrom(graph_def)
  return meta_graph


@functools.lru_cache()
def load_graph_func(saved_model_dir: str, saved_model_tags: str,
                    saved_model_signature_key: str):
  """Loads a graph function in TF2."""
  imported = saved_model_load.load(
      export_dir=saved_model_dir, tags=saved_model_tags)
  graph_func = imported.signatures[saved_model_signature_key]
  return convert_to_constants.convert_variables_to_constants_v2(graph_func)


### Test Classes
class ModelConfig(
    collections.namedtuple("ModelConfig", [
        "saved_model_dir", "saved_model_tags", "saved_model_signature_key",
        "default_batch_size"
    ])):
  """Configurations for test models."""

  def __new__(cls,
              saved_model_dir: str,
              saved_model_tags: Sequence[str] = (tag_constants.SERVING,),
              saved_model_signature_key: str = (
                  signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY),
              default_batch_size: int = 1):
    return super(ModelConfig,
                 cls).__new__(cls, saved_model_dir, saved_model_tags,
                              saved_model_signature_key, default_batch_size)


class TestResult(
    collections.namedtuple("TestResult", [
        "model_config", "enable_gpu", "output_names", "output_tensors",
        "model_latency", "trt_convert_params"
    ])):
  """Configuration and results for a single model testing."""

  def __new__(cls,
              model_config: ModelConfig,
              enable_gpu: bool,
              output_names: Sequence[str],
              output_tensors: Sequence[np.ndarray],
              model_latency: List[float],
              trt_convert_params: trt.TrtConversionParams = None):
    return super(TestResult,
                 cls).__new__(cls, model_config, enable_gpu, output_names,
                              output_tensors, model_latency, trt_convert_params)


class TestResultCollection(
    collections.namedtuple("TestResultCollection", [
        "test_name", "model_config", "cpu_base_result", "gpu_base_result",
        "trt_results"
    ])):
  """Configuration and results for a series of model testing."""

  def __new__(cls,
              test_name: str,
              model_config: ModelConfig,
              cpu_base_result: TestResult,
              gpu_base_result: TestResult,
              trt_results: Sequence[TestResult] = tuple()):
    return super(TestResultCollection,
                 cls).__new__(cls, test_name, model_config, cpu_base_result,
                              gpu_base_result, trt_results)

  @property
  def results(self) -> Iterable[TestResult]:
    return filter(
        lambda x: x is not None,
        itertools.chain([self.cpu_base_result, self.gpu_base_result],
                        self.trt_results))


class _ModelHandlerBase(metaclass=abc.ABCMeta):
  """Base class for running a model."""

  def __init__(self, model_config: ModelConfig):
    self._model_config = model_config

  def __str__(self) -> str:
    return str(self._model_config)

  def __repr__(self) -> str:
    return "{}({})".format(self.__class__.__name__, str(self))

  @property
  def model_config(self) -> ModelConfig:
    return self._model_config

  @property
  def input_tensort_names(self) -> Sequence[str]:
    """Names of input tensors."""

  @property
  def output_tensor_names(self) -> Sequence[str]:
    """Names of output tensors."""

  @abc.abstractmethod
  def generate_random_inputs(
      self,
      batch_size: Optional[int] = None
  ) -> Mapping[str, Union[np.ndarray, framework_ops.Tensor]]:
    """Generates mapping from names to input tensors."""

  @abc.abstractmethod
  def run(self,
          inputs=None,
          warmup_iterations: int = 10,
          benchmark_iterations: int = 100,
          enable_gpu: bool = True) -> TestResult:
    """Runs the model with provided or randomly generated input tensors.

    Args:
      inputs: Mapping from names to input ndarrays in TF1, or a sequence of
        tensors in TF2. If `None`, ramdomly generated inputs will be used
        instead.
      warmup_iterations: Number of inferences to warm up the runtime.
      benchmark_iterations: Number of inferences to measure the latency.
      enable_gpu: Whether it is allowed to use GPU or not.

    Returns:
      `TestResult` summarizing latency and numerics information.
    """


class ModelHandlerV1(_ModelHandlerBase):
  """Runs a model in TF1."""

  @property
  def meta_graph(self) -> meta_graph_pb2.MetaGraphDef:
    return load_meta_graph(
        saved_model_dir=self.model_config.saved_model_dir,
        saved_model_tags=self.model_config.saved_model_tags,
        saved_model_signature_key=self.model_config.saved_model_signature_key)

  @property
  def input_tensor_info(self) -> Mapping[str, meta_graph_pb2.TensorInfo]:
    return self.meta_graph.signature_def[
        self.model_config.saved_model_signature_key].inputs

  @property
  def output_tensor_info(self) -> Mapping[str, meta_graph_pb2.TensorInfo]:
    return self.meta_graph.signature_def[
        self.model_config.saved_model_signature_key].outputs

  @property
  def input_tensort_names(self) -> Sequence[str]:
    return [info.name for info in self.input_tensor_info.values()]

  @property
  def output_tensor_names(self) -> Sequence[str]:
    return [info.name for info in self.output_tensor_info.values()]

  def generate_random_inputs(self,
                             batch_size: Optional[int] = None
                            ) -> Mapping[str, np.ndarray]:
    batch_size = batch_size or self.model_config.default_batch_size
    return {
        tensor_info.name: _generate_random_tensor_v1(tensor_info, batch_size)
        for tensor_info in self.input_tensor_info.values()
    }

  def run(self,
          inputs: Optional[Mapping[str, np.ndarray]] = None,
          warmup_iterations=10,
          benchmark_iterations=100,
          enable_gpu=True) -> TestResult:
    inputs = inputs or self.generate_random_inputs()
    config_proto = None
    if not enable_gpu:
      config_proto = config_pb2.ConfigProto(device_count={"CPU": 1, "GPU": 0})
    logging.info("Running model inference!")
    with framework_ops.Graph().as_default():
      with session.Session(config=config_proto) as sess:
        importer.import_graph_def(self.meta_graph.graph_def, name="")
        try:
          output_tensor_names = self.output_tensor_names
          for _ in range(warmup_iterations):
            sess.run(fetches=output_tensor_names, feed_dict=inputs)
          latency = []
          for _ in range(benchmark_iterations):
            before = time.time()
            outputs = sess.run(fetches=output_tensor_names, feed_dict=inputs)
            latency.append(time.time() - before)
        except Exception as exc:
          raise RuntimeError("Failed to run model inference! "
                             "Model information: {}".format(str(self))) from exc
    return TestResult(
        model_config=self.model_config,
        enable_gpu=enable_gpu,
        model_latency=latency,
        output_names=self.output_tensor_names,
        output_tensors=outputs)


class ModelHandlerV2(_ModelHandlerBase):
  """Runs a model in TF2."""

  @property
  def graph_func(self):
    graph_func = load_graph_func(
        saved_model_dir=self.model_config.saved_model_dir,
        saved_model_tags=self.model_config.saved_model_tags,
        saved_model_signature_key=self.model_config.saved_model_signature_key)
    return convert_to_constants.convert_variables_to_constants_v2(graph_func)

  @property
  def input_tensor_names(self):
    return [tensor.name for tensor in self.graph_func.inputs]

  @property
  def output_tensor_names(self):
    return [tensor.name for tensor in self.graph_func.outputs]

  def generate_random_inputs(self,
                             batch_size: Optional[int] = None
                            ) -> Sequence[framework_ops.Tensor]:
    batch_size = batch_size or self.model_config.default_batch_size
    return [
        _generate_random_tensor_v2(tensor, batch_size)
        for tensor in self.graph_func.inputs
    ]

  def run(self,
          inputs: Optional[Sequence[framework_ops.Tensor]] = None,
          warmup_iterations=10,
          benchmark_iterations=100,
          enable_gpu=True) -> TestResult:
    inputs = inputs or self.generate_random_inputs()
    try:
      device = "/device:gpu:0" if enable_gpu else "/device:cpu:0"
      with framework_ops.device(device):
        for _ in range(warmup_iterations):
          self.graph_func(*inputs)
        latency = []
        for _ in range(benchmark_iterations):
          before = time.time()
          outputs = self.graph_func(*inputs)
          latency.append(time.time() - before)
    except Exception as exc:
      raise RuntimeError("Failed to run model inference! "
                         "Model information: {}".format(str(self))) from exc
    return TestResult(
        model_config=self.model_config,
        enable_gpu=enable_gpu,
        model_latency=latency,
        output_names=self.output_tensor_names,
        output_tensors=outputs)


class _TrtModelHandlerBase(_ModelHandlerBase):
  """Base class for converting and running a model."""

  def __init__(
      self,
      model_config: ModelConfig,
      trt_convert_params: trt.TrtConversionParams,
  ):
    super(_TrtModelHandlerBase, self).__init__(model_config)
    self._trt_convert_params = trt_convert_params

    self._converter = self._create_converter(trt_convert_params)
    self._conversion_is_saved = False

  @abc.abstractmethod
  def _create_converter(self, trt_convert_params: trt.TrtConversionParams):
    """Creates a converter for the corresponding TF version."""

  @abc.abstractmethod
  def _check_conversion(self, conversion_output):
    """Checks if conversion output has any TensorRT engines."""

  def _check_contains_trt_engine(self, graph_def: graph_pb2.GraphDef):
    if "TRTEngineOp" not in [node.op for node in graph_def.node]:
      raise RuntimeError("Failed to convert to TensorRT! "
                         "Model Information: {}".format(str(self)))

  def __str__(self) -> str:
    base = super(_TrtModelHandlerBase, self).__str__()
    return "{}, TrtConversionParams: {}".format(base,
                                                str(self._trt_convert_params))

  @property
  def trt_convert_params(self) -> trt.TrtConversionParams:
    return self._trt_convert_params

  @abc.abstractmethod
  def convert(self,
              calibration_inputs: Optional[Mapping[str, np.ndarray]] = None,
              num_runs=1) -> None:
    """Converts the model with TensorRT and calibrates if using INT8 precision mode.

    Args:
      calibration_inputs: Mapping from input names to ndarrays in TF1. Or a
        sequence of tensors in TF2. Used as calibration data.
      num_runs: Number of calibration runs.
    """

  def save(self,
           output_saved_model_dir: Optional[str] = None,
           overwrite=True) -> None:
    """Saves a TensorRT converted model."""
    if self._conversion_is_saved and not overwrite:
      return
    output_saved_model_dir = output_saved_model_dir or tempfile.mkdtemp()
    logging.info("Saving TensorRT model to %s!", output_saved_model_dir)
    self._converter.save(output_saved_model_dir)
    self._model_config = self.model_config._replace(
        saved_model_dir=output_saved_model_dir)
    self._conversion_is_saved = True


class TrtModelHandlerV1(_TrtModelHandlerBase, ModelHandlerV1):
  """Converts a TF1 model with TensorRT and runs the converted model."""

  def _create_converter(self, trt_convert_params: trt.TrtConversionParams):
    conversion_nodes_denylist = self.output_tensor_names
    return trt.TrtGraphConverter(
        input_saved_model_dir=self.model_config.saved_model_dir,
        input_saved_model_tags=self.model_config.saved_model_tags,
        input_saved_model_signature_key=(
            self.model_config.saved_model_signature_key),
        nodes_denylist=conversion_nodes_denylist,
        max_workspace_size_bytes=trt_convert_params.max_workspace_size_bytes,
        precision_mode=trt_convert_params.precision_mode,
        minimum_segment_size=trt_convert_params.minimum_segment_size,
        maximum_cached_engines=trt_convert_params.maximum_cached_engines,
        use_calibration=trt_convert_params.use_calibration,
        max_batch_size=self.model_config.default_batch_size,
        is_dynamic_op=False,
    )

  _check_conversion = _TrtModelHandlerBase._check_contains_trt_engine

  def convert(self,
              calibration_inputs: Optional[Mapping[str, np.ndarray]] = None,
              num_runs=1) -> None:
    logging.info("Converting with TensorRT!")
    self._check_conversion(self._converter.convert())

    if (self.trt_convert_params.precision_mode == trt.TrtPrecisionMode.INT8 and
        self.trt_convert_params.use_calibration):
      logging.info("Calibrating with TensorRT!")
      if not calibration_inputs:
        raise ValueError("Must provide calibration data "
                         "when using TensorRT calibration!")
      try:
        self._converter.calibrate(
            fetch_names=self.output_tensor_names,
            num_runs=num_runs,
            feed_dict_fn=lambda: calibration_inputs)
      except Exception as exc:
        raise RuntimeError("Failed to calibrate! "
                           "Model Information: {}".format(str(self))) from exc

  def run(self,
          inputs: Optional[Mapping[str, np.ndarray]] = None,
          warmup_iterations=10,
          benchmark_iterations=100) -> TestResult:
    self.save(overwrite=False)
    self._check_conversion(self.meta_graph.graph_def)
    logging.info("Running with TensorRT!")
    test_result = ModelHandlerV1.run(
        self, inputs, warmup_iterations, benchmark_iterations, enable_gpu=True)
    return test_result._replace(trt_convert_params=self._trt_convert_params)


class TrtModelHandlerV2(_TrtModelHandlerBase, ModelHandlerV2):
  """Converts a TF2 model with TensorRT and runs the converted model."""

  def _create_converter(self, trt_convert_params: trt.TrtConversionParams):
    return trt.TrtGraphConverterV2(
        input_saved_model_dir=self.model_config.saved_model_dir,
        input_saved_model_tags=self.model_config.saved_model_tags,
        input_saved_model_signature_key=(
            self.model_config.saved_model_signature_key),
        **trt_convert_params._asdict())

  def _check_conversion(self, graph_func):
    graph_def = graph_func.graph.as_graph_def()
    self._check_contains_trt_engine(graph_def)

  def convert(self,
              calibration_inputs: Optional[Sequence[
                  framework_ops.Tensor]] = None,
              num_runs=1) -> None:
    logging.info("Converting with TensorRT!")

    calibration_input_fn = None
    if (self.trt_convert_params.precision_mode == trt.TrtPrecisionMode.INT8 and
        self.trt_convert_params.use_calibration):
      logging.info("Calibrating with TensorRT at the same time!")
      if not calibration_inputs:
        raise ValueError("Must provide calibration data "
                         "when using TensorRT calibration!")

      def gets_calibration_input():
        for _ in range(num_runs):
          yield calibration_inputs

      calibration_input_fn = gets_calibration_input

    self._check_conversion(self._converter.convert(calibration_input_fn))

  def run(self,
          inputs: Optional[Sequence[framework_ops.Tensor]] = None,
          warmup_iterations=10,
          benchmark_iterations=100) -> TestResult:
    self.save(overwrite=False)
    self._check_conversion(self.graph_func)
    logging.info("Running with TensorRT!")
    test_result = ModelHandlerV2.run(
        self, inputs, warmup_iterations, benchmark_iterations, enable_gpu=True)
    return test_result._replace(trt_convert_params=self._trt_convert_params)


class _ModelHandlerManagerBase(metaclass=abc.ABCMeta):
  """Manages a series of ModelHandlers for aggregrated testing/benchmarking."""

  def __init__(
      self, name: str, model_config: ModelConfig,
      default_trt_convert_params: trt.TrtConversionParams,
      trt_convert_params_updater: Callable[[trt.TrtConversionParams],
                                           Iterable[trt.TrtConversionParams]]):
    self._ori_model = self.model_handler_cls(model_config)
    self._trt_models = []
    for trt_convert_params in trt_convert_params_updater(
        default_trt_convert_params):
      trt_model = self.trt_model_handler_cls(
          model_config, trt_convert_params=trt_convert_params)
      self._trt_models.append(trt_model)

    self._name = name
    self._result_collection = None

  def __str__(self) -> str:
    return "Input Model: {}".format(str(self._ori_model))

  def __repr__(self) -> str:
    return "{}({})".format(self.__class__.__name__, str(self))

  @property
  @classmethod
  @abc.abstractmethod
  def model_handler_cls(cls):
    """The modle handler class. ModelHandleV1/ModelHandlerV2."""

  @property
  @classmethod
  @abc.abstractmethod
  def trt_model_handler_cls(cls):
    """The TensorRTmodle handler class. TrtModelHandleV1/TrtModelHandlerV2."""

  @property
  def name(self) -> str:
    return self._name

  @property
  def model_config(self) -> ModelConfig:
    return self._ori_model.model_config

  def generate_random_inputs(self, batch_size: Optional[int] = None):
    return self._ori_model.generate_random_inputs(batch_size)

  def convert(self, calibration_inputs=None, num_runs=1) -> None:
    """Converts models with TensorRT and calibrates if using INT8 precision mode.

    Args:
      calibration_inputs: Mapping from input names to ndarrays in TF1. Or a
        sequence of tensors in TF2. Used as calibration data.
      num_runs: Number of calibration runs.
    """
    for trt_model in self._trt_models:
      trt_model.convert(calibration_inputs, num_runs)

  def run(self,
          inputs=None,
          warmup_iterations: int = 10,
          benchmark_iterations: int = 100) -> TestResultCollection:
    """Runs model inference with provided or randomly generated input tensors.

    Args:
      inputs: Mapping from names to input ndarrays in TF1. Or a sequence of
        tensors in TF2. If `None`, ramdomly generated input tensors will be used
        instead.
      warmup_iterations: Number of inferences to warm up the runtime.
      benchmark_iterations: Number of inferences to measure the latency.

    Returns:
      `TestResultCollection` summarizing latency and numerics information for
      different TensorRT conversion settings.
    """
    inputs = inputs or self.generate_random_inputs()

    def run_model(model, **kwargs):
      return model.run(inputs, warmup_iterations, benchmark_iterations,
                       **kwargs)

    # Some models include operations that can only run on GPU.
    try:
      cpu_base_result = run_model(self._ori_model, enable_gpu=False)
    except RuntimeError as err:
      logging.info("%s cannot run on CPU. Reason: %s.",
                   self._ori_model.model_config, err)
      cpu_base_result = None
    gpu_base_result = run_model(self._ori_model, enable_gpu=True)
    trt_results = list(map(run_model, self._trt_models))

    return TestResultCollection(
        test_name=self._name,
        model_config=self.model_config,
        cpu_base_result=cpu_base_result,
        gpu_base_result=gpu_base_result,
        trt_results=trt_results)


class ModelHandlerManagerV1(_ModelHandlerManagerBase):
  """Manages a series of ModelHandlers for aggregrated testing/benchmarking in TF1."""

  model_handler_cls = ModelHandlerV1
  trt_model_handler_cls = TrtModelHandlerV1


class ModelHandlerManagerV2(_ModelHandlerManagerBase):
  """Manages a series of ModelHandlers for aggregrated testing/benchmarking in TF2."""

  model_handler_cls = ModelHandlerV2
  trt_model_handler_cls = TrtModelHandlerV2
