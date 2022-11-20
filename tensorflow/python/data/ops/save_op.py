# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Implementation of SaveDataset in Python."""
import os

from tensorflow.python.checkpoint import checkpoint as checkpoint_lib
from tensorflow.python.checkpoint import checkpoint_management
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import structured_function
from tensorflow.python.data.util import structure
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_experimental_dataset_ops as ged_ops
from tensorflow.python.platform import gfile
from tensorflow.python.util import lazy_loader

# TODO(b/238903802): Use TypeSpec serialization methods directly.
nested_structure_coder = lazy_loader.LazyLoader(
    "nested_structure_coder", globals(),
    "tensorflow.python.saved_model.nested_structure_coder")


def _save(input_dataset,
          path,
          compression=None,
          shard_func=None,
          checkpoint_args=None):
  """Implements the save function and checkpoint functionality."""
  if context.executing_eagerly() and checkpoint_args:
    save_dataset = _SaveDataset(input_dataset, path, shard_func, compression)
    save_iterator = iter(save_dataset)

    if "checkpoint" in checkpoint_args:
      raise ValueError(
          "'Invalid `checkpoint_args`. `checkpoint_args` are not allowed "
          "to include 'checkpoint'."
      )
    checkpoint = checkpoint_lib.Checkpoint(iterator=save_iterator)
    checkpoint_args["checkpoint"] = checkpoint
    manager = checkpoint_management.CheckpointManager(**checkpoint_args)
    checkpoint.restore(manager.latest_checkpoint)

    for _ in enumerate(save_iterator):
      if "step_counter" in checkpoint_args:
        checkpoint_args["step_counter"].assign_add(delta=1)
      manager.save(check_interval=True)
  else:
    dataset, shard_func, use_shard_func, path = set_save_dataset_attributes(
        input_dataset, shard_func, path)
    ged_ops.save_dataset(
        dataset._variant_tensor,   # pylint: disable=protected-access
        path=path,
        shard_func_other_args=shard_func.captured_inputs,
        compression=compression,
        shard_func=shard_func,
        use_shard_func=use_shard_func)


class _SaveDataset(dataset_ops.UnaryDataset):
  """"A dataset that loads previously saved dataset."""

  def __init__(self, dataset, path, shard_func, compression):
    self._element_spec = dataset.element_spec
    self._shard_func = shard_func
    dataset, shard_func, use_shard_func, path = set_save_dataset_attributes(
        dataset, shard_func, path)
    variant_tensor = ged_ops.save_dataset_v2(
        dataset._variant_tensor,  # pylint: disable=protected-access
        path=path,
        shard_func_other_args=shard_func.captured_inputs,
        shard_func=shard_func,
        use_shard_func=use_shard_func,
        compression=compression,
        output_types=structure.get_flat_tensor_types(dataset.element_spec),
        output_shapes=structure.get_flat_tensor_shapes(dataset.element_spec),
    )
    super().__init__(dataset, variant_tensor)

  def _functions(self):
    return [self._shard_func]

  @property
  def element_spec(self):
    return self._element_spec


def set_save_dataset_attributes(dataset, shard_func, path):
  """Sets parameters for SaveDatasetOp and SaveDatasetV2Op."""
  if shard_func is None:
    use_shard_func = False
    shard_func = lambda *x: None  # a dummy function that will not be used
  else:
    use_shard_func = True
  wrapped_func = structured_function.StructuredFunctionWrapper(
      shard_func,
      "save()",
      input_structure=dataset.element_spec,
      add_to_graph=False)
  encoded = nested_structure_coder.encode_structure(dataset.element_spec)
  gfile.MakeDirs(path)
  with gfile.GFile(os.path.join(path, dataset_ops.DATASET_SPEC_FILENAME),
                   "wb") as f:
    f.write(encoded.SerializeToString())
  path = ops.convert_to_tensor(path, dtype=dtypes.string, name="path")
  shard_func = wrapped_func.function
  shard_func.add_to_graph(ops.get_default_graph())
  # pylint: disable=protected-access
  dataset._apply_debug_options()
  return dataset, shard_func, use_shard_func, path
