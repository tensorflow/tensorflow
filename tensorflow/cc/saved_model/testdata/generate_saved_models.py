# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Standalone utility to generate some test saved models."""

import os

from absl import app

from tensorflow.python.client import session as session_lib
from tensorflow.python.compat import v2_compat
from tensorflow.python.eager import def_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.module import module
from tensorflow.python.ops import io_ops
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.saved_model import save_options
from tensorflow.python.saved_model import saved_model
from tensorflow.python.training.tracking import tracking


class VarsAndArithmeticObjectGraph(module.Module):
  """Three vars (one in a sub-module) and compute method."""

  def __init__(self):
    self.x = variables.Variable(1.0, name="variable_x")
    self.y = variables.Variable(2.0, name="variable_y")
    self.child = module.Module()
    self.child.z = variables.Variable(3.0, name="child_variable")
    self.child.c = ops.convert_to_tensor(5.0)

  @def_function.function(input_signature=[
      tensor_spec.TensorSpec((), dtypes.float32),
      tensor_spec.TensorSpec((), dtypes.float32)
  ])
  def compute(self, a, b):
    return (a + self.x) * (b + self.y) / (self.child.z) + self.child.c


class ReferencesParent(module.Module):

  def __init__(self, parent):
    super(ReferencesParent, self).__init__()
    self.parent = parent
    self.my_variable = variables.Variable(3., name="MyVariable")


# Creates a cyclic object graph.
class CyclicModule(module.Module):

  def __init__(self):
    super(CyclicModule, self).__init__()
    self.child = ReferencesParent(self)


class AssetModule(module.Module):

  def __init__(self):
    self.asset = tracking.Asset(
        test.test_src_dir_path("cc/saved_model/testdata/test_asset.txt"))

  @def_function.function(input_signature=[])
  def read_file(self):
    return io_ops.read_file(self.asset)


class StaticHashTableModule(module.Module):
  """A module with an Asset, StaticHashTable, and a lookup function."""

  def __init__(self):
    self.asset = tracking.Asset(
        test.test_src_dir_path(
            "cc/saved_model/testdata/static_hashtable_asset.txt"))
    self.table = lookup_ops.StaticHashTable(
        lookup_ops.TextFileInitializer(self.asset, dtypes.string,
                                       lookup_ops.TextFileIndex.WHOLE_LINE,
                                       dtypes.int64,
                                       lookup_ops.TextFileIndex.LINE_NUMBER),
        -1)

  @def_function.function(
      input_signature=[tensor_spec.TensorSpec(shape=None, dtype=dtypes.string)])
  def lookup(self, word):
    return self.table.lookup(word)


def get_simple_session():
  ops.disable_eager_execution()
  sess = session_lib.Session()
  variables.Variable(1.)
  sess.run(variables.global_variables_initializer())
  return sess


MODULE_CTORS = {
    "VarsAndArithmeticObjectGraph": (VarsAndArithmeticObjectGraph, 2),
    "CyclicModule": (CyclicModule, 2),
    "AssetModule": (AssetModule, 2),
    "StaticHashTableModule": (StaticHashTableModule, 2),
    "SimpleV1Model": (get_simple_session, 1)
}


def main(args):
  if len(args) != 3:
    print("Expected: {export_path} {ModuleName}")
    print("Allowed ModuleNames:", MODULE_CTORS.keys())
    return 1

  _, export_path, module_name = args
  module_ctor, version = MODULE_CTORS.get(module_name)
  if not module_ctor:
    print("Expected ModuleName to be one of:", MODULE_CTORS.keys())
    return 2
  os.makedirs(export_path)

  tf_module = module_ctor()
  if version == 2:
    options = save_options.SaveOptions(save_debug_info=True)
    saved_model.save(tf_module, export_path, options=options)
  else:
    builder = saved_model.builder.SavedModelBuilder(export_path)
    builder.add_meta_graph_and_variables(tf_module, ["serve"])
    builder.save()


if __name__ == "__main__":
  v2_compat.enable_v2_behavior()
  app.run(main)
