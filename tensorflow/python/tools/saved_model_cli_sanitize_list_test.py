# tensorflow/python/tools/saved_model_cli_sanitize_list_test.py
# Copyright 2025 The TensorFlow Authors.
# Licensed under the Apache License, Version 2.0

"""Unit tests for trim & validation helpers in saved_model_cli."""

import argparse
import collections
import os
import sys
import types
import unittest

from absl.testing import parameterized

_STUBS_INSTALLED = False


def _package(name):
  module = sys.modules.get(name)
  if module is None:
    module = types.ModuleType(name)
    module.__path__ = []
    sys.modules[name] = module
  return module


def _register(name, **attrs):
  module = types.ModuleType(name)
  for key, value in attrs.items():
    setattr(module, key, value)
  sys.modules[name] = module
  return module


def _install_saved_model_cli_stubs():
  global _STUBS_INSTALLED
  if _STUBS_INSTALLED:
    return
  tools_dir = os.path.dirname(__file__)
  if tools_dir not in sys.path:
    sys.path.insert(0, tools_dir)

  _package('tensorflow')
  _package('tensorflow.core')
  example_pkg = _package('tensorflow.core.example')

  class _FeatureValues:

    def __init__(self):
      self.float_list = types.SimpleNamespace(value=[])
      self.bytes_list = types.SimpleNamespace(value=[])
      self.int64_list = types.SimpleNamespace(value=[])

  class _Example:

    def __init__(self):
      feature_factory = lambda: _FeatureValues()
      self.features = types.SimpleNamespace(
          feature=collections.defaultdict(feature_factory))

    def SerializeToString(self):
      return b'serialized'

  example_pkg.example_pb2 = _register(
      'tensorflow.core.example.example_pb2', Example=_Example)

  framework_pkg = _package('tensorflow.core.framework')
  framework_pkg.types_pb2 = _register(
      'tensorflow.core.framework.types_pb2',
      DataType=type('DataType', (), {
          'items': staticmethod(lambda: [])
      }))

  proto_pkg = _package('tensorflow.core.protobuf')
  proto_pkg.saved_model_pb2 = _register(
      'tensorflow.core.protobuf.saved_model_pb2',
      SavedModel=lambda: types.SimpleNamespace(meta_graphs=[]))

  class _ConfigProto:

    def __init__(self, experimental=None):
      self.experimental = experimental

  proto_pkg.config_pb2 = _register(
      'tensorflow.core.protobuf.config_pb2', ConfigProto=_ConfigProto)

  _package('tensorflow.python')
  tools_pkg = _package('tensorflow.python.tools')
  if tools_dir not in tools_pkg.__path__:
    tools_pkg.__path__.append(tools_dir)

  tools_pkg.saved_model_aot_compile = _register(
      'tensorflow.python.tools.saved_model_aot_compile',
      convert=lambda *args, **kwargs: None)

  tools_pkg.saved_model_utils = _register(
      'tensorflow.python.tools.saved_model_utils',
      get_saved_model_tag_sets=lambda *args, **kwargs: [],
      get_meta_graph_def=lambda *args, **kwargs: types.SimpleNamespace(),
      read_saved_model=lambda *args, **kwargs: types.SimpleNamespace(
          meta_graphs=[]))

  util_pkg = _package('tensorflow.python.util')

  def _as_bytes(value):
    return value.encode('utf-8') if isinstance(value, str) else value

  compat_mod = _register(
      'tensorflow.python.util.compat',
      as_bytes=_as_bytes,
      as_str_any=str,
      collections_abc=collections.abc,
      integral_types=(int,))
  util_pkg.compat = compat_mod

  lib_pkg = _package('tensorflow.python.lib')
  io_pkg = _package('tensorflow.python.lib.io')

  class _StubFileIO(object):

    def __init__(self, *args, **kwargs):
      raise IOError('File IO stub')

  file_io_mod = _register(
      'tensorflow.python.lib.io.file_io',
      file_exists=lambda path: False,
      FileIO=_StubFileIO)
  io_pkg.file_io = file_io_mod

  saved_model_pkg = _package('tensorflow.python.saved_model')
  saved_model_pkg.constants = _register(
      'tensorflow.python.saved_model.constants',
      SAVED_MODEL_FILENAME_PB='saved_model.pb',
      SAVED_MODEL_FILENAME_PBTXT='saved_model.pbtxt')
  saved_model_pkg.loader = _register(
      'tensorflow.python.saved_model.loader',
      load=lambda *args, **kwargs: None)
  saved_model_pkg.load = _register(
      'tensorflow.python.saved_model.load',
      load=lambda *args, **kwargs: None)
  saved_model_pkg.load_options = _register(
      'tensorflow.python.saved_model.load_options',
      LoadOptions=lambda **kwargs: types.SimpleNamespace(**kwargs))
  saved_model_pkg.save = _register(
      'tensorflow.python.saved_model.save',
      save=lambda *args, **kwargs: None)
  saved_model_pkg.signature_constants = _register(
      'tensorflow.python.saved_model.signature_constants',
      DEFAULT_SERVING_SIGNATURE_DEF_KEY='serving_default')

  tpu_pkg = _package('tensorflow.python.tpu')
  tpu_pkg.tpu = _register(
      'tensorflow.python.tpu.tpu',
      initialize_system=lambda: None)

  debug_pkg = _package('tensorflow.python.debug')
  wrappers_pkg = _package('tensorflow.python.debug.wrappers')
  debug_pkg.wrappers = wrappers_pkg
  wrappers_pkg.local_cli_wrapper = _register(
      'tensorflow.python.debug.wrappers.local_cli_wrapper',
      LocalCLIDebugWrapperSession=lambda sess: sess)

  client_pkg = _package('tensorflow.python.client')

  class _Session(object):

    def __init__(self, *args, **kwargs):
      pass

    def __enter__(self):
      return self

    def __exit__(self, exc_type, exc, tb):
      return False

    def run(self, *args, **kwargs):
      return []

  client_pkg.session = _register(
      'tensorflow.python.client.session', Session=_Session)

  eager_pkg = _package('tensorflow.python.eager')
  eager_pkg.def_function = _register(
      'tensorflow.python.eager.def_function',
      function=lambda fn: fn)
  eager_pkg.function = _register(
      'tensorflow.python.eager.function', __doc__='')

  framework_pkg = _package('tensorflow.python.framework')
  framework_pkg.meta_graph = _register(
      'tensorflow.python.framework.meta_graph',
      ops_used_by_graph_def=lambda graph_def: set())
  framework_pkg.ops = _register(
      'tensorflow.python.framework.ops',
      Graph=type('Graph', (), {}))
  framework_pkg.tensor_spec = _register(
      'tensorflow.python.framework.tensor_spec',
      TensorSpec=type('TensorSpec', (), {}))

  platform_pkg = _package('tensorflow.python.platform')
  platform_pkg.tf_logging = _register(
      'tensorflow.python.platform.tf_logging',
      info=lambda *args, **kwargs: None,
      warning=lambda *args, **kwargs: None)

  class _DummyNpz(dict):
    files = ()

  numpy_lib = types.SimpleNamespace(
      npyio=types.SimpleNamespace(NpzFile=_DummyNpz))
  _register(
      'numpy',
      load=lambda *args, **kwargs: _DummyNpz(),
      save=lambda *args, **kwargs: None,
      asarray=lambda value, *args, **kwargs: value,
      ndarray=type('ndarray', (), {}),
      lib=numpy_lib)


  try:
    import absl.flags as absl_flags
  except ImportError:
    absl_pkg = _package('absl')
    absl_flags = _register('absl.flags')
    absl_pkg.flags = absl_flags
  argparse_flags_mod = _register(
      'absl.flags.argparse_flags',
      ArgumentParser=argparse.ArgumentParser)
  absl_flags.argparse_flags = argparse_flags_mod

  _STUBS_INSTALLED = True


_install_saved_model_cli_stubs()
from tensorflow.python.tools import saved_model_cli as smcli


class SanitizeNonEmptyStrListTest(parameterized.TestCase):

  def test_trims_and_drops_empty_and_none(self):
    self.assertEqual(
        smcli._sanitize_nonempty_str_list(
            [' a ', '', '\t', None, 'b '], 'field'),
        ['a', 'b'])

  def test_raises_on_all_empty_like_inputs(self):
    with self.assertRaisesRegex(ValueError, 'field.*at least one non-empty'):
      smcli._sanitize_nonempty_str_list(['  ', '\n', '', None], 'field')


class DenylistParsingTest(parameterized.TestCase):

  @parameterized.parameters(
      ('OpA, OpB, , ,,OpC', {'OpA', 'OpB', 'OpC'}),
      ('  , , ', set()),
      ('', set()),
      (' ReadFile ,WriteFile , PrintV2',
       {'ReadFile', 'WriteFile', 'PrintV2'}),
  )
  def test_get_op_denylist_set_trims_and_ignores_empties(self, raw, expected):
    self.assertEqual(smcli._get_op_denylist_set(raw), expected)


if __name__ == '__main__':
  unittest.main()
