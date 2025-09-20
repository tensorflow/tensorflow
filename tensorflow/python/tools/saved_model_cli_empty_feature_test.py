# pylint: disable=line-too-long
import importlib
import sys
import types
import collections
import unittest


_STUBS_INSTALLED = False


def _ensure_package(name, path=None):
  module = sys.modules.get(name)
  if module is None:
    module = types.ModuleType(name)
    module.__path__ = []
    sys.modules[name] = module
  if path and path not in module.__path__:
    module.__path__.append(path)
  return module


def _install_saved_model_cli_stubs():
  global _STUBS_INSTALLED
  if _STUBS_INSTALLED:
    return
  sys.path.insert(0, '.')
  collections_abc = __import__('collections').abc
  argparse = __import__('argparse')

  _ensure_package('tensorflow')
  _ensure_package('tensorflow.python')
  _ensure_package('tensorflow.python.tools', 'tensorflow/python/tools')
  _ensure_package('tensorflow.python.util')
  compat_mod = types.ModuleType('tensorflow.python.util.compat')
  compat_mod.as_bytes = lambda x: x.encode('utf-8') if isinstance(x, str) else x
  compat_mod.as_str_any = lambda x: str(x)
  compat_mod.collections_abc = collections_abc
  compat_mod.integral_types = int
  sys.modules['tensorflow.python.util.compat'] = compat_mod

  _ensure_package('tensorflow.python.lib')
  _ensure_package('tensorflow.python.lib.io')
  file_io_mod = types.ModuleType('tensorflow.python.lib.io.file_io')
  file_io_mod.file_exists = lambda path: False
  class _StubFile(object):
    def __init__(self, *args, **kwargs):
      raise IOError('File IO stub')
  file_io_mod.FileIO = _StubFile
  sys.modules['tensorflow.python.lib.io.file_io'] = file_io_mod

  _ensure_package('tensorflow.python.saved_model')
  constants_mod = types.ModuleType('tensorflow.python.saved_model.constants')
  constants_mod.SAVED_MODEL_FILENAME_PBTXT = 'saved_model.pbtxt'
  constants_mod.SAVED_MODEL_FILENAME_PB = 'saved_model.pb'
  sys.modules['tensorflow.python.saved_model.constants'] = constants_mod

  loader_mod = types.ModuleType('tensorflow.python.saved_model.loader')
  loader_mod.load = lambda *args, **kwargs: None
  sys.modules['tensorflow.python.saved_model.loader'] = loader_mod

  load_mod = types.ModuleType('tensorflow.python.saved_model.load')
  load_mod.load = lambda *args, **kwargs: None
  sys.modules['tensorflow.python.saved_model.load'] = load_mod

  load_options_mod = types.ModuleType('tensorflow.python.saved_model.load_options')
  load_options_mod.LoadOptions = lambda **kwargs: types.SimpleNamespace(**kwargs)
  sys.modules['tensorflow.python.saved_model.load_options'] = load_options_mod

  save_mod = types.ModuleType('tensorflow.python.saved_model.save')
  save_mod.save = lambda *args, **kwargs: None
  sys.modules['tensorflow.python.saved_model.save'] = save_mod

  signature_constants_mod = types.ModuleType('tensorflow.python.saved_model.signature_constants')
  signature_constants_mod.DEFAULT_SERVING_SIGNATURE_DEF_KEY = 'serving_default'
  sys.modules['tensorflow.python.saved_model.signature_constants'] = signature_constants_mod

  debug_pkg = _ensure_package('tensorflow.python.debug')
  wrappers_pkg = _ensure_package('tensorflow.python.debug.wrappers')
  local_wrapper_mod = types.ModuleType('tensorflow.python.debug.wrappers.local_cli_wrapper')
  local_wrapper_mod.LocalCLIDebugWrapperSession = lambda sess: sess
  sys.modules['tensorflow.python.debug.wrappers.local_cli_wrapper'] = local_wrapper_mod
  wrappers_pkg.local_cli_wrapper = local_wrapper_mod
  debug_pkg.wrappers = wrappers_pkg

  saved_aot_mod = types.ModuleType('tensorflow.python.tools.saved_model_aot_compile')
  saved_aot_mod.convert = lambda *args, **kwargs: None
  sys.modules['tensorflow.python.tools.saved_model_aot_compile'] = saved_aot_mod

  numpy_mod = types.ModuleType('numpy')
  class _DummyNpz(dict):
    files = ()
    def __contains__(self, key):
      return False
  numpy_mod.load = lambda *args, **kwargs: _DummyNpz()
  numpy_mod.save = lambda *args, **kwargs: None
  numpy_mod.asarray = lambda x, *args, **kwargs: x
  numpy_mod.ndarray = type('ndarray', (), {})
  numpy_mod.lib = types.SimpleNamespace(npyio=types.SimpleNamespace(NpzFile=_DummyNpz))
  sys.modules['numpy'] = numpy_mod

  _ensure_package('tensorflow.core')
  _ensure_package('tensorflow.core.example')

  class _FeatureNamespace:
    def __init__(self):
      self.float_list = types.SimpleNamespace(value=[])
      self.bytes_list = types.SimpleNamespace(value=[])
      self.int64_list = types.SimpleNamespace(value=[])

  class _Example:
    def __init__(self):
      self.features = types.SimpleNamespace(feature=collections.defaultdict(_FeatureNamespace))
    def SerializeToString(self):
      return b'serialized'

  example_mod = types.ModuleType('tensorflow.core.example.example_pb2')
  example_mod.Example = _Example
  sys.modules['tensorflow.core.example.example_pb2'] = example_mod

  _ensure_package('tensorflow.core.framework')
  types_mod = types.ModuleType('tensorflow.core.framework.types_pb2')
  class _DataType:
    @staticmethod
    def items():
      return []
  types_mod.DataType = _DataType
  sys.modules['tensorflow.core.framework.types_pb2'] = types_mod

  _ensure_package('tensorflow.core.protobuf')
  saved_model_pb2_mod = types.ModuleType('tensorflow.core.protobuf.saved_model_pb2')
  saved_model_pb2_mod.SavedModel = lambda: types.SimpleNamespace(meta_graphs=[])
  sys.modules['tensorflow.core.protobuf.saved_model_pb2'] = saved_model_pb2_mod

  config_pb2_mod = types.ModuleType('tensorflow.core.protobuf.config_pb2')
  class _Experimental:
    def __init__(self, **kwargs):
      for key, value in kwargs.items():
        setattr(self, key, value)
  class _ConfigProto:
    class Experimental(_Experimental):
      pass
    def __init__(self, experimental=None):
      self.experimental = experimental
  config_pb2_mod.ConfigProto = _ConfigProto
  sys.modules['tensorflow.core.protobuf.config_pb2'] = config_pb2_mod

  try:
    import google.protobuf.message  # pylint: disable=unused-import
    import google.protobuf.text_format  # pylint: disable=unused-import
  except Exception:  # pragma: no cover
    _ensure_package('google')
    _ensure_package('google.protobuf')
    message_mod = types.ModuleType('google.protobuf.message')
    class DecodeError(Exception):
      pass
    message_mod.DecodeError = DecodeError
    sys.modules['google.protobuf.message'] = message_mod
    text_mod = types.ModuleType('google.protobuf.text_format')
    class ParseError(Exception):
      pass
    text_mod.ParseError = ParseError
    sys.modules['google.protobuf.text_format'] = text_mod

  platform_pkg = _ensure_package('tensorflow.python.platform')
  tf_logging = types.ModuleType('tensorflow.python.platform.tf_logging')
  tf_logging.info = lambda *args, **kwargs: None
  tf_logging.warning = lambda *args, **kwargs: None
  sys.modules['tensorflow.python.platform.tf_logging'] = tf_logging
  platform_pkg.tf_logging = tf_logging

  client_pkg = _ensure_package('tensorflow.python.client')
  session_mod = types.ModuleType('tensorflow.python.client.session')
  class _Session:
    def __init__(self, *args, **kwargs):
      pass
    def __enter__(self):
      return self
    def __exit__(self, exc_type, exc, tb):
      return False
    def run(self, *args, **kwargs):
      return []
  session_mod.Session = _Session
  sys.modules['tensorflow.python.client.session'] = session_mod
  client_pkg.session = session_mod

  debug_wrapper = types.ModuleType('tensorflow.python.debug.wrappers.local_cli_wrapper')
  debug_wrapper.LocalCLIDebugWrapperSession = lambda sess: sess
  sys.modules['tensorflow.python.debug.wrappers.local_cli_wrapper'] = debug_wrapper

  eager_pkg = _ensure_package('tensorflow.python.eager')
  def_function_mod = types.ModuleType('tensorflow.python.eager.def_function')
  def_function_mod.function = lambda f: f
  sys.modules['tensorflow.python.eager.def_function'] = def_function_mod
  defun_mod = types.ModuleType('tensorflow.python.eager.function')
  sys.modules['tensorflow.python.eager.function'] = defun_mod
  eager_pkg.def_function = def_function_mod
  eager_pkg.function = defun_mod

  framework_pkg = _ensure_package('tensorflow.python.framework')
  meta_graph_mod = types.ModuleType('tensorflow.python.framework.meta_graph')
  meta_graph_mod.ops_used_by_graph_def = lambda graph_def: set()
  sys.modules['tensorflow.python.framework.meta_graph'] = meta_graph_mod
  framework_pkg.meta_graph = meta_graph_mod
  ops_mod = types.ModuleType('tensorflow.python.framework.ops')
  ops_mod.Graph = type('Graph', (), {})
  sys.modules['tensorflow.python.framework.ops'] = ops_mod
  framework_pkg.ops = ops_mod
  tensor_spec_mod = types.ModuleType('tensorflow.python.framework.tensor_spec')
  tensor_spec_mod.TensorSpec = type('TensorSpec', (), {})
  sys.modules['tensorflow.python.framework.tensor_spec'] = tensor_spec_mod
  framework_pkg.tensor_spec = tensor_spec_mod

  tpu_pkg = _ensure_package('tensorflow.python.tpu')
  tpu_mod = types.ModuleType('tensorflow.python.tpu.tpu')
  tpu_mod.initialize_system = lambda: None
  sys.modules['tensorflow.python.tpu.tpu'] = tpu_mod
  tpu_pkg.tpu = tpu_mod

  absl_mod = types.ModuleType('absl')
  sys.modules['absl'] = absl_mod
  app_mod = types.ModuleType('absl.app')
  app_mod.run = lambda *args, **kwargs: None
  sys.modules['absl.app'] = app_mod
  flags_mod = types.ModuleType('absl.flags')
  class _Flag:
    def __init__(self, default):
      self.value = default
  def _define(default):
    def _inner(*args, **kwargs):
      return _Flag(default)
    return _inner
  flags_mod.DEFINE_string = _define('')
  flags_mod.DEFINE_bool = _define(False)
  flags_mod.DEFINE_integer = _define(0)
  flags_mod.DEFINE_enum = _define('')
  flags_mod.FLAGS = types.SimpleNamespace()
  sys.modules['absl.flags'] = flags_mod
  argparse_flags_mod = types.ModuleType('absl.flags.argparse_flags')
  argparse_flags_mod.ArgumentParser = argparse.ArgumentParser
  sys.modules['absl.flags.argparse_flags'] = argparse_flags_mod
  absl_mod.app = app_mod
  absl_mod.flags = flags_mod

  _STUBS_INSTALLED = True


class SavedModelCliExampleEmptyFeatureTest(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    _install_saved_model_cli_stubs()
    cls.saved_model_cli = importlib.import_module('tensorflow.python.tools.saved_model_cli')

  def test_create_example_string_rejects_empty_feature_list(self):
    with self.assertRaisesRegex(ValueError, 'must contain at least one value'):
      self.saved_model_cli._create_example_string({'ids': []})

  def test_create_example_string_accepts_int_values(self):
    payload = self.saved_model_cli._create_example_string({'ids': [1, 2, 3]})
    self.assertIsInstance(payload, bytes)


if __name__ == '__main__':
  unittest.main()
