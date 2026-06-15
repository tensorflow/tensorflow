# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
#
# ==============================================================================
"""A visitor class that generates protobufs for each python object."""

import enum
import functools
import inspect
import re

from google.protobuf import message
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import deprecation
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect
from tensorflow.tools.api.lib import api_objects_pb2

# Following object need to be handled individually.
_CORNER_CASES = {
    '': {
        'tools': {}
    },
    'test.TestCase': {},
    'test.TestCase.failureException': {},
    'train.NanLossDuringTrainingError': {
        'message': {}
    },
    'estimator.NanLossDuringTrainingError': {
        'message': {}
    },
    'train.LooperThread': {
        'isAlive': {},
        'join': {},
        'native_id': {}
    }
}

_NORMALIZE_TYPE = {
    # Keep Union aliases stable across Python versions with different
    # implementation types for typing.Union.
    "<class 'typing._UnionGenericAlias'>": 'typing.Union',
    "<class 'typing.Union'>": 'typing.Union',
    "<class 'enum.EnumMeta'>": "<class 'enum.EnumType'>",
}
_NORMALIZE_ISINSTANCE = {
    "<class "
    "'tensorflow.lite.python.op_hint.OpHint.OpHintArgumentTracker'>":  # pylint: disable=line-too-long
        "<class "
        "'tensorflow.lite.python.op_hint.OpHintArgumentTracker'>",
    "<class "
    "'tensorflow.python.training.monitored_session._MonitoredSession.StepContext'>":  # pylint: disable=line-too-long
        "<class "
        "'tensorflow.python.training.monitored_session.StepContext'>",
    "<class "
    "'tensorflow.python.ops.variables.Variable.SaveSliceInfo'>":
        '<class '
        "'tensorflow.python.ops.variables.SaveSliceInfo'>"
}


def _SkipMember(cls, member):
  return member == 'with_traceback' or (
      member in ('name', 'value')
      and isinstance(cls, type)
      and issubclass(cls, enum.Enum)
  )


# CPython regularly adds public members to these bases. Inherited members from
# them are runtime details rather than TensorFlow API.
_SIGNATURE_CLASS = getattr(inspect, 'Signature', None)
_UNSTABLE_EXTERNAL_BASE_CLASSES = tuple(
    cls
    for cls in (BaseException, enum.Enum, int, _SIGNATURE_CLASS)
    if cls is not None
)
_TENSORFLOW_FAMILY_MARKERS = ('tensorflow', 'tf_keras', 'keras')
_OWNERSHIP_IDENTIFIER_RE = re.compile(r'[A-Za-z_][A-Za-z0-9_]*')


def _NormalizeType(ty):
  return _NORMALIZE_TYPE.get(ty, ty)


def _NormalizeIsInstance(ty):
  return _NORMALIZE_ISINSTANCE.get(ty, ty)


def _IsApiMethod(obj):
  """Return whether obj should be serialized as a proto method."""
  # Keep callable wrapper objects in the member field across Python versions.
  if isinstance(obj, functools.partial):
    return False
  return tf_inspect.isroutine(obj)


def _IsTensorFlowFamilySegment(segment):
  return segment in _TENSORFLOW_FAMILY_MARKERS or segment.startswith(
      'tensorflow_'
  )


def _HasTensorFlowFamilyMarker(value):
  if not value:
    return False
  return any(
      _IsTensorFlowFamilySegment(segment)
      for segment in _OWNERSHIP_IDENTIFIER_RE.findall(str(value))
  )


def _ClassOwnershipEvidence(cls):
  module = getattr(cls, '__module__', '')
  if module:
    yield module

  try:
    inspected_module = inspect.getmodule(cls)
  except TypeError:
    inspected_module = None
  inspected_module_name = getattr(inspected_module, '__name__', '')
  if inspected_module_name and inspected_module_name != module:
    yield inspected_module_name

  yield str(cls)
  yield repr(cls)


def _IsTensorFlowOwnedClass(cls):
  return any(
      _HasTensorFlowFamilyMarker(value)
      for value in _ClassOwnershipEvidence(cls)
  )


def _IsUnstableExternalBase(cls):
  if _IsTensorFlowOwnedClass(cls):
    return False
  return any(issubclass(cls, base) for base in _UNSTABLE_EXTERNAL_BASE_CLASSES)


def _OwnerClass(cls, member):
  try:
    mro = tf_inspect.getmro(cls)
  except TypeError:
    return None

  for base in mro:
    if member in getattr(base, '__dict__', ()):
      return base
  return None


def _IsUnstableExternalInheritedMember(cls, member):
  if not tf_inspect.isclass(cls):
    return False
  owner = _OwnerClass(cls, member)
  if owner is None or owner is cls:
    return False
  if owner is object and member == '__init__':
    return any(
        issubclass(cls, base) for base in _UNSTABLE_EXTERNAL_BASE_CLASSES
    )
  return _IsUnstableExternalBase(owner)


def _SanitizedArgSpec(obj):
  """Get an ArgSpec string that is free of addresses.

  We have callables as function arg defaults. This results in addresses in
  getargspec output. This function returns a sanitized string list of base
  classes.

  Args:
    obj: A python routine for us the create the sanitized arspec of.

  Returns:
    string, a string representation of the argspec.
  """
  output_string = ''
  unsanitized_arg_spec = tf_inspect.getargspec(obj)

  for clean_attr in ('args', 'varargs', 'keywords'):
    output_string += '%s=%s, ' % (clean_attr,
                                  getattr(unsanitized_arg_spec, clean_attr))

  if unsanitized_arg_spec.defaults:
    sanitized_defaults = []
    for val in unsanitized_arg_spec.defaults:
      str_val = str(val)
      # Sanitize argspecs that have hex code in them.
      if ' at 0x' in str_val:
        sanitized_defaults.append('%s instance>' % str_val.split(' at ')[0])
      else:
        sanitized_defaults.append(str_val)

    output_string += 'defaults=%s, ' % sanitized_defaults

  else:
    output_string += 'defaults=None'

  return output_string


def _GenerateArgsSpec(doc):
  """Generate args spec from a method docstring."""
  args_spec = []
  doc = re.search(r'\(.*\)', doc)
  if not doc:
    return None
  # remove parentheses
  doc = doc.group().strip('(').strip(')')
  doc_split = doc.split(',')
  for s in doc_split:
    arg = re.search(r'\w+', s)
    if not arg:
      return None
    args_spec.append(f'\'{arg.group()}\'')
  return ', '.join(args_spec)


def _ParseDocstringArgSpec(doc):
  """Get an ArgSpec string from a method docstring.

  This method is used to generate argspec for C extension functions that follow
  pybind11 DocString format function signature. For example:
  `foo_function(a: int, b: string) -> None...`

  Args:
    doc: A python string which starts with function signature.

  Returns:
    string: a argspec string representation if successful. If not, return None.

  Raises:
    ValueError: Raised when failed to parse the input docstring.
  """
  # Check if the docstring begins with a function signature
  match = re.search(r'^\w+\(.*\)', doc)
  args_spec = _GenerateArgsSpec(doc)
  if (not match) or (args_spec is None):
    raise ValueError(f'Failed to parse argspec from docstring: {doc}')

  # TODO(panzf): implement parsing docs with varargs, keywords, and defaults
  output_string = (
      f'args=[{args_spec}], varargs=None, keywords=None, defaults=None')
  return output_string


def _SanitizedMRO(obj):
  """Get a list of superclasses with minimal amount of non-TF classes.

  Based on many parameters like python version, OS, protobuf implementation
  or changes in google core libraries the list of superclasses of a class
  can change. We only return the first non-TF class to be robust to non API
  affecting changes. The Method Resolution Order returned by `tf_inspect.getmro`
  is still maintained in the return value.

  Args:
    obj: A python routine for us the create the sanitized arspec of.

  Returns:
    list of strings, string representation of the class names.
  """
  return_list = []
  for cls in tf_inspect.getmro(obj):
    if cls.__name__ == '_NewClass':
      # Ignore class created by @deprecated_alias decorator.
      continue
    str_repr = _NormalizeType(str(cls))
    return_list.append(str_repr)
    # Class type that has keras in their name should also be monitored. This
    # will cover any class imported from third_party/py/keras.
    if 'tensorflow' not in str_repr and 'keras' not in str_repr:
      break

    # Hack - tensorflow.test.StubOutForTesting may or may not be type <object>
    # depending on the environment. To avoid inconsistency, break after we add
    # StubOutForTesting to the return_list.
    if 'StubOutForTesting' in str_repr:
      break

  return return_list


def _IsProtoClass(obj):
  """Returns whether the passed obj is a Protocol Buffer class."""
  return isinstance(obj, type) and issubclass(obj, message.Message)


class PythonObjectToProtoVisitor:
  """A visitor that summarizes given python objects as protobufs."""

  def __init__(self, default_path='tensorflow'):
    # A dict to store all protocol buffers.
    # Keyed by "path" to the object.
    self._protos = {}
    self._default_path = default_path

  def GetProtos(self):
    """Return the list of protos stored."""
    return self._protos

  def __call__(self, path, parent, children):
    # The path to the object.
    lib_path = self._default_path + '.' + path if path else self._default_path
    _, parent = tf_decorator.unwrap(parent)

    if tf_inspect.isclass(parent):
      children[:] = [
          (name, child)
          for name, child in children
          if not _IsUnstableExternalInheritedMember(parent, name)
      ]

    # A small helper method to construct members(children) protos.
    def _AddMember(member_name, member_obj, proto):
      """Add the child object to the object being constructed."""
      _, member_obj = tf_decorator.unwrap(member_obj)
      if _SkipMember(parent, member_name) or isinstance(
          member_obj, deprecation.HiddenTfApiAttribute
      ):
        return
      is_tuple_subclass = isinstance(parent, type) and issubclass(parent, tuple)
      is_allowed_dunder = member_name == '__init__' or (
          member_name == '__new__' and is_tuple_subclass
      )
      if is_allowed_dunder or not member_name.startswith('_'):
        if _IsApiMethod(member_obj):
          new_method = proto.member_method.add()
          new_method.name = member_name
          # If member_obj is a python builtin, there is no way to get its
          # argspec, because it is implemented on the C side. It also has no
          # func_code.
          if hasattr(member_obj, '__code__'):
            new_method.argspec = _SanitizedArgSpec(member_obj)
          else:
            # Try to parse argspec based on docstring for exposed C++ functions
            if member_name != '__init__' and hasattr(member_obj, '__doc__'):
              doc = member_obj.__doc__
              try:
                spec_str = _ParseDocstringArgSpec(doc)
              except ValueError:
                pass
              else:
                new_method.argspec = spec_str
        else:
          new_member = proto.member.add()
          new_member.name = member_name
          if tf_inspect.ismodule(member_obj):
            new_member.mtype = "<class 'module'>"
          else:
            new_member.mtype = _NormalizeType(str(type(member_obj)))

    parent_corner_cases = _CORNER_CASES.get(path, {})

    if path not in _CORNER_CASES or parent_corner_cases:
      # Decide if we have a module or a class.
      if tf_inspect.ismodule(parent):
        # Create a module object.
        module_obj = api_objects_pb2.TFAPIModule()
        for name, child in children:
          if name in parent_corner_cases:
            # If we have an empty entry, skip this object.
            if parent_corner_cases[name]:
              module_obj.member.add(**(parent_corner_cases[name]))
          else:
            _AddMember(name, child, module_obj)

        # Store the constructed module object.
        self._protos[lib_path] = api_objects_pb2.TFAPIObject(
            path=lib_path, tf_module=module_obj)
      elif _IsProtoClass(parent):
        proto_obj = api_objects_pb2.TFAPIProto()
        parent.DESCRIPTOR.CopyToProto(proto_obj.descriptor)

        # Store the constructed proto object.
        self._protos[lib_path] = api_objects_pb2.TFAPIObject(
            path=lib_path, tf_proto=proto_obj)
      elif tf_inspect.isclass(parent):
        # Construct a class.
        class_obj = api_objects_pb2.TFAPIClass()
        class_obj.is_instance.extend(
            _NormalizeIsInstance(i) for i in _SanitizedMRO(parent))
        for name, child in children:
          if name in parent_corner_cases:
            # If we have an empty entry, skip this object.
            if parent_corner_cases[name]:
              class_obj.member.add(**(parent_corner_cases[name]))
          else:
            _AddMember(name, child, class_obj)

        # Store the constructed class object.
        self._protos[lib_path] = api_objects_pb2.TFAPIObject(
            path=lib_path, tf_class=class_obj)
      else:
        logging.error(
            'Illegal call to ApiProtoDump::_py_obj_to_proto.'
            'Object is neither a module nor a class: %s', path)
