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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect
from tensorflow.tools.api.lib import api_objects_pb2

# Following object need to be handled individually.
_CORNER_CASES = {
    '': {'tools': {}},
    'test.TestCase': {},
    'test.TestCase.failureException': {},
}


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
    str_repr = str(cls)
    return_list.append(str_repr)
    if 'tensorflow' not in str_repr:
      break

    # Hack - tensorflow.test.StubOutForTesting may or may not be type <object>
    # depending on the environment. To avoid inconsistency, break after we add
    # StubOutForTesting to the return_list.
    if 'StubOutForTesting' in str_repr:
      break

  return return_list


class PythonObjectToProtoVisitor(object):
  """A visitor that summarizes given python objects as protobufs."""

  def __init__(self):
    # A dict to store all protocol buffers.
    # Keyed by "path" to the object.
    self._protos = {}

  def GetProtos(self):
    """Return the list of protos stored."""
    return self._protos

  def __call__(self, path, parent, children):
    # The path to the object.
    lib_path = 'tensorflow.%s' % path if path else 'tensorflow'

    # A small helper method to construct members(children) protos.
    def _AddMember(member_name, member_obj, proto):
      """Add the child object to the object being constructed."""
      _, member_obj = tf_decorator.unwrap(member_obj)
      if member_name == '__init__' or not member_name.startswith('_'):
        if tf_inspect.isroutine(member_obj):
          new_method = proto.member_method.add()
          new_method.name = member_name
          # If member_obj is a python builtin, there is no way to get its
          # argspec, because it is implemented on the C side. It also has no
          # func_code.
          if getattr(member_obj, 'func_code', None):
            new_method.argspec = _SanitizedArgSpec(member_obj)
        else:
          new_member = proto.member.add()
          new_member.name = member_name
          new_member.mtype = str(type(member_obj))

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
      elif tf_inspect.isclass(parent):
        # Construct a class.
        class_obj = api_objects_pb2.TFAPIClass()
        class_obj.is_instance.extend(_SanitizedMRO(parent))
        for name, child in children:
          if name in parent_corner_cases:
            # If we have an empty entry, skip this object.
            if parent_corner_cases[name]:
              module_obj.member.add(**(parent_corner_cases[name]))
          else:
            _AddMember(name, child, class_obj)

        # Store the constructed class object.
        self._protos[lib_path] = api_objects_pb2.TFAPIObject(
            path=lib_path, tf_class=class_obj)
      else:
        logging.error('Illegal call to ApiProtoDump::_py_obj_to_proto.'
                      'Object is neither a module nor a class: %s', path)
