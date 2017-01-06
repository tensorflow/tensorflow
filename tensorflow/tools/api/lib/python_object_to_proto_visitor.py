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

"""A visitor class that generates protobufs for each pyton object."""

import inspect

from tensorflow.tools.api.lib import api_objects_pb2
from tensorflow.python.platform import tf_logging as logging


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
    def _add_member(member_name, member_obj, proto):
      """Add the child object to the object being constructed."""
      if not member_name.startswith('_'):
        if inspect.isroutine(member_obj):
          new_method = proto.member_methods.add()
          new_method.name = member_name

          # We have some API object that are direct subclasses of python
          # builtins. These classes/methods might not have func_code, as they
          # simply provide method descriptors on python end. Seems like no way
          # to get args for such methods.
          if getattr(member_obj, 'func_code', None):
            for i in range(member_obj.func_code.co_argcount):
              new_method.args.add().arg_name = (
                  member_obj.func_code.co_varnames[i])
        else:
          new_member = proto.members.add()
          new_member.name = member_name
          new_member.mtype = str(type(member_obj))

    # Decide if we have a module or a class.
    if inspect.ismodule(parent):
      # Create a module object.
      module_obj = api_objects_pb2.TFAPIModule(name=parent.__name__)
      for name, child in children:
        _add_member(name, child, module_obj)

      # Store the constructed module object.
      self._protos[lib_path] = api_objects_pb2.TFAPIObject(
          path=lib_path,
          tf_module=module_obj)
    elif inspect.isclass(parent):
      # Construct a class.
      class_obj = api_objects_pb2.TFAPIClass(name=parent.__name__)
      class_obj.is_instance.extend(str(x) for x in inspect.getmro(parent))
      for name, child in children:
        _add_member(name, child, class_obj)

      # Store the constructed class object.
      self._protos[lib_path] = api_objects_pb2.TFAPIObject(
          path=lib_path,
          tf_class=class_obj)
    else:
      logging.error('Illegal call to ApiProtoDump::_py_obj_to_proto.'
                    'Object is neither a module nor a class: %s',
                    path)
