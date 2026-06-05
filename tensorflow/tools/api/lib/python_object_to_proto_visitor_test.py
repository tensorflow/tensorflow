# Copyright 2026 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for Python object API proto visitor."""

import enum
import functools
import inspect
import types

from tensorflow.python.platform import googletest
from tensorflow.tools.api.lib import python_object_to_proto_visitor as visitor_lib


def _TensorFlowOwnedBase(module):
  return type('TensorFlowOwnedBase', (BaseException,), {
      '__module__': module,
      'tf_owned_member': property(lambda self: None),
  })


class PythonObjectToProtoVisitorTest(googletest.TestCase):

  def test_partial_member_is_recorded_as_member(self):

    def target(value):
      return value

    module = types.ModuleType('fake_module')
    visitor = visitor_lib.PythonObjectToProtoVisitor()

    visitor('', module, [('generate', functools.partial(target, 1))])

    proto = visitor.GetProtos()['tensorflow']
    self.assertEqual(['generate'],
                     [member.name for member in proto.tf_module.member])
    self.assertEqual("<class 'functools.partial'>",
                     proto.tf_module.member[0].mtype)
    self.assertEqual([],
                     [method.name for method in proto.tf_module.member_method])

  def test_tensorflow_owned_class_matches_mid_string_module(self):
    cls = _TensorFlowOwnedBase('third_party.py.tensorflow.python.framework')

    self.assertTrue(visitor_lib._IsTensorFlowOwnedClass(cls))

  def test_tensorflow_owned_class_matches_tensorflow_family_segments(self):
    modules = (
        'third_party.py.keras.src.layers',
        'third_party.py.tensorflow_probability.python',
        'third_party.py.tf_keras.src.engine',
    )

    for module in modules:
      cls = _TensorFlowOwnedBase(module)

      self.assertTrue(visitor_lib._IsTensorFlowOwnedClass(cls), module)

  def test_tensorflow_owned_class_rejects_embedded_token(self):
    cls = _TensorFlowOwnedBase('third_party.py.not_tensorflow.python')

    self.assertFalse(visitor_lib._IsTensorFlowOwnedClass(cls))

  def test_unstable_external_runtime_member_is_recognized(self):

    class Exported(TypeError):
      pass

    self.assertTrue(
        visitor_lib._IsUnstableExternalInheritedMember(Exported, 'args'))

  def test_direct_runtime_member_override_is_recognized(self):

    class Exported(TypeError):

      def add_note(self, note):
        del note

    self.assertFalse(
        visitor_lib._IsUnstableExternalInheritedMember(Exported, 'add_note'))

  def test_tensorflow_family_inherited_member_is_recognized(self):
    base = _TensorFlowOwnedBase('third_party.py.tensorflow.python.framework')
    exported = type('Exported', (base,), {'__module__': 'public_api'})

    self.assertFalse(
        visitor_lib._IsUnstableExternalInheritedMember(
            exported, 'tf_owned_member'))

  def test_enum_runtime_member_is_recognized(self):

    class Exported(enum.Enum):
      VALUE = 1

    self.assertTrue(
        visitor_lib._IsUnstableExternalInheritedMember(Exported, 'name'))

  def test_inspect_signature_runtime_member_is_recognized(self):

    class Exported(inspect.Signature):
      pass

    self.assertTrue(
        visitor_lib._IsUnstableExternalInheritedMember(Exported, 'bind'))

  def test_visitor_filters_only_unstable_external_inherited_members(self):
    base = _TensorFlowOwnedBase('third_party.py.tensorflow.python.framework')
    exported = type('Exported', (base,), {'__module__': 'public_api'})
    children = [
        ('args', BaseException.args),
        ('tf_owned_member', base.tf_owned_member),
    ]

    visitor = visitor_lib.PythonObjectToProtoVisitor()
    visitor('errors.Exported', exported, children)

    self.assertEqual(['tf_owned_member'], [name for name, _ in children])


if __name__ == '__main__':
  googletest.main()
