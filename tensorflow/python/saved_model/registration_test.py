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
"""Trackable class registration tests.

For integrated tests, see registration_saving_test.py.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

from tensorflow.python.eager import test
from tensorflow.python.saved_model import registration
from tensorflow.python.training.tracking import base


@registration.register_serializable()
class RegisteredClass(base.Trackable):
  pass


@registration.register_serializable(name="Subclass")
class RegisteredSubclass(RegisteredClass):
  pass


@registration.register_serializable(package="testing")
class CustomPackage(base.Trackable):
  pass


@registration.register_serializable(package="testing", name="name")
class CustomPackageAndName(base.Trackable):
  pass


class SerializableRegistrationTest(test.TestCase, parameterized.TestCase):

  @parameterized.parameters([
      (RegisteredClass, "Custom.RegisteredClass"),
      (RegisteredSubclass, "Custom.Subclass"),
      (CustomPackage, "testing.CustomPackage"),
      (CustomPackageAndName, "testing.name"),
  ])
  def test_registration(self, expected_cls, expected_name):
    obj = expected_cls()
    self.assertEqual(registration.get_registered_name(obj), expected_name)
    self.assertIs(
        registration.get_registered_class(expected_name), expected_cls)

  def test_get_invalid_name(self):
    self.assertIsNone(registration.get_registered_class("invalid name"))

  def test_get_unregistered_class(self):

    class NotRegistered(base.Trackable):
      pass

    no_register = NotRegistered
    self.assertIsNone(registration.get_registered_name(no_register))

  def test_duplicate_registration(self):

    @registration.register_serializable()
    class Duplicate(base.Trackable):
      pass

    dup = Duplicate()
    self.assertEqual(registration.get_registered_name(dup), "Custom.Duplicate")
    # Registrations with different names are ok.
    registration.register_serializable(package="duplicate")(Duplicate)
    # Registrations are checked in reverse order.
    self.assertEqual(
        registration.get_registered_name(dup), "duplicate.Duplicate")
    # Both names should resolve to the same class.
    self.assertIs(
        registration.get_registered_class("Custom.Duplicate"), Duplicate)
    self.assertIs(
        registration.get_registered_class("duplicate.Duplicate"), Duplicate)

    # Registrations of the same name fails
    with self.assertRaisesRegex(ValueError, "already been registered"):
      registration.register_serializable(
          package="testing", name="CustomPackage")(
              Duplicate)

  def test_register_non_class_fails(self):
    obj = RegisteredClass()
    with self.assertRaisesRegex(ValueError, "must be a class"):
      registration.register_serializable()(obj)

  def test_register_bad_predicate_fails(self):
    with self.assertRaisesRegex(ValueError, "must be callable"):
      registration.register_serializable(predicate=0)

  def test_predicate(self):

    class Predicate(base.Trackable):

      def __init__(self, register_this):
        self.register_this = register_this

    registration.register_serializable(
        name="RegisterThisOnlyTrue",
        predicate=lambda x: isinstance(x, Predicate) and x.register_this)(
            Predicate)

    a = Predicate(True)
    b = Predicate(False)
    self.assertEqual(
        registration.get_registered_name(a), "Custom.RegisterThisOnlyTrue")
    self.assertIsNone(registration.get_registered_name(b))

    registration.register_serializable(
        name="RegisterAllPredicate",
        predicate=lambda x: isinstance(x, Predicate))(
            Predicate)

    self.assertEqual(
        registration.get_registered_name(a), "Custom.RegisterAllPredicate")
    self.assertEqual(
        registration.get_registered_name(b), "Custom.RegisterAllPredicate")


if __name__ == "__main__":
  test.main()
