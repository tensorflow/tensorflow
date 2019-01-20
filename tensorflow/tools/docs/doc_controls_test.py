# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for documentation control decorators."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.platform import googletest
from tensorflow.tools.docs import doc_controls


class DocControlsTest(googletest.TestCase):

  def test_do_not_generate_docs(self):

    @doc_controls.do_not_generate_docs
    def dummy_function():
      pass

    self.assertTrue(doc_controls.should_skip(dummy_function))

  def test_do_not_doc_on_method(self):
    """The simple decorator is not aware of inheritance."""

    class Parent(object):

      @doc_controls.do_not_generate_docs
      def my_method(self):
        pass

    class Child(Parent):

      def my_method(self):
        pass

    class GrandChild(Child):
      pass

    self.assertTrue(doc_controls.should_skip(Parent.my_method))
    self.assertFalse(doc_controls.should_skip(Child.my_method))
    self.assertFalse(doc_controls.should_skip(GrandChild.my_method))

    self.assertTrue(doc_controls.should_skip_class_attr(Parent, 'my_method'))
    self.assertFalse(doc_controls.should_skip_class_attr(Child, 'my_method'))
    self.assertFalse(
        doc_controls.should_skip_class_attr(GrandChild, 'my_method'))

  def test_do_not_doc_inheritable(self):

    class Parent(object):

      @doc_controls.do_not_doc_inheritable
      def my_method(self):
        pass

    class Child(Parent):

      def my_method(self):
        pass

    class GrandChild(Child):
      pass

    self.assertTrue(doc_controls.should_skip(Parent.my_method))
    self.assertFalse(doc_controls.should_skip(Child.my_method))
    self.assertFalse(doc_controls.should_skip(GrandChild.my_method))

    self.assertTrue(doc_controls.should_skip_class_attr(Parent, 'my_method'))
    self.assertTrue(doc_controls.should_skip_class_attr(Child, 'my_method'))
    self.assertTrue(
        doc_controls.should_skip_class_attr(GrandChild, 'my_method'))

  def test_do_not_doc_inheritable_property(self):

    class Parent(object):

      @property
      @doc_controls.do_not_doc_inheritable
      def my_method(self):
        pass

    class Child(Parent):

      @property
      def my_method(self):
        pass

    class GrandChild(Child):
      pass

    self.assertTrue(doc_controls.should_skip(Parent.my_method))
    self.assertFalse(doc_controls.should_skip(Child.my_method))
    self.assertFalse(doc_controls.should_skip(GrandChild.my_method))

    self.assertTrue(doc_controls.should_skip_class_attr(Parent, 'my_method'))
    self.assertTrue(doc_controls.should_skip_class_attr(Child, 'my_method'))
    self.assertTrue(
        doc_controls.should_skip_class_attr(GrandChild, 'my_method'))

  def test_do_not_doc_inheritable_staticmethod(self):

    class GrandParent(object):

      def my_method(self):
        pass

    class Parent(GrandParent):

      @staticmethod
      @doc_controls.do_not_doc_inheritable
      def my_method():
        pass

    class Child(Parent):

      @staticmethod
      def my_method():
        pass

    class GrandChild(Child):
      pass

    self.assertFalse(doc_controls.should_skip(GrandParent.my_method))
    self.assertTrue(doc_controls.should_skip(Parent.my_method))
    self.assertFalse(doc_controls.should_skip(Child.my_method))
    self.assertFalse(doc_controls.should_skip(GrandChild.my_method))

    self.assertFalse(
        doc_controls.should_skip_class_attr(GrandParent, 'my_method'))
    self.assertTrue(doc_controls.should_skip_class_attr(Parent, 'my_method'))
    self.assertTrue(doc_controls.should_skip_class_attr(Child, 'my_method'))
    self.assertTrue(
        doc_controls.should_skip_class_attr(GrandChild, 'my_method'))

  def test_for_subclass_implementers(self):

    class GrandParent(object):

      def my_method(self):
        pass

    class Parent(GrandParent):

      @doc_controls.for_subclass_implementers
      def my_method(self):
        pass

    class Child(Parent):
      pass

    class GrandChild(Child):

      def my_method(self):
        pass

    class Grand2Child(Child):
      pass

    self.assertFalse(
        doc_controls.should_skip_class_attr(GrandParent, 'my_method'))
    self.assertFalse(doc_controls.should_skip_class_attr(Parent, 'my_method'))
    self.assertTrue(doc_controls.should_skip_class_attr(Child, 'my_method'))
    self.assertTrue(
        doc_controls.should_skip_class_attr(GrandChild, 'my_method'))
    self.assertTrue(
        doc_controls.should_skip_class_attr(Grand2Child, 'my_method'))

  def test_for_subclass_implementers_short_circuit(self):

    class GrandParent(object):

      @doc_controls.for_subclass_implementers
      def my_method(self):
        pass

    class Parent(GrandParent):

      def my_method(self):
        pass

    class Child(Parent):

      @doc_controls.do_not_doc_inheritable
      def my_method(self):
        pass

    class GrandChild(Child):

      @doc_controls.for_subclass_implementers
      def my_method(self):
        pass

    class Grand2Child(Child):
      pass

    self.assertFalse(
        doc_controls.should_skip_class_attr(GrandParent, 'my_method'))
    self.assertTrue(doc_controls.should_skip_class_attr(Parent, 'my_method'))
    self.assertTrue(doc_controls.should_skip_class_attr(Child, 'my_method'))
    self.assertFalse(
        doc_controls.should_skip_class_attr(GrandChild, 'my_method'))
    self.assertTrue(
        doc_controls.should_skip_class_attr(Grand2Child, 'my_method'))


if __name__ == '__main__':
  googletest.main()
