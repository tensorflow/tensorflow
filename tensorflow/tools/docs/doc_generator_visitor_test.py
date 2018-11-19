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
# ==============================================================================
"""Tests for tools.docs.doc_generator_visitor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import types

from tensorflow.python.platform import googletest
from tensorflow.tools.docs import doc_generator_visitor
from tensorflow.tools.docs import generate_lib


class NoDunderVisitor(doc_generator_visitor.DocGeneratorVisitor):

  def __call__(self, parent_name, parent, children):
    """Drop all the dunder methods to make testing easier."""
    children = [
        (name, obj) for (name, obj) in children if not name.startswith('_')
    ]
    super(NoDunderVisitor, self).__call__(parent_name, parent, children)


class DocGeneratorVisitorTest(googletest.TestCase):

  def test_call_module(self):
    visitor = doc_generator_visitor.DocGeneratorVisitor()
    visitor(
        'doc_generator_visitor', doc_generator_visitor,
        [('DocGeneratorVisitor', doc_generator_visitor.DocGeneratorVisitor)])

    self.assertEqual({'doc_generator_visitor': ['DocGeneratorVisitor']},
                     visitor.tree)
    self.assertEqual({
        'doc_generator_visitor': doc_generator_visitor,
        'doc_generator_visitor.DocGeneratorVisitor':
        doc_generator_visitor.DocGeneratorVisitor,
    }, visitor.index)

  def test_call_class(self):
    visitor = doc_generator_visitor.DocGeneratorVisitor()
    visitor(
        'DocGeneratorVisitor', doc_generator_visitor.DocGeneratorVisitor,
        [('index', doc_generator_visitor.DocGeneratorVisitor.index)])

    self.assertEqual({'DocGeneratorVisitor': ['index']},
                     visitor.tree)
    self.assertEqual({
        'DocGeneratorVisitor': doc_generator_visitor.DocGeneratorVisitor,
        'DocGeneratorVisitor.index':
        doc_generator_visitor.DocGeneratorVisitor.index
    }, visitor.index)

  def test_call_raises(self):
    visitor = doc_generator_visitor.DocGeneratorVisitor()
    with self.assertRaises(RuntimeError):
      visitor('non_class_or_module', 'non_class_or_module_object', [])

  def test_duplicates_module_class_depth(self):

    class Parent(object):

      class Nested(object):
        pass

    tf = types.ModuleType('tf')
    tf.Parent = Parent
    tf.submodule = types.ModuleType('submodule')
    tf.submodule.Parent = Parent

    visitor = generate_lib.extract(
        [('tf', tf)],
        private_map={},
        do_not_descend_map={},
        visitor_cls=NoDunderVisitor)

    self.assertEqual({
        'tf.submodule.Parent':
            sorted([
                'tf.Parent',
                'tf.submodule.Parent',
            ]),
        'tf.submodule.Parent.Nested':
            sorted([
                'tf.Parent.Nested',
                'tf.submodule.Parent.Nested',
            ]),
    }, visitor.duplicates)

    self.assertEqual({
        'tf.Parent.Nested': 'tf.submodule.Parent.Nested',
        'tf.Parent': 'tf.submodule.Parent',
    }, visitor.duplicate_of)

    self.assertEqual({
        id(Parent): 'tf.submodule.Parent',
        id(Parent.Nested): 'tf.submodule.Parent.Nested',
        id(tf): 'tf',
        id(tf.submodule): 'tf.submodule',
    }, visitor.reverse_index)

  def test_duplicates_contrib(self):

    class Parent(object):
      pass

    tf = types.ModuleType('tf')
    tf.contrib = types.ModuleType('contrib')
    tf.submodule = types.ModuleType('submodule')
    tf.contrib.Parent = Parent
    tf.submodule.Parent = Parent

    visitor = generate_lib.extract(
        [('tf', tf)],
        private_map={},
        do_not_descend_map={},
        visitor_cls=NoDunderVisitor)

    self.assertEqual({
        'tf.submodule.Parent':
            sorted(['tf.contrib.Parent', 'tf.submodule.Parent']),
    }, visitor.duplicates)

    self.assertEqual({
        'tf.contrib.Parent': 'tf.submodule.Parent',
    }, visitor.duplicate_of)

    self.assertEqual({
        id(tf): 'tf',
        id(tf.submodule): 'tf.submodule',
        id(Parent): 'tf.submodule.Parent',
        id(tf.contrib): 'tf.contrib',
    }, visitor.reverse_index)

  def test_duplicates_defining_class(self):

    class Parent(object):
      obj1 = object()

    class Child(Parent):
      pass

    tf = types.ModuleType('tf')
    tf.Parent = Parent
    tf.Child = Child

    visitor = generate_lib.extract(
        [('tf', tf)],
        private_map={},
        do_not_descend_map={},
        visitor_cls=NoDunderVisitor)

    self.assertEqual({
        'tf.Parent.obj1': sorted([
            'tf.Parent.obj1',
            'tf.Child.obj1',
        ]),
    }, visitor.duplicates)

    self.assertEqual({
        'tf.Child.obj1': 'tf.Parent.obj1',
    }, visitor.duplicate_of)

    self.assertEqual({
        id(tf): 'tf',
        id(Parent): 'tf.Parent',
        id(Child): 'tf.Child',
        id(Parent.obj1): 'tf.Parent.obj1',
    }, visitor.reverse_index)

  def test_duplicates_module_depth(self):

    class Parent(object):
      pass

    tf = types.ModuleType('tf')
    tf.submodule = types.ModuleType('submodule')
    tf.submodule.submodule2 = types.ModuleType('submodule2')
    tf.Parent = Parent
    tf.submodule.submodule2.Parent = Parent

    visitor = generate_lib.extract(
        [('tf', tf)],
        private_map={},
        do_not_descend_map={},
        visitor_cls=NoDunderVisitor)

    self.assertEqual({
        'tf.Parent': sorted(['tf.Parent', 'tf.submodule.submodule2.Parent']),
    }, visitor.duplicates)

    self.assertEqual({
        'tf.submodule.submodule2.Parent': 'tf.Parent'
    }, visitor.duplicate_of)

    self.assertEqual({
        id(tf): 'tf',
        id(tf.submodule): 'tf.submodule',
        id(tf.submodule.submodule2): 'tf.submodule.submodule2',
        id(Parent): 'tf.Parent',
    }, visitor.reverse_index)

  def test_duplicates_name(self):

    class Parent(object):
      obj1 = object()

    Parent.obj2 = Parent.obj1

    tf = types.ModuleType('tf')
    tf.submodule = types.ModuleType('submodule')
    tf.submodule.Parent = Parent

    visitor = generate_lib.extract(
        [('tf', tf)],
        private_map={},
        do_not_descend_map={},
        visitor_cls=NoDunderVisitor)

    self.assertEqual({
        'tf.submodule.Parent.obj1':
            sorted([
                'tf.submodule.Parent.obj1',
                'tf.submodule.Parent.obj2',
            ]),
    }, visitor.duplicates)

    self.assertEqual({
        'tf.submodule.Parent.obj2': 'tf.submodule.Parent.obj1',
    }, visitor.duplicate_of)

    self.assertEqual({
        id(tf): 'tf',
        id(tf.submodule): 'tf.submodule',
        id(Parent): 'tf.submodule.Parent',
        id(Parent.obj1): 'tf.submodule.Parent.obj1',
    }, visitor.reverse_index)

if __name__ == '__main__':
  googletest.main()
