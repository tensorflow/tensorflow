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

from tensorflow.python.platform import googletest
from tensorflow.tools.docs import doc_generator_visitor


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

  def test_duplicates(self):
    visitor = doc_generator_visitor.DocGeneratorVisitor()
    visitor(
        'submodule.DocGeneratorVisitor',
        doc_generator_visitor.DocGeneratorVisitor,
        [('index', doc_generator_visitor.DocGeneratorVisitor.index),
         ('index2', doc_generator_visitor.DocGeneratorVisitor.index)])
    visitor(
        'submodule2.DocGeneratorVisitor',
        doc_generator_visitor.DocGeneratorVisitor,
        [('index', doc_generator_visitor.DocGeneratorVisitor.index),
         ('index2', doc_generator_visitor.DocGeneratorVisitor.index)])
    visitor(
        'DocGeneratorVisitor2',
        doc_generator_visitor.DocGeneratorVisitor,
        [('index', doc_generator_visitor.DocGeneratorVisitor.index),
         ('index2', doc_generator_visitor.DocGeneratorVisitor.index)])

    duplicate_of, duplicates = visitor.find_duplicates()

    # The shorter path should be master, or if equal, the lexicographically
    # first will be.
    self.assertEqual(
        {'DocGeneratorVisitor2': sorted(['submodule.DocGeneratorVisitor',
                                         'submodule2.DocGeneratorVisitor',
                                         'DocGeneratorVisitor2']),
         'DocGeneratorVisitor2.index': sorted([
             'submodule.DocGeneratorVisitor.index',
             'submodule.DocGeneratorVisitor.index2',
             'submodule2.DocGeneratorVisitor.index',
             'submodule2.DocGeneratorVisitor.index2',
             'DocGeneratorVisitor2.index',
             'DocGeneratorVisitor2.index2'
         ]),
        }, duplicates)
    self.assertEqual({
        'submodule.DocGeneratorVisitor': 'DocGeneratorVisitor2',
        'submodule.DocGeneratorVisitor.index': 'DocGeneratorVisitor2.index',
        'submodule.DocGeneratorVisitor.index2': 'DocGeneratorVisitor2.index',
        'submodule2.DocGeneratorVisitor': 'DocGeneratorVisitor2',
        'submodule2.DocGeneratorVisitor.index': 'DocGeneratorVisitor2.index',
        'submodule2.DocGeneratorVisitor.index2': 'DocGeneratorVisitor2.index',
        'DocGeneratorVisitor2.index2': 'DocGeneratorVisitor2.index'
    }, duplicate_of)


if __name__ == '__main__':
  googletest.main()
