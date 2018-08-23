/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
%include "tensorflow/python/platform/base.i"

%{
#include "tensorflow/python/util/util.h"
%}

%ignoreall

%unignore tensorflow;
%unignore tensorflow::swig;
// The %exception block defined in tf_session.i releases the Python GIL for
// the length of each wrapped method. This file is included in tensorflow.i
// after tf_session.i and inherits this definition. We disable this behavior
// for functions in this module because they use python methods that need GIL.
// TODO(iga): Find a way not to leak such definitions across files.

%unignore tensorflow::swig::RegisterSequenceClass;
%noexception tensorflow::swig::RegisterSequenceClass;

%unignore tensorflow::swig::RegisterMappingClass;
%noexception tensorflow::swig::RegisterMappingClass;

%unignore tensorflow::swig::RegisterSparseTensorValueClass;
%noexception tensorflow::swig::RegisterSparseTensorValueClass;

%feature("docstring") tensorflow::swig::IsSequence
"""Returns a true if its input is a collections.Sequence (except strings).

Args:
  seq: an input sequence.

Returns:
  True if the sequence is a not a string and is a collections.Sequence or a
  dict.
"""
%unignore tensorflow::swig::IsSequence;
%noexception tensorflow::swig::IsSequence;

%unignore tensorflow::swig::IsNamedtuple;
%noexception tensorflow::swig::IsNamedtuple;

%feature("docstring") tensorflow::swig::IsMapping
"""Returns True iff `instance` is a `collections.Mapping`.

Args:
  instance: An instance of a Python object.

Returns:
  True if `instance` is a `collections.Mapping`.
"""
%unignore tensorflow::swig::IsMapping;
%noexception tensorflow::swig::IsMapping;

%feature("docstring") tensorflow::swig::SameNamedtuples
"Returns True if the two namedtuples have the same name and fields."
%unignore tensorflow::swig::SameNamedtuples;
%noexception tensorflow::swig::SameNamedtuples;

%unignore tensorflow::swig::AssertSameStructure;
%noexception tensorflow::swig::AssertSameStructure;

%feature("docstring") tensorflow::swig::Flatten
"""Returns a flat list from a given nested structure.

If `nest` is not a sequence, tuple, or dict, then returns a single-element
list: `[nest]`.

In the case of dict instances, the sequence consists of the values, sorted by
key to ensure deterministic behavior. This is true also for `OrderedDict`
instances: their sequence order is ignored, the sorting order of keys is
used instead. The same convention is followed in `pack_sequence_as`. This
correctly repacks dicts and `OrderedDict`s after they have been flattened,
and also allows flattening an `OrderedDict` and then repacking it back using
a corresponding plain dict, or vice-versa.
Dictionaries with non-sortable keys cannot be flattened.

Users must not modify any collections used in `nest` while this function is
running.

Args:
  nest: an arbitrarily nested structure or a scalar object. Note, numpy
      arrays are considered scalars.

Returns:
  A Python list, the flattened version of the input.

Raises:
  TypeError: The nest is or contains a dict with non-sortable keys.
"""
%unignore tensorflow::swig::Flatten;
%noexception tensorflow::swig::Flatten;

%unignore tensorflow::swig::IsSequenceForData;
%noexception tensorflow::swig::IsSequenceForData;

%unignore tensorflow::swig::FlattenForData;
%noexception tensorflow::swig::FlattenForData;

%unignore tensorflow::swig::AssertSameStructureForData;
%noexception tensorflow::swig::AssertSameStructureForData;

%include "tensorflow/python/util/util.h"

%unignoreall
