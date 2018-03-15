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

// Functions for getting information about kernels registered in the binary.
#ifndef TENSORFLOW_PYTHON_UTIL_UTIL_H_
#define TENSORFLOW_PYTHON_UTIL_UTIL_H_

#include <Python.h>

namespace tensorflow {
namespace swig {

// Implements the same interface as tensorflow.util.nest.is_sequence
// Returns a true if its input is a collections.Sequence (except strings).
//
// Args:
//   seq: an input sequence.
//
// Returns:
//   True if the sequence is a not a string and is a collections.Sequence or a
//   dict.
bool IsSequence(PyObject* o);

// Implements the same interface as tensorflow.util.nest.flatten
//
// Returns a flat list from a given nested structure.
//
// If `nest` is not a sequence, tuple, or dict, then returns a single-element
// list: `[nest]`.
//
// In the case of dict instances, the sequence consists of the values, sorted by
// key to ensure deterministic behavior. This is true also for `OrderedDict`
// instances: their sequence order is ignored, the sorting order of keys is
// used instead. The same convention is followed in `pack_sequence_as`. This
// correctly repacks dicts and `OrderedDict`s after they have been flattened,
// and also allows flattening an `OrderedDict` and then repacking it back using
// a correponding plain dict, or vice-versa.
// Dictionaries with non-sortable keys cannot be flattened.
//
// Args:
//   nest: an arbitrarily nested structure or a scalar object. Note, numpy
//       arrays are considered scalars.
//
// Returns:
//   A Python list, the flattened version of the input.
//   On error, returns nullptr
//
// Raises:
//   TypeError: The nest is or contains a dict with non-sortable keys.
PyObject* Flatten(PyObject* nested);

// RegisterSequenceClass is used to pass PyTypeObject for collections.Sequence
// (which is defined in python) into the C++ world.
// Alternative approach could be to import the collections modules and retrieve
// the type from the module. This approach also requires some trigger from
// Python so that we know that Python interpreter had been initialzied.
void RegisterSequenceClass(PyObject* sequence_class);

}  // namespace swig
}  // namespace tensorflow

#endif  // TENSORFLOW_PYTHON_UTIL_UTIL_H_
