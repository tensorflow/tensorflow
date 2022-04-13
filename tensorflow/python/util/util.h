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

#include <string>

namespace tensorflow {
namespace swig {

// Implements `tensorflow.util.nest.is_nested`.
bool IsNested(PyObject* o);

// Implements `tensorflow.util.nest.is_nested_or_composite`.
bool IsNestedOrComposite(PyObject* o);

// Returns a true if its input is a CompositeTensor or a TypeSpec.
//
// Args:
//   o: the object to check.
//
// Returns:
//   True if the object is a CompositeTensor.
bool IsCompositeTensor(PyObject* o);

// Returns a true if its input is a TypeSpec, but is not a TensorSpec.
//
// Args:
//   o: the object to check.
//
// Returns:
//   True if the object is a TypeSpec, but is not a TensorSpec.
bool IsTypeSpec(PyObject* o);

// Implements the same interface as tensorflow.util.nest.is_namedtuple
// Returns Py_True iff `instance` should be considered a `namedtuple`.
//
// Args:
//   instance: An instance of a Python object.
//   strict: If True, `instance` is considered to be a `namedtuple` only if
//       it is a "plain" namedtuple. For instance, a class inheriting
//       from a `namedtuple` will be considered to be a `namedtuple`
//       iff `strict=False`.
//
// Returns:
//   True if `instance` is a `namedtuple`.
PyObject* IsNamedtuple(PyObject* o, bool strict);

// Returns a true if its input is a collections.Mapping.
//
// Args:
//   o: the object to be checked.
//
// Returns:
//   True if the object subclasses mapping.
bool IsMapping(PyObject* o);

// Returns a true if its input is a collections.MutableMapping.
//
// Args:
//   o: the object to be checked.
//
// Returns:
//   True if the object subclasses mapping.
bool IsMutableMapping(PyObject* o);

// Returns a true if its input is a (possibly wrapped) tuple.
//
// Args:
//   o: the object to be checked.
//
// Returns:
//   True if the object is a tuple.
bool IsTuple(PyObject* o);

// Returns a true if its input is a collections.MappingView.
//
// Args:
//   o: the object to be checked.
//
// Returns:
//   True if the object subclasses mapping.
bool IsMappingView(PyObject* o);

// Returns a true if its input has a `__tf_dispatch__` attribute.
//
// Args:
//   o: the input to be checked.
//
// Returns:
//   True if `o` has a `__tf_dispatch__` attribute.
bool IsDispatchable(PyObject* o);

// A version of PyMapping_Keys that works in C++11
//
// Args:
//   o: The input to extract keys from
//
// Returns:
//   A new reference to a list of keys in the mapping.
PyObject* MappingKeys(PyObject* o);

// Returns a true if its input is an instance of an attr.s decorated class.
//
// Args:
//   o: the input to be checked.
//
// Returns:
//   True if the object is an instance of an attr.s decorated class.
bool IsAttrs(PyObject* o);

// Returns a true if its input is an ops.Tensor.
//
// Args:
//   o: the input to be checked.
//
// Returns:
//   True if the object is a tensor.
bool IsTensor(PyObject* o);

// Returns true if its input is a tf.TensorSpec.
//
// Args:
//   o: the input to be checked.
//
// Returns:
//   True if the object is a TensorSpec.
bool IsTensorSpec(PyObject* o);

// Returns a true if its input is an eager.EagerTensor.
//
// Args:
//   o: the input to be checked.
//
// Returns:
//   True if the object is an eager tensor (or mimicking as one).
bool IsEagerTensorSlow(PyObject* o);

// Returns a true if its input is a ResourceVariable.
//
// Args:
//   o: the input to be checked.
//
// Returns:
//   True if the object is a ResourceVariable.
bool IsResourceVariable(PyObject* o);

// Returns a true if its input is an OwnedIterator.
//
// Args:
//   o: the input to be checked.
//
// Returns:
//   True if the object is an OwnedIterator.
bool IsOwnedIterator(PyObject* o);

// Returns a true if its input is a Variable.
//
// Args:
//   o: the input to be checked.
//
// Returns:
//   True if the object is a Variable.
bool IsVariable(PyObject* o);

// Returns a true if its input is an ops.IndexesSlices.
//
// Args:
//   o: the input to be checked.
//
// Returns:
//   True if the object is an ops.IndexedSlices.
bool IsIndexedSlices(PyObject* o);

// Implements the same interface as tensorflow.util.nest.same_namedtuples
// Returns Py_True iff the two namedtuples have the same name and fields.
// Raises RuntimeError if `o1` or `o2` don't look like namedtuples (don't have
// '_fields' attribute).
PyObject* SameNamedtuples(PyObject* o1, PyObject* o2);

// Implements `tensorflow.util.nest.assert_same_structrure`.
PyObject* AssertSameStructure(PyObject* o1, PyObject* o2, bool check_types,
                              bool expand_composites);

// Implements `tensorflow.util.nest.flatten`.
PyObject* Flatten(PyObject* nested, bool expand_composites = false);

// The tensorflow.python.data package has its own nest utility that follows very
// slightly different semantics for its functions than the tensorflow.python
// nest utility. Returns True if its input is a nested structure for tf.data.
//
// Main differences are (this is copied from nest.py in the
// tensorflow.data.util):
//
// 1. It removes support for lists as a level of nesting in nested structures.
// 2. It adds support for `SparseTensorValue` as an atomic element.
bool IsNestedForData(PyObject* o);

// Flatten specialized for `tf.data`. Additional comments about
// difference in functionality can be found in nest.py in
// `tensorflow.python.data.util` and in the comments for Flatten above.
PyObject* FlattenForData(PyObject* nested);

// AssertSameStructure specialized for `tf.data`.
PyObject* AssertSameStructureForData(PyObject* o1, PyObject* o2,
                                     bool check_types);

// Registers a Python object so it can be looked up from c++.  The set of
// valid names, and the expected values for those names, are listed in
// the documentation for `RegisteredPyObjects`.  Returns PyNone.
PyObject* RegisterPyObject(PyObject* name, PyObject* value);

// Variant of RegisterPyObject that requires the object's value to be a type.
PyObject* RegisterType(PyObject* type_name, PyObject* type);

// Returns a borrowed reference to an object that was registered with
// RegisterPyObject.  (Do not call Py_DECREF on the result).
PyObject* GetRegisteredPyObject(const std::string& name);

}  // namespace swig
}  // namespace tensorflow

#endif  // TENSORFLOW_PYTHON_UTIL_UTIL_H_
