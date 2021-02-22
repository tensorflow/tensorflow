/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_PYTHON_COMPAT_NEST_H_
#define TENSORFLOW_PYTHON_COMPAT_NEST_H_

#include <Python.h>

namespace tensorflow {
// Returns a dictionary with flattened keys and values.
//
// Args:
//   dict: the dictionary to zip
//
// Returns:
//   An new reference to the zipped dictionary.
//
// Raises:
//   TypeError: If the input is not a dictionary.
//   ValueError: If any key and value do not have the same structure layout, or
//       if keys are not unique.
PyObject* FlattenDictItems(PyObject* dict);

}  // namespace tensorflow

#endif  // TENSORFLOW_PYTHON_COMPAT_NEST_H_
