/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_PYTHON_LIB_CORE_NUMPY_H_
#define TENSORFLOW_PYTHON_LIB_CORE_NUMPY_H_

#ifdef PyArray_Type
#error "Numpy cannot be included before numpy.h."
#endif

// Disallow Numpy 1.7 deprecated symbols.
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

// We import_array in the tensorflow init function only.
#define PY_ARRAY_UNIQUE_SYMBOL _tensorflow_numpy_api
#ifndef TF_IMPORT_NUMPY
#define NO_IMPORT_ARRAY
#endif

#include <Python.h>

#include "numpy/arrayobject.h"
#include "numpy/ufuncobject.h"

namespace tensorflow {

// Import numpy.  This wrapper function exists so that the
// PY_ARRAY_UNIQUE_SYMBOL can be safely defined in a .cc file to
// avoid weird linking issues.  Should be called only from our
// module initialization function.
void ImportNumpy();

}  // namespace tensorflow

#endif  // TENSORFLOW_PYTHON_LIB_CORE_NUMPY_H_
