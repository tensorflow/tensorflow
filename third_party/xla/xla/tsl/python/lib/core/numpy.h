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

#ifndef XLA_TSL_PYTHON_LIB_CORE_NUMPY_H_
#define XLA_TSL_PYTHON_LIB_CORE_NUMPY_H_

#ifdef PyArray_Type
#error "Numpy cannot be included before numpy.h."
#endif

// Disallow Numpy 1.7 deprecated symbols.
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

// We import_array in the XLA init function only.
#define PY_ARRAY_UNIQUE_SYMBOL _xla_numpy_api
#ifndef XLA_IMPORT_NUMPY
#define NO_IMPORT_ARRAY
#endif

// Prevent linking error with numpy>=2.1.0
// error: undefined hidden symbol: _xla_numpy_apiPyArray_RUNTIME_VERSION
// Without this define, Numpy's API symbols will have hidden symbol visibility,
// which may break things if Bazel chooses to build a cc_library target into
// its own .so file. Bazel typically does this for debug builds.
#define NPY_API_SYMBOL_ATTRIBUTE

// clang-format off
// Place `<locale>` before <Python.h> to avoid build failure in macOS.
#include <locale>

#include <Python.h>
// clang-format on

#include "numpy/arrayobject.h"   // IWYU pragma: export
#include "numpy/ndarraytypes.h"  // IWYU pragma: export
#include "numpy/npy_common.h"    // IWYU pragma: export
#include "numpy/numpyconfig.h"   // IWYU pragma: export
#include "numpy/ufuncobject.h"   // IWYU pragma: export

namespace tsl {

// Import numpy.  This wrapper function exists so that the
// PY_ARRAY_UNIQUE_SYMBOL can be safely defined in a .cc file to
// avoid weird linking issues.  Should be called only from our
// module initialization function.
void ImportNumpy();

}  // namespace tsl

#endif  // XLA_TSL_PYTHON_LIB_CORE_NUMPY_H_
