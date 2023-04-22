/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_PYTHON_UTIL_FUNCTION_PARAMETER_CANONICALIZER_H_
#define TENSORFLOW_PYTHON_UTIL_FUNCTION_PARAMETER_CANONICALIZER_H_

#include <Python.h>

#include <vector>

#include "absl/types/span.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/python/lib/core/safe_pyobject_ptr.h"

namespace tensorflow {

// A class that Canonicalizes Python arg & kwargs parameters.
class FunctionParameterCanonicalizer {
 public:
  // `arg_names` is a list of argument names, and `defaults` is default PyObject
  // instances for arguments. `default` is aligned to the end.
  FunctionParameterCanonicalizer(absl::Span<const char*> arg_names,
                                 absl::Span<PyObject*> defaults);

  // Returns the total number of arguments.
  ABSL_MUST_USE_RESULT
  int GetArgSize() const { return interned_arg_names_.size(); }

  // Canonicalizes `args` and `kwargs` by the spec specified at construction.
  // It's written to `result`. Returns `true` if Canonicalization was
  // successful, and `false` otherwise. When it fails, it also sets CPython
  // error status.
  // This function does not update reference counter of any Python objects.
  // `PyObject*`s in `result` are borrowed references from `args`, `kwargs`, and
  // possibly `defaults_`, and will be only valid if `args` and `kwargs` are
  // still alive.
  ABSL_MUST_USE_RESULT
  ABSL_ATTRIBUTE_HOT
  bool Canonicalize(PyObject* args, PyObject* kwargs,
                    absl::Span<PyObject*> result);

 private:
  // Simple linear search of `name` in `interned_arg_names`. If found, returns
  // the index. If not found, returns `interned_arg_names.size()`.
  ABSL_MUST_USE_RESULT
  ABSL_ATTRIBUTE_HOT
  std::size_t InternedArgNameLinearSearch(PyObject* name);

  // Check if `interned_arg_names_` is unique.
  bool AreInternedArgNamesUnique();

  // TODO(kkb): Use one `std::vector` and two `absl:Span`s instead to improve
  // cache locality.
  std::vector<Safe_PyObjectPtr> interned_arg_names_;
  std::vector<Safe_PyObjectPtr> defaults_;
  const int positional_args_size_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_PYTHON_UTIL_FUNCTION_PARAMETER_CANONICALIZER_H_
