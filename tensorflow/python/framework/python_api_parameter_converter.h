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
#ifndef TENSORFLOW_PYTHON_UTIL_PYTHON_API_PARAMETER_CONVERTER_H_
#define TENSORFLOW_PYTHON_UTIL_PYTHON_API_PARAMETER_CONVERTER_H_

#include <Python.h>

#include <map>
#include <string>
#include <vector>

#include "absl/types/span.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/python/framework/op_def_util.h"
#include "tensorflow/python/framework/python_api_info.h"
#include "tensorflow/python/framework/python_tensor_converter.h"
#include "tensorflow/python/lib/core/safe_pyobject_ptr.h"

namespace tensorflow {

// Converts the canoncialized parameters to the expected types (in place).
//
//   * Input parameters (i.e., parameters that expect tensor values) are
//     converted to tensors (or lists of tensors) using
//     `tensor_converter.Convert`.
//   * Attribute parameters are converted to the expected type.
//   * Inferred attributes are written to `inferred_attrs`.  (Can be
//     nullptr if inferred attributes are not needed.)
//   * If there's a "name" parameter, then its value is not modified.
//
// Note: for list-of-tensor parameters, the elements of the list will be
// converted in-place.  Therefore, any list-of-tensor parameters should have
// their values copied to new lists before calling this method.  (See
// `CopyPythonAPITensorLists`.)
//
// Any values that are removed from `params` have their reference count
// decremented, and any objects added to `params` are new references.
//
// Returns true on success, or sets an exception and returns false on error.
ABSL_MUST_USE_RESULT
bool ConvertPythonAPIParameters(
    const PythonAPIInfo& api_info,
    const PythonTensorConverter& tensor_converter, absl::Span<PyObject*> params,
    PythonAPIInfo::InferredAttributes* inferred_attrs);

// Copies any parameters that expect a list of tensors to a new list.
// This ensures that any iterable value can be used, and also ensures that
// `ConvertPythonAPIParameters` can safely convert tensors in-place.
//
// Any values that are removed from `params` have their reference count
// decremented, and any objects added to `params` are new references.
//
// Returns true on success, or sets an exception and returns false on error.
ABSL_MUST_USE_RESULT
bool CopyPythonAPITensorLists(const PythonAPIInfo& api_info,
                              absl::Span<PyObject*> params);

}  // namespace tensorflow

#endif  // TENSORFLOW_PYTHON_UTIL_PYTHON_API_PARAMETER_CONVERTER_H_
