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
#ifndef TENSORFLOW_PYTHON_FRAMEWORK_OP_DEF_UTIL_H_
#define TENSORFLOW_PYTHON_FRAMEWORK_OP_DEF_UTIL_H_

#include <Python.h>

#include <string>

#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/python/lib/core/safe_pyobject_ptr.h"

namespace tensorflow {

// Enumerated type corresponding with string values in AttrDef::type.
enum class AttributeType {
  UNKNOWN,
  ANY,          // "any"
  FLOAT,        // "float"
  INT,          // "int"
  STRING,       // "string"
  BOOL,         // "bool"
  DTYPE,        // "type" (tf.dtypes.DType)
  SHAPE,        // "shape" (tf.TensorShape)
  TENSOR,       // "tensor" (tf.TensorProto)
  LIST_ANY,     // "list(any)"
  LIST_FLOAT,   // "list(float)"
  LIST_INT,     // "list(int)"
  LIST_STRING,  // "list(string)"
  LIST_BOOL,    // "list(bool)"
  LIST_DTYPE,   // "list(dtype)"
  LIST_SHAPE,   // "list(shape)"
  LIST_TENSOR   // "list(tensor)"
};

// Returns the enumerated value corresponding to a given string (e.g.
// "string" or "list(string)".
AttributeType AttributeTypeFromName(const std::string& type_name);

// Returns the string corresponding to a given enumerated value.
std::string AttributeTypeToName(AttributeType attr_type);

// Converts `value` to the specified type and returns a new reference to the
// converted value (if possible); or sets a Python exception and returns
// nullptr.  This function is optimized to be fast if `value` already has the
// desired type.
//
//   * 'any' values are returned as-is.
//   * 'float' values are converted by calling float(value).
//   * 'int' values are converted by calling int(value).
//   * 'string' values are returned as-is if they are (bytes, unicode);
//     otherwise, an exception is raised.
//   * 'bool' values are returned as-is if they are boolean; otherwise, an
//     exception is raised.
//   * 'dtype' values are converted using `dtypes.as_dtype`.
//   * 'shape' values are converted using `tensor_shape.as_shape`.
//   * 'tensor' values are returned as-is if they are a `TensorProto`; or are
//     parsed into `TensorProto` using `textformat.merge` if they are a string.
//     Otherwise, an exception is raised.
//   * 'list(*)' values are copied to a new list, and then each element is
//     converted (in-place) as described above.  (If the value is not iterable,
//     or if conversion fails for any item, then an exception is raised.)
Safe_PyObjectPtr ConvertPyObjectToAttributeType(PyObject* value,
                                                AttributeType type);

// Converts a c++ `AttrValue` protobuf message to a Python object; or sets a
// Python exception and returns nullptr if an error occurs.
Safe_PyObjectPtr AttrValueToPyObject(const AttrValue& attr_value);

// Converts a c++ `DataType` protobuf enum to a Python object; or sets a
// Python exception and returns nullptr if an error occurs.
Safe_PyObjectPtr DataTypeToPyObject(const DataType& data_type);

// Converts a c++ `TensorShapeProto` message to a Python object; or sets a
// Python exception and returns nullptr if an error occurs.
Safe_PyObjectPtr TensorShapeProtoToPyObject(
    const TensorShapeProto& tensor_shape);

// TODO(edloper): Define TensorProtoToPyObject?

}  // namespace tensorflow

#endif  // TENSORFLOW_PYTHON_FRAMEWORK_OP_DEF_UTIL_H_
