/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "pybind11/detail/common.h"
#include "pybind11/pybind11.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"

namespace {

inline int DataTypeId(tensorflow::DataType dt) { return static_cast<int>(dt); }

// A variant of tensorflow::DataTypeString which uses fixed-width names
// for floating point data types. This behavior is compatible with that of
// existing pure Python DType.
const std::string DataTypeStringCompat(tensorflow::DataType dt) {
  switch (dt) {
    case tensorflow::DataType::DT_HALF:
      return "float16";
    case tensorflow::DataType::DT_HALF_REF:
      return "float16_ref";
    case tensorflow::DataType::DT_FLOAT:
      return "float32";
    case tensorflow::DataType::DT_FLOAT_REF:
      return "float32_ref";
    case tensorflow::DataType::DT_DOUBLE:
      return "float64";
    case tensorflow::DataType::DT_DOUBLE_REF:
      return "float64_ref";
    default:
      return tensorflow::DataTypeString(dt);
  }
}

}  // namespace

namespace tensorflow {

constexpr DataTypeSet kNumPyIncompatibleTypes =
    ToSet(DataType::DT_RESOURCE) | ToSet(DataType::DT_VARIANT);

inline bool DataTypeIsNumPyCompatible(DataType dt) {
  return !kNumPyIncompatibleTypes.Contains(dt);
}

}  // namespace tensorflow

namespace py = pybind11;

PYBIND11_MODULE(_dtypes, m) {
  py::class_<tensorflow::DataType>(m, "DType")
      .def(py::init([](py::object obj) {
        auto id = static_cast<int>(py::int_(obj));
        if (tensorflow::DataType_IsValid(id) &&
            id != static_cast<int>(tensorflow::DT_INVALID)) {
          return static_cast<tensorflow::DataType>(id);
        }
        throw py::type_error(
            py::str("{} does not correspond to a valid tensorflow::DataType")
                .format(id));
      }))
      // For compatibility with pure-Python DType.
      .def_property_readonly("_type_enum", &DataTypeId)
      .def_property_readonly(
          "as_datatype_enum", &DataTypeId,
          "Returns a `types_pb2.DataType` enum value based on this data type.")

      .def_property_readonly("name",
                             [](tensorflow::DataType self) {
#if PY_MAJOR_VERSION < 3
                               return py::bytes(DataTypeStringCompat(self));
#else
                               return DataTypeStringCompat(self);
#endif
                             })
      .def_property_readonly(
          "size",
          [](tensorflow::DataType self) {
            return tensorflow::DataTypeSize(tensorflow::BaseType(self));
          })

      .def("__repr__",
           [](tensorflow::DataType self) {
             return py::str("tf.{}").format(DataTypeStringCompat(self));
           })
      .def("__str__",
           [](tensorflow::DataType self) {
             return py::str("<dtype: {!r}>")
#if PY_MAJOR_VERSION < 3
                 .format(py::bytes(DataTypeStringCompat(self)));
#else
                 .format(DataTypeStringCompat(self));
#endif
           })
      .def("__hash__", &DataTypeId)

      .def_property_readonly(
          "is_numpy_compatible",
          [](tensorflow::DataType self) {
            return tensorflow::DataTypeIsNumPyCompatible(
                tensorflow::BaseType(self));
          },
          "Returns whether this data type has a compatible NumPy data type.")

      .def_property_readonly(
          "is_bool",
          [](tensorflow::DataType self) {
            return tensorflow::BaseType(self) == tensorflow::DT_BOOL;
          },
          "Returns whether this is a boolean data type.")
      .def_property_readonly(
          "is_complex",
          [](tensorflow::DataType self) {
            return tensorflow::DataTypeIsComplex(tensorflow::BaseType(self));
          },
          "Returns whether this is a complex floating point type.")
      .def_property_readonly(
          "is_floating",
          [](tensorflow::DataType self) {
            return tensorflow::DataTypeIsFloating(tensorflow::BaseType(self));
          },
          "Returns whether this is a (non-quantized, real) floating point "
          "type.")
      .def_property_readonly(
          "is_integer",
          [](tensorflow::DataType self) {
            return tensorflow::DataTypeIsInteger(tensorflow::BaseType(self));
          },
          "Returns whether this is a (non-quantized) integer type.")
      .def_property_readonly(
          "is_quantized",
          [](tensorflow::DataType self) {
            return tensorflow::DataTypeIsQuantized(tensorflow::BaseType(self));
          },
          "Returns whether this is a quantized data type.")
      .def_property_readonly(
          "is_unsigned",
          [](tensorflow::DataType self) {
            return tensorflow::DataTypeIsUnsigned(tensorflow::BaseType(self));
          },
          R"doc(Returns whether this type is unsigned.

Non-numeric, unordered, and quantized types are not considered unsigned, and
this function returns `False`.)doc");
}
