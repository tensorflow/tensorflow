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
#include "tensorflow/python/framework/op_def_util.h"

#include <map>

#include "absl/strings/str_cat.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/python/util/util.h"

using ::tensorflow::swig::GetRegisteredPyObject;

#if PY_MAJOR_VERSION < 3
#define PY_STRING_CHECK(x) (PyString_Check(x) || PyUnicode_Check(x))
#define PY_INT_CHECK(x) (PyInt_Check(x))
#define PY_INT_TYPE PyInt_Type
#else
#define PY_STRING_CHECK(x) (PyBytes_Check(x) || PyUnicode_Check(x))
#define PY_INT_CHECK(x) (PyLong_Check(x))
#define PY_INT_TYPE PyLong_Type
#endif

namespace tensorflow {

namespace {

const std::map<std::string, AttributeType>* AttributeTypeNameMap() {
  static auto* type_map = new std::map<std::string, AttributeType>(
      {{"any", AttributeType::ANY},
       {"float", AttributeType::FLOAT},
       {"int", AttributeType::INT},
       {"string", AttributeType::STRING},
       {"bool", AttributeType::BOOL},
       {"shape", AttributeType::SHAPE},
       {"type", AttributeType::DTYPE},
       {"tensor", AttributeType::TENSOR},
       {"list(any)", AttributeType::LIST_ANY},
       {"list(float)", AttributeType::LIST_FLOAT},
       {"list(int)", AttributeType::LIST_INT},
       {"list(string)", AttributeType::LIST_STRING},
       {"list(bool)", AttributeType::LIST_BOOL},
       {"list(type)", AttributeType::LIST_DTYPE},
       {"list(shape)", AttributeType::LIST_SHAPE},
       {"list(tensor)", AttributeType::LIST_TENSOR}});
  return type_map;
}

// Note: we define functors for converting value types (rather than simple
// functions) so we can define a generic ConvertListAttr method.  These
// functors all return a new reference on success, or nullptr on failure.
// They do not (necessarily) call PyErr_SetString.

struct ConvertAnyFunctor {
  Safe_PyObjectPtr operator()(PyObject* value) {
    Py_INCREF(value);
    return Safe_PyObjectPtr(value);
  }
};

struct ConvertFloatFunctor {
  Safe_PyObjectPtr operator()(PyObject* value) {
    Safe_PyObjectPtr result;
    if (PyFloat_Check(value)) {
      Py_INCREF(value);
      result.reset(value);
    } else if (!PY_STRING_CHECK(value)) {
      result.reset(PyObject_CallFunctionObjArgs(
          reinterpret_cast<PyObject*>(&PyFloat_Type), value, nullptr));
    }
    return result;
  }
};

struct ConvertIntFunctor {
  Safe_PyObjectPtr operator()(PyObject* value) {
    Safe_PyObjectPtr result;
    if (PY_INT_CHECK(value)) {
      Py_INCREF(value);
      result.reset(value);
    } else if (!PY_STRING_CHECK(value)) {
      result.reset(PyObject_CallFunctionObjArgs(
          reinterpret_cast<PyObject*>(&PY_INT_TYPE), value, nullptr));
    }
    return result;
  }
};

struct ConvertStringFunctor {
  Safe_PyObjectPtr operator()(PyObject* value) {
    Safe_PyObjectPtr result;
    if (PY_STRING_CHECK(value)) {
      Py_INCREF(value);
      result.reset(value);
    }
    return result;
  }
};

// TODO(edloper): Should we allow ints (or any other values) to be converted
// to booleans?  Currently, TensorFlow does not do this conversion for attribute
// values in _MakeBool or make_bool.
struct ConvertBoolFunctor {
  Safe_PyObjectPtr operator()(PyObject* value) {
    Safe_PyObjectPtr result;
    if (PyBool_Check(value)) {
      Py_INCREF(value);
      result.reset(value);
    }
    return result;
  }
};

struct ConvertDTypeFunctor {
  Safe_PyObjectPtr operator()(PyObject* value) {
    Safe_PyObjectPtr result;
    // The following symbols are registered in op_def_library.py
    static PyObject* dtype = GetRegisteredPyObject("tf.dtypes.DType");
    static PyObject* as_dtype = GetRegisteredPyObject("tf.dtypes.as_dtype");
    if (reinterpret_cast<PyObject*>(value->ob_type) == dtype) {
      Py_INCREF(value);
      result.reset(value);
    } else {
      result.reset(PyObject_CallFunctionObjArgs(as_dtype, value, nullptr));
    }
    return result;
  }
};

struct ConvertTensorShapeFunctor {
  Safe_PyObjectPtr operator()(PyObject* value) {
    Safe_PyObjectPtr result;
    // The following symbols are registered in op_def_library.py
    static PyObject* shape = GetRegisteredPyObject("tf.TensorShape");
    static PyObject* as_shape = GetRegisteredPyObject("tf.as_shape");
    if (reinterpret_cast<PyObject*>(value->ob_type) == shape) {
      Py_INCREF(value);
      result.reset(value);
    } else {
      result.reset(PyObject_CallFunctionObjArgs(as_shape, value, nullptr));
    }
    return result;
  }
};

struct ConvertTensorProtoFunctor {
  Safe_PyObjectPtr operator()(PyObject* value) {
    Safe_PyObjectPtr result;
    // The following symbols are registered in op_def_library.py
    static PyObject* tensor_proto = GetRegisteredPyObject("tf.TensorProto");
    static PyObject* text_format_parse =
        GetRegisteredPyObject("text_format.Parse");
    if (reinterpret_cast<PyObject*>(value->ob_type) == tensor_proto) {
      Py_INCREF(value);
      result.reset(value);
    } else if (PY_STRING_CHECK(value)) {
      result.reset(PyObject_CallObject(tensor_proto, nullptr));
      if (result) {
        PyObject_CallFunctionObjArgs(text_format_parse, value, result.get(),
                                     nullptr);
      }
    }
    return result;
  }
};

// Converts `value` to a list of elements with the same type, using
// `convert_functor` to convert each element.
template <typename T>
Safe_PyObjectPtr ConvertListAttr(PyObject* value, T convert_functor) {
  // Copy the list.
  Safe_PyObjectPtr result(PySequence_List(value));
  if (!result) return nullptr;

  // Check the type of each item in the list.
  Py_ssize_t len = PySequence_Fast_GET_SIZE(result.get());
  PyObject** items = PySequence_Fast_ITEMS(result.get());
  for (Py_ssize_t i = 0; i < len; ++i) {
    if (!PyFloat_Check(value)) {
      Safe_PyObjectPtr item = convert_functor(items[i]);
      if (!item) return nullptr;
      PySequence_SetItem(result.get(), i, item.get());
    }
  }
  return result;
}

// Returns the given `value` value, converted to the indicated type.
// Returns nullptr if `value` is not convertible.
Safe_PyObjectPtr ConvertAttrOrNull(PyObject* value, AttributeType attr_type) {
  switch (attr_type) {
    case AttributeType::ANY:
      return ConvertAnyFunctor()(value);
    case AttributeType::FLOAT:
      return ConvertFloatFunctor()(value);
    case AttributeType::INT:
      return ConvertIntFunctor()(value);
    case AttributeType::STRING:
      return ConvertStringFunctor()(value);
    case AttributeType::BOOL:
      return ConvertBoolFunctor()(value);
    case AttributeType::DTYPE:
      return ConvertDTypeFunctor()(value);
    case AttributeType::SHAPE:
      return ConvertTensorShapeFunctor()(value);
    case AttributeType::TENSOR:
      return ConvertTensorProtoFunctor()(value);
    case AttributeType::LIST_ANY:
      return ConvertListAttr(value, ConvertAnyFunctor());
    case AttributeType::LIST_FLOAT:
      return ConvertListAttr(value, ConvertFloatFunctor());
    case AttributeType::LIST_INT:
      return ConvertListAttr(value, ConvertIntFunctor());
    case AttributeType::LIST_STRING:
      return ConvertListAttr(value, ConvertStringFunctor());
    case AttributeType::LIST_BOOL:
      return ConvertListAttr(value, ConvertBoolFunctor());
    case AttributeType::LIST_DTYPE:
      return ConvertListAttr(value, ConvertDTypeFunctor());
    case AttributeType::LIST_SHAPE:
      return ConvertListAttr(value, ConvertTensorShapeFunctor());
    case AttributeType::LIST_TENSOR:
      return ConvertListAttr(value, ConvertTensorProtoFunctor());
    default:
      return nullptr;
  }
}

}  // namespace

AttributeType AttributeTypeFromName(const std::string& type_name) {
  const auto* type_map = AttributeTypeNameMap();
  auto it = type_map->find(type_name);
  return it != type_map->end() ? it->second : AttributeType::UNKNOWN;
}

std::string AttributeTypeToName(AttributeType attr_type) {
  for (const auto& pair : *AttributeTypeNameMap()) {
    if (pair.second == attr_type) {
      return pair.first;
    }
  }
  return "<unknown>";
}

Safe_PyObjectPtr ConvertPyObjectToAttributeType(PyObject* value,
                                                AttributeType type) {
  Safe_PyObjectPtr result = ConvertAttrOrNull(value, type);
  if (!result) {
    auto err = absl::StrCat("Failed to convert value of type '",
                            value->ob_type->tp_name, "' to type '",
                            AttributeTypeToName(type), "'.");
    PyErr_SetString(PyExc_TypeError, err.c_str());
  }

  return result;
}

}  // namespace tensorflow
