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
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/python/lib/core/safe_pyobject_ptr.h"
#include "tensorflow/python/util/util.h"

using ::tensorflow::swig::GetRegisteredPyObject;

#if PY_MAJOR_VERSION < 3
// Python 2.x:
#define PY_STRING_CHECK(x) (PyString_Check(x) || PyUnicode_Check(x))
#define PY_STRING_FROMSTRING(x) (PyString_FromString(x))
#define PY_INT_CHECK(x) (PyInt_Check(x))
#define PY_INT_TYPE PyInt_Type
#define PY_INT_FROM_LONG(x) (PyInt_FromLong(x))
#else
// Python 3.x:
#define PY_STRING_CHECK(x) (PyBytes_Check(x) || PyUnicode_Check(x))
#define PY_STRING_FROMSTRING(x) (PyUnicode_FromString(x))
#define PY_INT_CHECK(x) (PyLong_Check(x))
#define PY_INT_TYPE PyLong_Type
#define PY_INT_FROM_LONG(x) (PyLong_FromLong(x))
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
        if (!PyObject_CallFunctionObjArgs(text_format_parse, value,
                                          result.get(), nullptr)) {
          return nullptr;
        }
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

// Returns a new reference to Py_True or Py_False depending on b.
PyObject* PyBool_FromBool(bool b) {
  PyObject* result = b ? Py_True : Py_False;
  Py_INCREF(result);
  return result;
}

Safe_PyObjectPtr AttrValueListToPyObject(AttrValue::ListValue list) {
  if (list.s_size()) {
    Safe_PyObjectPtr result(PyList_New(list.s_size()));
    for (int i = 0; i < list.s_size(); ++i) {
      PyList_SET_ITEM(result.get(), i, PY_STRING_FROMSTRING(list.s(i).c_str()));
    }
    return result;
  } else if (list.i_size()) {
    Safe_PyObjectPtr result(PyList_New(list.i_size()));
    for (int i = 0; i < list.i_size(); ++i) {
      PyList_SET_ITEM(result.get(), i, PY_INT_FROM_LONG(list.i(i)));
    }
    return result;
  } else if (list.f_size()) {
    Safe_PyObjectPtr result(PyList_New(list.f_size()));
    for (int i = 0; i < list.f_size(); ++i) {
      PyList_SET_ITEM(result.get(), i, PyFloat_FromDouble(list.f(i)));
    }
    return result;
  } else if (list.b_size()) {
    Safe_PyObjectPtr result(PyList_New(list.b_size()));
    for (int i = 0; i < list.b_size(); ++i) {
      PyList_SET_ITEM(result.get(), i, PyBool_FromBool(list.b(i)));
    }
    return result;
  } else if (list.type_size()) {
    Safe_PyObjectPtr result(PyList_New(list.type_size()));
    for (int i = 0; i < list.type_size(); ++i) {
      Safe_PyObjectPtr item(DataTypeToPyObject(list.type(i)));
      Py_INCREF(item.get());
      PyList_SET_ITEM(result.get(), i, item.get());
    }
    return result;
  } else if (list.shape_size()) {
    Safe_PyObjectPtr result(PyList_New(list.shape_size()));
    for (int i = 0; i < list.shape_size(); ++i) {
      Safe_PyObjectPtr item(TensorShapeProtoToPyObject(list.shape(i)));
      Py_INCREF(item.get());
      PyList_SET_ITEM(result.get(), i, item.get());
    }
    return result;
  } else if (list.tensor_size() || list.func_size()) {
    // TODO(edloper): Add support for tensorflow::AttrValue::kTensor.
    PyErr_SetString(PyExc_TypeError, "Unsupported AttrValue type");
    return nullptr;
  } else {
    // Empty list
    return Safe_PyObjectPtr(PyList_New(0));
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

Safe_PyObjectPtr AttrValueToPyObject(const AttrValue& attr_value) {
  switch (attr_value.value_case()) {
    case tensorflow::AttrValue::kS:
      return Safe_PyObjectPtr(PY_STRING_FROMSTRING(attr_value.s().c_str()));
    case tensorflow::AttrValue::kI:
      return Safe_PyObjectPtr(PY_INT_FROM_LONG(attr_value.i()));
    case tensorflow::AttrValue::kF:
      return Safe_PyObjectPtr(PyFloat_FromDouble(attr_value.f()));
    case tensorflow::AttrValue::kB:
      return Safe_PyObjectPtr(PyBool_FromBool(attr_value.b()));
    case tensorflow::AttrValue::kType:
      return DataTypeToPyObject(attr_value.type());
    case tensorflow::AttrValue::kShape:
      return TensorShapeProtoToPyObject(attr_value.shape());
    case tensorflow::AttrValue::kList:
      return AttrValueListToPyObject(attr_value.list());
    default:
      // TODO(edloper): Add support for tensorflow::AttrValue::kTensor.
      PyErr_SetString(PyExc_ValueError, "Unsupported AttrValue type");
      return nullptr;
  }
}

Safe_PyObjectPtr DataTypeToPyObject(const DataType& data_type) {
  Safe_PyObjectPtr enum_value(PY_INT_FROM_LONG(data_type));
  return ConvertDTypeFunctor()(enum_value.get());
}

Safe_PyObjectPtr TensorShapeProtoToPyObject(
    const TensorShapeProto& tensor_shape) {
  if (tensor_shape.unknown_rank()) {
    return ConvertTensorShapeFunctor()(Py_None);
  } else {
    Safe_PyObjectPtr dims(PyTuple_New(tensor_shape.dim_size()));
    for (int i = 0; i < tensor_shape.dim_size(); ++i) {
      PyTuple_SET_ITEM(dims.get(), i,
                       PY_INT_FROM_LONG(tensor_shape.dim(i).size()));
    }
    return ConvertTensorShapeFunctor()(dims.get());
  }
}

}  // namespace tensorflow
