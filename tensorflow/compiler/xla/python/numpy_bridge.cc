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

#include "tensorflow/compiler/xla/python/numpy_bridge.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {

namespace swig {

namespace numpy {

int PrimitiveTypeToNumpyType(PrimitiveType primitive_type) {
  switch (primitive_type) {
    case PRED:
      return NPY_BOOL;
    case S8:
      return NPY_INT8;
    case S16:
      return NPY_INT16;
    case S32:
      return NPY_INT32;
    case S64:
      return NPY_INT64;
    case U8:
      return NPY_UINT8;
    case U16:
      return NPY_UINT16;
    case U32:
      return NPY_UINT32;
    case U64:
      return NPY_UINT64;
    case F16:
      return NPY_FLOAT16;
    case F32:
      return NPY_FLOAT32;
    case F64:
      return NPY_FLOAT64;
    case TUPLE:
      return NPY_OBJECT;
    default:
      LOG(FATAL) << "No Numpy type for XLA primitive type " << primitive_type;
  }
}

PrimitiveType NumpyTypeToPrimitiveType(int np_type) {
  switch (np_type) {
    case NPY_BOOL:
      return PRED;
    case NPY_INT8:
      return S8;
    case NPY_INT16:
      return S16;
    case NPY_INT32:
      return S32;
    case NPY_INT64:
      return S64;
    case NPY_UINT8:
      return U8;
    case NPY_UINT16:
      return U16;
    case NPY_UINT32:
      return U32;
    case NPY_UINT64:
      return U64;
    case NPY_FLOAT16:
      return F16;
    case NPY_FLOAT32:
      return F32;
    case NPY_FLOAT64:
      return F64;
    case NPY_OBJECT:
      return TUPLE;
    default:
      LOG(FATAL) << "No XLA primitive type for Numpy type " << np_type;
  }
}

bool NumpyTypeIsValid(int np_type) {
  switch (np_type) {
    case NPY_BOOL:
    case NPY_INT8:
    case NPY_INT16:
    case NPY_INT32:
    case NPY_INT64:
    case NPY_UINT8:
    case NPY_UINT16:
    case NPY_UINT32:
    case NPY_UINT64:
    case NPY_FLOAT16:
    case NPY_FLOAT32:
    case NPY_FLOAT64:
    case NPY_OBJECT:
      return true;
    default:
      return false;
  }
}

PyObject* PyShapeInfoFromXlaShape(const Shape& shape) {
  int np_typenum = PrimitiveTypeToNumpyType(shape.element_type());
  PyArray_Descr* np_dtype = PyArray_DescrFromType(np_typenum);

  PyObject* dimensions;
  if (ShapeUtil::IsTuple(shape)) {
    int num_elements = ShapeUtil::TupleElementCount(shape);
    dimensions = PyTuple_New(ShapeUtil::TupleElementCount(shape));
    for (int i = 0; i < num_elements; ++i) {
      PyTuple_SET_ITEM(
          dimensions, i,
          PyShapeInfoFromXlaShape(ShapeUtil::GetTupleElementShape(shape, i)));
    }
  } else {
    int rank = ShapeUtil::Rank(shape);
    dimensions = PyTuple_New(rank);
    for (int i = 0; i < rank; ++i) {
      PyTuple_SET_ITEM(dimensions, i,
                       LongToPyIntOrPyLong(ShapeUtil::GetDimension(shape, i)));
    }
  }
  return PyTuple_Pack(2, np_dtype, dimensions);
}

// Precondition: o->ob_type == &PyArrayDescr_Type
static int NumpyTypenum(PyObject* o) {
  return reinterpret_cast<PyArray_Descr*>(o)->type_num;
}

// Extracts the string held inside r and returns it as a C++ string.
//
// NOTE: this is an internal helper for conversion to a C++, and so decrefs r.
static string ExtractStringAndDecref(PyObject* r) {
  auto error = [r] {
    return tensorflow::strings::Printf("<failed conversion of %p>", r);
  };
  if (r == nullptr) {
    return error();
  }
#if PY_MAJOR_VERSION < 3
  string result = PyString_AsString(r);
#else
  PyObject* bytes = PyUnicode_AsEncodedString(r, 0, 0);
  if (bytes == nullptr) {
    return error();
  }
  CHECK(PyBytes_Check(bytes));
  string result = PyBytes_AsString(bytes);
  Py_DECREF(bytes);
#endif
  Py_DECREF(r);
  return result;
}

// Safely returns a str of the given Python object o as a C++ string.
static string PyObjectCppStr(PyObject* o) {
  PyObject* s = PyObject_Str(o);
  return ExtractStringAndDecref(s);
}

string PyObjectCppRepr(PyObject* o) {
  PyObject* r = PyObject_Repr(o);
  return ExtractStringAndDecref(r);
}

StatusOr<Shape> XlaShapeFromPyShape(PyObject* o) {
  auto error = [o](const string& prefix) {
    return InvalidArgument("%s; got %s", prefix.c_str(),
                           PyObjectCppRepr(o).c_str());
  };

  auto get_attr = [o, &error](const string& field) -> StatusOr<PyObject*> {
    PyObject* result =
        PyObject_GetAttrString(o, const_cast<char*>(field.c_str()));
    if (result == nullptr) {
      return error(tensorflow::strings::StrCat(
          "Failed to get attribute of Shape object:", field));
    }
    return result;
  };

  auto call_method = [o, &error](const string& method) -> StatusOr<PyObject*> {
    PyObject* result =
        PyObject_CallMethod(o, const_cast<char*>(method.c_str()), nullptr);
    if (result == nullptr) {
      return error(tensorflow::strings::StrCat(
          "Failed to call method of shape object:", method));
    }
    return result;
  };

  PyObject* np_type;
  TF_ASSIGN_OR_RETURN(np_type, get_attr("np_dtype"));
  if (np_type->ob_type != &PyArrayDescr_Type) {
    return error("Shape attribute np_dtype is not an integer numpy dtype");
  }
  if (!NumpyTypeIsValid(NumpyTypenum(np_type))) {
    return error("Shape attribute np_dtype is not a valid integer numpy dtype");
  }
  const PrimitiveType element_type =
      NumpyTypeToPrimitiveType(NumpyTypenum(np_type));
  Py_DECREF(np_type);

  if (element_type == TUPLE) {
    PyObject* py_subshapes;
    TF_ASSIGN_OR_RETURN(py_subshapes, call_method("tuple_shapes"));
    if (!PyTuple_Check(py_subshapes)) {
      return error(
          "Return value of Shape method tuple_shapes() is not a tuple");
    }
    const int length = PyTuple_Size(py_subshapes);
    std::vector<Shape> subshapes;
    subshapes.reserve(length);
    for (int i = 0; i < length; i++) {
      TF_ASSIGN_OR_RETURN(
          const Shape& subshape,
          XlaShapeFromPyShape(PyTuple_GetItem(py_subshapes, i)));
      subshapes.push_back(subshape);
    }
    Py_DECREF(py_subshapes);
    return ShapeUtil::MakeTupleShape(subshapes);
  } else {
    PyObject* py_dimensions;
    PyObject* py_minor_to_major;
    TF_ASSIGN_OR_RETURN(py_dimensions, call_method("dimensions"));
    TF_ASSIGN_OR_RETURN(py_minor_to_major, call_method("minor_to_major"));
    if (!PyTuple_Check(py_dimensions)) {
      return error("Return value of Shape method dimensions() is not a tuple");
    }
    if (py_minor_to_major != Py_None && !PyTuple_Check(py_minor_to_major)) {
      return error(
          "Return value of Shape method minor_to_major() is neither a tuple "
          "nor None");
    }
    const int length = PyTuple_Size(py_dimensions);
    if (py_minor_to_major != Py_None &&
        length != PyTuple_Size(py_minor_to_major)) {
      return error(
          "Shape methods dimensions() and minor_to_major() return "
          "different-length tuples");
    }
    std::vector<int64> dimensions(length);
    std::vector<int64> minor_to_major(length);
    for (int i = 0; i < length; i++) {
      dimensions[i] = PyIntOrPyLongToLong(PyTuple_GetItem(py_dimensions, i));
      if (dimensions[i] == -1 && PyErr_Occurred()) {
        return error("Dimension is not an int");
      }

      if (py_minor_to_major != Py_None) {
        minor_to_major[i] =
            PyIntOrPyLongToLong(PyTuple_GetItem(py_minor_to_major, i));
        if (minor_to_major[i] == -1 && PyErr_Occurred()) {
          return error("Minor-to-major value is not an int");
        }
      }
    }
    bool with_layout = py_minor_to_major != Py_None;
    Py_DECREF(py_dimensions);
    Py_DECREF(py_minor_to_major);
    if (with_layout) {
      return ShapeUtil::MakeShapeWithLayout(element_type, dimensions,
                                            minor_to_major);
    } else {
      return ShapeUtil::MakeShape(element_type, dimensions);
    }
  }
}

// Helper that retrieves the member with attr_name, stringifies it if is not
// None, and returns it as a C++ string.
static tensorflow::gtl::optional<string> GetAttrAsString(
    PyObject* o, const string& attr_name) {
  if (!PyObject_HasAttrString(o, attr_name.c_str())) {
    return tensorflow::gtl::nullopt;
  }
  PyObject* attr = PyObject_GetAttrString(o, attr_name.c_str());
  if (attr == Py_None) {
    Py_DECREF(attr);
    return tensorflow::gtl::nullopt;
  }
  string result = PyObjectCppStr(attr);
  Py_DECREF(attr);
  return result;
}

// Helper that retrieves the member with attr_name, checks that it is an integer
// if it is not None, and returns it as an int32 value.
static tensorflow::gtl::optional<int32> GetAttrAsInt32(
    PyObject* o, const string& attr_name) {
  if (!PyObject_HasAttrString(o, attr_name.c_str())) {
    return tensorflow::gtl::nullopt;
  }
  PyObject* attr = PyObject_GetAttrString(o, attr_name.c_str());
  if (attr == Py_None) {
    Py_DECREF(attr);
    return tensorflow::gtl::nullopt;
  }
  if (!CheckPyIntOrLong(attr)) {
    Py_DECREF(attr);
    return tensorflow::gtl::nullopt;
  }
  long value = PyIntOrPyLongToLong(attr);  // NOLINT
  Py_DECREF(attr);
  if (value == -1 && PyErr_Occurred() != nullptr) {
    return tensorflow::gtl::nullopt;
  }
  if (static_cast<int32>(value) != value) {
    return tensorflow::gtl::nullopt;
  }
  return value;
}

StatusOr<OpMetadata> OpMetadataFromPyObject(PyObject* o) {
  OpMetadata result;
  tensorflow::gtl::optional<string> op_type = GetAttrAsString(o, "op_type");
  if (op_type.has_value()) {
    result.set_op_type(op_type.value());
  }
  tensorflow::gtl::optional<string> op_name = GetAttrAsString(o, "op_name");
  if (op_name.has_value()) {
    result.set_op_name(op_name.value());
  }
  tensorflow::gtl::optional<string> source_file =
      GetAttrAsString(o, "source_file");
  if (source_file.has_value()) {
    result.set_source_file(source_file.value());
  }
  tensorflow::gtl::optional<int32> source_line =
      GetAttrAsInt32(o, "source_line");
  if (source_line.has_value()) {
    result.set_source_line(source_line.value());
  }
  return result;
}

PyObject* PyObjectFromXlaLiteral(const Literal& literal) {
  if (ShapeUtil::IsTuple(literal.shape())) {
    int num_elements = ShapeUtil::TupleElementCount(literal.shape());
    PyObject* tuple = PyTuple_New(num_elements);
    for (int i = 0; i < num_elements; i++) {
      PyTuple_SET_ITEM(
          tuple, i, PyObjectFromXlaLiteral(LiteralView::Create(literal, {i})));
    }
    return tuple;
  } else {
    int rank = ShapeUtil::Rank(literal.shape());
    std::vector<long> dimensions(rank);  // NOLINT - PyArray requires a long*
    for (int i = 0; i < rank; i++) {
      dimensions[i] = ShapeUtil::GetDimension(literal.shape(), i);
    }
    int np_type = PrimitiveTypeToNumpyType(literal.shape().element_type());
    PyObject* array =
        PyArray_EMPTY(rank, dimensions.data(), np_type, /*fortran=*/0);
    CopyLiteralToNumpyArray(np_type, literal,
                            reinterpret_cast<PyArrayObject*>(array));
    return array;
  }
}

StatusOr<std::unique_ptr<Literal>> XlaLiteralFromPyObject(PyObject* o) {
  if (PyTuple_Check(o)) {
    int num_elements = PyTuple_Size(o);
    std::vector<std::unique_ptr<Literal>> elements;
    elements.reserve(num_elements);
    for (int i = 0; i < num_elements; i++) {
      PyObject* element = PyTuple_GetItem(o, i);
      TF_ASSIGN_OR_RETURN(auto literal, XlaLiteralFromPyObject(element));
      elements.push_back(std::move(literal));
    }
    return Literal::MakeTupleOwned(std::move(elements));
  } else if (PyArray_Check(o)) {
    PyArrayObject* py_array = reinterpret_cast<PyArrayObject*>(o);
    int rank = PyArray_NDIM(py_array);
    std::vector<int64> dimensions(rank);
    for (int i = 0; i < rank; i++) {
      dimensions[i] = PyArray_DIM(py_array, i);
    }
    int np_type = PyArray_TYPE(py_array);
    auto literal = Literal::CreateFromDimensions(
        NumpyTypeToPrimitiveType(np_type), dimensions);
    TF_RETURN_IF_ERROR(
        CopyNumpyArrayToLiteral(np_type, py_array, literal.get()));
    return std::move(literal);
  } else {
    return InvalidArgument(
        "Non-tuple or Numpy array encountered in conversion to XLA literal.");
  }
}

Status CopyNumpyArrayToLiteral(int np_type, PyArrayObject* py_array,
                               Literal* literal) {
  switch (np_type) {
    case NPY_BOOL:
      CopyNumpyArrayToLiteral<bool>(py_array, literal);
      break;
    case NPY_INT32:
      CopyNumpyArrayToLiteral<int32>(py_array, literal);
      break;
    case NPY_INT64:
      CopyNumpyArrayToLiteral<int64>(py_array, literal);
      break;
    case NPY_UINT8:
      CopyNumpyArrayToLiteral<uint8>(py_array, literal);
      break;
    case NPY_UINT32:
      CopyNumpyArrayToLiteral<uint32>(py_array, literal);
      break;
    case NPY_UINT64:
      CopyNumpyArrayToLiteral<uint64>(py_array, literal);
      break;
    case NPY_FLOAT16:
      CopyNumpyArrayToLiteral<half>(py_array, literal);
      break;
    case NPY_FLOAT32:
      CopyNumpyArrayToLiteral<float>(py_array, literal);
      break;
    case NPY_FLOAT64:
      CopyNumpyArrayToLiteral<double>(py_array, literal);
      break;
    default:
      return InvalidArgument(
          "No XLA literal container for Numpy type number: %d", np_type);
  }
  return Status::OK();
}

void CopyLiteralToNumpyArray(int np_type, const Literal& literal,
                             PyArrayObject* py_array) {
  switch (np_type) {
    case NPY_BOOL:
      CopyLiteralToNumpyArray<bool>(literal, py_array);
      break;
    case NPY_INT32:
      CopyLiteralToNumpyArray<int32>(literal, py_array);
      break;
    case NPY_INT64:
      CopyLiteralToNumpyArray<int64>(literal, py_array);
      break;
    case NPY_UINT8:
      CopyLiteralToNumpyArray<uint8>(literal, py_array);
      break;
    case NPY_UINT32:
      CopyLiteralToNumpyArray<uint32>(literal, py_array);
      break;
    case NPY_UINT64:
      CopyLiteralToNumpyArray<uint64>(literal, py_array);
      break;
    case NPY_FLOAT16:
      CopyLiteralToNumpyArray<half>(literal, py_array);
      break;
    case NPY_FLOAT32:
      CopyLiteralToNumpyArray<float>(literal, py_array);
      break;
    case NPY_FLOAT64:
      CopyLiteralToNumpyArray<double>(literal, py_array);
      break;
    default:
      LOG(FATAL) << "No XLA literal container for Numpy type" << np_type;
  }
}

PyObject* LongToPyIntOrPyLong(long x) {  // NOLINT
#if PY_MAJOR_VERSION < 3
  return PyInt_FromLong(x);
#else
  return PyLong_FromLong(x);
#endif
}

long PyIntOrPyLongToLong(PyObject* o) {  // NOLINT
#if PY_MAJOR_VERSION < 3
  return PyInt_AsLong(o);
#else
  return PyLong_AsLong(o);
#endif
}

bool CheckPyIntOrLong(PyObject* o) {
#if PY_MAJOR_VERSION < 3
  return PyInt_Check(o);
#else
  if (!PyLong_Check(o)) {
    return false;
  }
  int overflow = 0;
  PyLong_AsLongAndOverflow(o, &overflow);
  return (overflow == 0);
#endif
}

PyObject* PyNumberToPyInt(PyObject* o) {
#if PY_MAJOR_VERSION < 3
  return PyNumber_Int(o);
#else
  return PyNumber_Long(o);
#endif
}

}  // namespace numpy

}  // namespace swig

}  // namespace xla
