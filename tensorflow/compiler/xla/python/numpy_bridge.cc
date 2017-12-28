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

bool CheckPyShapeInfo(PyObject* o) {
  // The object is a tuple (a pair)
  if (!PyTuple_Check(o)) {
    PyErr_SetString(PyExc_TypeError, "Shape record must be a tuple");
    return false;
  }
  if (PyTuple_Size(o) != 2) {
    PyErr_SetString(PyExc_ValueError, "Shape record tuple must be of length 2");
    return false;
  }

  // It has a first element, which is a numpy dtype object
  PyObject* first = PyTuple_GetItem(o, 0);
  if (!first) {
    return false;
  }
  if (first->ob_type != &PyArrayDescr_Type) {
    PyErr_SetString(
        PyExc_TypeError,
        "Shape record does not have a numpy dtype as its first element");
    return false;
  }
  const int np_type = NumpyTypenum(first);
  if (!NumpyTypeIsValid(np_type)) {
    PyErr_SetString(PyExc_ValueError,
                    "Shape record has an invalid integer dtype");
    return false;
  }

  // It has a second element, which is a tuple, either of shape
  // records or of Python ints
  PyObject* second = PyTuple_GetItem(o, 1);
  if (!second) {
    return false;
  }
  if (!PyTuple_Check(second)) {
    PyErr_SetString(PyExc_TypeError,
                    "Shape record does not have a tuple as its second element");
    return false;
  }
  const int length = PyTuple_Size(second);
  const PrimitiveType element_type = NumpyTypeToPrimitiveType(np_type);
  for (int i = 0; i < length; i++) {
    PyObject* dimension = PyTuple_GetItem(second, i);
    if (element_type == TUPLE) {
      if (!CheckPyShapeInfo(dimension)) {
        return false;
      }
    } else if (!CheckPyIntOrLong(dimension)) {
      PyErr_SetString(PyExc_TypeError,
                      "Non-tuple shape record has a non-integer dimension");
      return false;
    }
  }

  return true;
}

// Precondition: CheckPyShapeInfo(o)
Shape XlaShapeFromPyShapeInfo(PyObject* o) {
  const int np_type = NumpyTypenum(PyTuple_GetItem(o, 0));
  const PrimitiveType element_type = NumpyTypeToPrimitiveType(np_type);
  PyObject* py_dimensions = PyTuple_GetItem(o, 1);
  const int length = PyTuple_Size(py_dimensions);
  if (element_type == TUPLE) {
    std::vector<Shape> subshapes;
    subshapes.reserve(length);
    for (int i = 0; i < length; i++) {
      subshapes.push_back(
          XlaShapeFromPyShapeInfo(PyTuple_GetItem(py_dimensions, i)));
    }
    return ShapeUtil::MakeTupleShape(subshapes);
  } else {
    std::vector<int64> dimensions(length);
    for (int i = 0; i < length; i++) {
      dimensions[i] = PyIntOrPyLongToLong(PyTuple_GetItem(py_dimensions, i));
      if (dimensions[i] == -1) {
        CHECK(!PyErr_Occurred());
      }
    }
    return ShapeUtil::MakeShape(element_type, dimensions);
  }
}

PyObject* PyObjectFromXlaLiteral(const Literal& literal) {
  if (ShapeUtil::IsTuple(literal.shape())) {
    const std::vector<Literal>& tuple_literals = literal.tuple_literals();
    int num_elements = ShapeUtil::TupleElementCount(literal.shape());
    PyObject* tuple = PyTuple_New(num_elements);
    for (int i = 0; i < num_elements; i++) {
      PyTuple_SET_ITEM(tuple, i, PyObjectFromXlaLiteral(tuple_literals[i]));
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

std::unique_ptr<Literal> XlaLiteralFromPyObject(PyObject* o) {
  if (PyTuple_Check(o)) {
    int num_elements = PyTuple_Size(o);
    std::vector<std::unique_ptr<Literal>> elements;
    elements.reserve(num_elements);
    for (int i = 0; i < num_elements; i++) {
      PyObject* element = PyTuple_GetItem(o, i);
      elements.push_back(XlaLiteralFromPyObject(element));
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
    CopyNumpyArrayToLiteral(np_type, py_array, literal.get());
    return literal;
  } else {
    LOG(FATAL)
        << "Non-tuple or Numpy array encountered in conversion to XLA literal";
  }
}

void CopyNumpyArrayToLiteral(int np_type, PyArrayObject* py_array,
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
      LOG(FATAL) << "No XLA literal container for Numpy type" << np_type;
  }
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
