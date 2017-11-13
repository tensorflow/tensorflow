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

#include "tensorflow/python/lib/core/py_seq_tensor.h"

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/python/lib/core/numpy.h"
#include "tensorflow/python/lib/core/safe_ptr.h"

namespace tensorflow {
namespace {

inline bool PyIsInstance(PyObject* obj, PyTypeObject* t) {
  return PyObject_IsInstance(obj, reinterpret_cast<PyObject*>(t));
}

inline PyObject* PyType(PyObject* obj) {
  return reinterpret_cast<PyObject*>(obj->ob_type);
}

bool IsPyString(PyObject* obj) {
  // TODO(josh11b): Support unicode strings in Python 2? bytearrays? NumPy string
  // types?
#if PY_MAJOR_VERSION >= 3
  return PyBytes_Check(obj) || PyUnicode_Check(obj);
#else
  return PyBytes_Check(obj);
#endif
}

bool IsPyInt(PyObject* obj) {
#if PY_MAJOR_VERSION >= 3
  return PyLong_Check(obj) ||
         PyIsInstance(obj, &PyIntegerArrType_Type);  // NumPy integers
#else
  return PyInt_Check(obj) || PyLong_Check(obj) ||
         PyIsInstance(obj, &PyIntegerArrType_Type);  // NumPy integers
#endif
}

bool IsPyFloat(PyObject* obj) {
  return PyFloat_Check(obj) ||
         PyIsInstance(obj, &PyFloatingArrType_Type);  // NumPy float types
}

// Converts Python object `c` that should hold a Python string into a
// C++ string in *out.  Returns nullptr on success, or a message on error.
// Defined below, but forward declared here for use in PyRepr.
const char* ConvertOneString(PyObject* v, string* out);

string PyRepr(PyObject* obj) {
  if (obj == nullptr) {
    return "<null>";
  }
  Safe_PyObjectPtr repr_obj = make_safe(PyObject_Repr(obj));
  if (repr_obj) {
    string repr_str;
    if (ConvertOneString(repr_obj.get(), &repr_str) == nullptr) {
      return repr_str;
    }
  }
  return "<error computing repr()>";
}

bool IsPyDimension(PyObject* obj) {
  const char* tp_name = obj->ob_type->tp_name;
  if (strcmp(tp_name, "Dimension") != 0) return false;
  bool ret =
      StringPiece(PyRepr(PyType(obj)))
          .ends_with("tensorflow.python.framework.tensor_shape.Dimension'>");
  return ret;
}

Status InferShapeAndType(PyObject* obj, TensorShape* shape, DataType* dtype) {
  while (true) {
    // We test strings first, in case a string is considered a sequence.
    if (IsPyString(obj)) {
      *dtype = DT_STRING;
    } else if (PySequence_Check(obj)) {
      auto length = PySequence_Length(obj);
      shape->AddDim(length);
      if (length > 0) {
        obj = PySequence_GetItem(obj, 0);
        continue;
      } else {
        *dtype = DT_INVALID;  // Invalid dtype for empty tensors.
      }
    } else if (IsPyFloat(obj)) {
      *dtype = DT_DOUBLE;
    } else if (PyBool_Check(obj) || PyIsInstance(obj, &PyBoolArrType_Type)) {
      // Have to test for bool before int, since IsInt(True/False) == true.
      *dtype = DT_BOOL;
    } else if (IsPyInt(obj)) {
      *dtype = DT_INT64;
    } else if (IsPyDimension(obj)) {
      *dtype = DT_INT64;
    } else if (PyComplex_Check(obj) ||
               PyIsInstance(obj, &PyComplexFloatingArrType_Type)) {  // NumPy
      *dtype = DT_COMPLEX128;
    } else {
      return errors::InvalidArgument("Attempt to convert a value (",
                                     PyRepr(obj),
                                     ") with an unsupported type (",
                                     PyRepr(PyType(obj)), ") to a Tensor.");
    }
    return Status::OK();
  }
}

// Error messages

const char ErrorConverting[] =
    "Error while converting Python sequence to Tensor.";
const char ErrorRectangular[] =
    "Can't convert non-rectangular Python sequence to Tensor.";
const char ErrorMixedTypes[] =
    "Can't convert Python sequence with mixed types to Tensor.";
const char ErrorOutOfRange[] =
    "Can't convert Python sequence with out-of-range integer to Tensor.";
const char ErrorOutOfRangeDouble[] =
    "Can't convert Python sequence with a value out of range for a "
    "double-precision float.";
const char ErrorConvertingUnicodeString[] =
    "Error converting unicode string while converting Python sequence to "
    "Tensor.";
const char ErrorFoundInt64[] =
    "Can't convert Python sequence with out-of-range integer to int32 Tensor.";
const char ErrorFoundFloat[] =
    "Can't convert Python sequence with floating point values to integer "
    "Tensor.";

// Template for defining a function for recursively convering obj into
// an array of TYPE using the conversion function CONVERT.
// Note that these helper functions require shape.dims() >= 1.

#define DEFINE_HELPER(FUNCTION, TYPE, TYPE_ENUM, CONVERT)                 \
  const char* FUNCTION##Helper(PyObject* obj, const TensorShape& shape,   \
                               TYPE** buf) {                              \
    if (TF_PREDICT_FALSE(obj == nullptr)) {                               \
      return ErrorConverting;                                             \
    }                                                                     \
    if (shape.dims() > 1) {                                               \
      /* Iterate over outer dim, and recursively convert each element. */ \
      const int64 s = shape.dim_size(0);                                  \
      if (TF_PREDICT_FALSE(s != PySequence_Length(obj))) {                \
        return ErrorRectangular;                                          \
      }                                                                   \
      TensorShape rest = shape;                                           \
      rest.RemoveDim(0);                                                  \
      for (int64 i = 0; i < s; ++i) {                                     \
        const char* error =                                               \
            FUNCTION##Helper(PySequence_GetItem(obj, i), rest, buf);      \
        if (TF_PREDICT_FALSE(error != nullptr)) return error;             \
      }                                                                   \
    } else {                                                              \
      Safe_PyObjectPtr seq = make_safe(PySequence_Fast(obj, ""));         \
      if (TF_PREDICT_FALSE(seq == nullptr)) return ErrorRectangular;      \
      const int64 s = shape.dim_size(0);                                  \
      if (TF_PREDICT_FALSE(s != PySequence_Fast_GET_SIZE(seq.get()))) {   \
        return ErrorRectangular;                                          \
      }                                                                   \
      PyObject** l = PySequence_Fast_ITEMS(seq.get());                    \
      for (int64 i = 0; i < s; ++i) {                                     \
        const char* error = CONVERT(l[i], *buf);                          \
        if (TF_PREDICT_FALSE(error != nullptr)) return error;             \
        ++*buf;                                                           \
      }                                                                   \
    }                                                                     \
    return nullptr;                                                       \
  }                                                                       \
  const char* FUNCTION(PyObject* obj, const TensorShape& shape,           \
                       Tensor* dest) {                                    \
    /* TODO(josh11b): Allocator & attributes? */                            \
    Tensor result(TYPE_ENUM, shape);                                      \
    if (shape.dims() == 0) { /* Scalar case */                            \
      TYPE value;                                                         \
      const char* error = CONVERT(obj, &value);                           \
      if (error != nullptr) return error;                                 \
      result.scalar<TYPE>()() = value;                                    \
    } else {                                                              \
      TYPE* buf = result.flat<TYPE>().data();                             \
      const char* error = FUNCTION##Helper(obj, shape, &buf);             \
      if (error != nullptr) return error;                                 \
    }                                                                     \
    *dest = result;                                                       \
    return nullptr;                                                       \
  }

// Int support

const char* ConvertOneInt64(PyObject* v, int64* out) {
#if PY_MAJOR_VERSION < 3
  if (TF_PREDICT_TRUE(PyInt_Check(v))) {
    *out = PyInt_AS_LONG(v);
    return nullptr;
  }
#endif
  if (TF_PREDICT_TRUE(PyLong_Check(v) || IsPyDimension(v))) {
    int overflow = 0;
    // Have to use LongLong for 64 bits, since long is 32 bits on Windows.
    *out = PyLong_AsLongLongAndOverflow(v, &overflow);
    if (TF_PREDICT_FALSE(overflow)) return ErrorOutOfRange;
    return nullptr;
  }
  if (PyIsInstance(v, &PyIntegerArrType_Type)) {  // NumPy integers
#if PY_MAJOR_VERSION < 3
    Safe_PyObjectPtr as_int = make_safe(PyNumber_Int(v));
#else
    Safe_PyObjectPtr as_int = make_safe(PyNumber_Long(v));
#endif
    return ConvertOneInt64(as_int.get(), out);
  }
  if (IsPyFloat(v)) return ErrorFoundFloat;
  return ErrorMixedTypes;
}

DEFINE_HELPER(ConvertInt64, int64, DT_INT64, ConvertOneInt64);

const char* ConvertOneInt32(PyObject* v, int32* out) {
  int64 i;
#if PY_MAJOR_VERSION < 3
  if (TF_PREDICT_TRUE(PyInt_Check(v))) {
    i = PyInt_AS_LONG(v);
  } else
#endif
      if (PyLong_Check(v) || IsPyDimension(v)) {
    int overflow = 0;
    // Have to use LongLong for 64 bits, since long is 32 bits on Windows.
    i = PyLong_AsLongLongAndOverflow(v, &overflow);
    if (TF_PREDICT_FALSE(overflow)) return ErrorOutOfRange;
  } else if (PyIsInstance(v, &PyIntegerArrType_Type)) {  // NumPy integers
#if PY_MAJOR_VERSION < 3
    Safe_PyObjectPtr as_int = make_safe(PyNumber_Int(v));
#else
    Safe_PyObjectPtr as_int = make_safe(PyNumber_Long(v));
#endif
    return ConvertOneInt32(as_int.get(), out);
  } else if (IsPyFloat(v)) {
    return ErrorFoundFloat;
  } else {
    return ErrorMixedTypes;
  }
  *out = static_cast<uint32>(static_cast<uint64>(i));
  // Check for 32-bit overflow.
  if (TF_PREDICT_FALSE(i != *out)) return ErrorFoundInt64;
  return nullptr;
}

DEFINE_HELPER(ConvertInt32, int32, DT_INT32, ConvertOneInt32);

// Floating-point support

template <class T>
const char* ConvertOneFloat(PyObject* v, T* out) {
  if (TF_PREDICT_TRUE(PyFloat_Check(v))) {
    *out = PyFloat_AS_DOUBLE(v);
    return nullptr;
  }
#if PY_MAJOR_VERSION < 3
  if (PyInt_Check(v)) {
    *out = PyInt_AS_LONG(v);
    return nullptr;
  }
#endif
  if (PyLong_Check(v)) {
    *out = PyLong_AsDouble(v);
    if (PyErr_Occurred()) return ErrorOutOfRangeDouble;
    return nullptr;
  }
  if (PyIsInstance(v, &PyFloatingArrType_Type)) {  // NumPy float types
    Safe_PyObjectPtr as_float = make_safe(PyNumber_Float(v));
    return ConvertOneFloat<T>(as_float.get(), out);
  }
  if (PyIsInstance(v, &PyIntegerArrType_Type)) {  // NumPy integers
#if PY_MAJOR_VERSION < 3
    Safe_PyObjectPtr as_int = make_safe(PyNumber_Int(v));
#else
    Safe_PyObjectPtr as_int = make_safe(PyNumber_Long(v));
#endif
    return ConvertOneFloat<T>(as_int.get(), out);
  }
  return ErrorMixedTypes;
}

DEFINE_HELPER(ConvertDouble, double, DT_DOUBLE, ConvertOneFloat<double>);
DEFINE_HELPER(ConvertFloat, float, DT_FLOAT, ConvertOneFloat<float>);

// String support

const char* ConvertOneString(PyObject* v, string* out) {
  if (PyBytes_Check(v)) {
    out->assign(PyBytes_AS_STRING(v), PyBytes_GET_SIZE(v));
    return nullptr;
  }
#if PY_MAJOR_VERSION >= 3
  if (PyUnicode_Check(v)) {
    Py_ssize_t size;
    const char* str = PyUnicode_AsUTF8AndSize(v, &size);
    if (str == nullptr) return ErrorConvertingUnicodeString;
    out->assign(str, size);
    return nullptr;
  }
#endif
  return ErrorMixedTypes;
}

DEFINE_HELPER(ConvertString, string, DT_STRING, ConvertOneString);

// Complex support

const char* ConvertOneComplex(PyObject* v, complex128* out) {
  if (PyComplex_Check(v)) {
    *out = complex128(PyComplex_RealAsDouble(v), PyComplex_ImagAsDouble(v));
    return nullptr;
  } else if (PyIsInstance(v, &PyComplexFloatingArrType_Type)) {  // NumPy
    auto as_complex = PyComplex_AsCComplex(v);
    *out = complex128(as_complex.real, as_complex.imag);
    return nullptr;
  }
  return ErrorMixedTypes;
}

DEFINE_HELPER(ConvertComplex, complex128, DT_COMPLEX128, ConvertOneComplex);

// Bool support

const char* ConvertOneBool(PyObject* v, bool* out) {
  if (v == Py_True) {
    *out = true;
  } else if (v == Py_False) {
    *out = false;
  } else if (PyIsInstance(v, &PyBoolArrType_Type)) {  // NumPy
    *out = PyObject_IsTrue(v);
  } else {
    return ErrorMixedTypes;
  }
  return nullptr;
}

DEFINE_HELPER(ConvertBool, bool, DT_BOOL, ConvertOneBool);

#undef DEFINE_HELPER

}  // namespace

#define RETURN_STRING_AS_STATUS(...)                             \
  do {                                                           \
    const char* _error = (__VA_ARGS__);                          \
    if (TF_PREDICT_TRUE(_error == nullptr)) return Status::OK(); \
    return errors::InvalidArgument(_error);                      \
  } while (0)

Status PySeqToTensor(PyObject* obj, PyObject* dtype, Tensor* ret) {
  DataType infer_dtype;
  TensorShape shape;
  TF_RETURN_IF_ERROR(InferShapeAndType(obj, &shape, &infer_dtype));
  DataType requested_dtype = DT_INVALID;
  if (dtype != Py_None) {
    int32 dtype_as_int = -1;
    if (ConvertOneInt32(dtype, &dtype_as_int) == nullptr) {
      requested_dtype = static_cast<DataType>(dtype_as_int);
    }
  }
  // NOTE(josh11b): If don't successfully convert to the requested type,
  // we just try instead to create a tensor of the inferred type and
  // let the caller convert it to the requested type using a cast
  // operation.
  switch (requested_dtype) {
    case DT_FLOAT:
      if (ConvertFloat(obj, shape, ret) == nullptr) return Status::OK();
      break;

    case DT_DOUBLE:
      if (ConvertDouble(obj, shape, ret) == nullptr) return Status::OK();
      break;

    case DT_INT64:
      if (ConvertInt64(obj, shape, ret) == nullptr) return Status::OK();
      break;

    case DT_INT32:
      if (ConvertInt32(obj, shape, ret) == nullptr) return Status::OK();
      break;

    case DT_COMPLEX128:
      if (ConvertComplex(obj, shape, ret) == nullptr) return Status::OK();
      break;

    case DT_STRING:
      if (ConvertString(obj, shape, ret) == nullptr) return Status::OK();
      break;

    case DT_BOOL:
      if (ConvertBool(obj, shape, ret) == nullptr) return Status::OK();
      break;

    default:
      break;
  }
  switch (infer_dtype) {
    case DT_DOUBLE:
      // TODO(josh11b): Handle mixed floats and complex numbers?
      if (requested_dtype == DT_INVALID) {
        // TensorFlow uses float32s to represent floating point numbers
        // by default (for space and speed over using doubles).
        RETURN_STRING_AS_STATUS(ConvertFloat(obj, shape, ret));
      } else {
        // We are going to do a cast to the user's requested dtype
        // after this.  We use doubles for this intermediate result so
        // we don't lose precision that might be representable in the
        // final type.
        RETURN_STRING_AS_STATUS(ConvertDouble(obj, shape, ret));
      }

    case DT_INT64:
      if (requested_dtype == DT_INVALID) {
        const char* error = ConvertInt32(obj, shape, ret);
        if (error == ErrorFoundInt64) {
          error = ConvertInt64(obj, shape, ret);
        }
        if (error == ErrorFoundFloat) {
          error = ConvertFloat(obj, shape, ret);
        }
        // TODO(josh11b): May also want to fall back to using doubles if
        // error == ErrorOutOfRange?
        RETURN_STRING_AS_STATUS(error);
      } else {
        const char* error = ConvertInt64(obj, shape, ret);
        if (error == ErrorFoundFloat) {
          error = ConvertDouble(obj, shape, ret);
        }
        RETURN_STRING_AS_STATUS(error);
      }

    case DT_STRING:
      RETURN_STRING_AS_STATUS(ConvertString(obj, shape, ret));

    case DT_COMPLEX128:
      RETURN_STRING_AS_STATUS(ConvertComplex(obj, shape, ret));

    case DT_BOOL:
      RETURN_STRING_AS_STATUS(ConvertBool(obj, shape, ret));

    case DT_INVALID:  // Only occurs for empty tensors.
      *ret = Tensor(requested_dtype == DT_INVALID ? DT_FLOAT : requested_dtype,
                    shape);
      return Status::OK();

    default:
      return errors::Unimplemented("Missing Python -> Tensor conversion for ",
                                   DataTypeString(infer_dtype));
  }

  return Status::OK();
}

}  // namespace tensorflow
