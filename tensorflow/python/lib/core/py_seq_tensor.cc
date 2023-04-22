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

#include "tensorflow/c/eager/tfe_context_internal.h"
#include "tensorflow/c/eager/tfe_tensorhandle_internal.h"
#include "tensorflow/c/tensor_interface.h"
#include "tensorflow/c/tf_tensor_internal.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/tstring.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/python/lib/core/ndarray_tensor.h"
#include "tensorflow/python/lib/core/ndarray_tensor_bridge.h"
#include "tensorflow/python/lib/core/numpy.h"
#include "tensorflow/python/lib/core/py_util.h"
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
  return PyBytes_Check(obj) || PyUnicode_Check(obj);
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

bool IsPyDouble(PyObject* obj) {
  return PyIsInstance(obj, &PyDoubleArrType_Type);  // NumPy double type.
}

bool IsNumpyHalf(PyObject* obj) {
  return PyIsInstance(obj, &PyHalfArrType_Type);
}

bool IsPyFloat(PyObject* obj) {
  return PyFloat_Check(obj) ||
         PyIsInstance(obj, &PyFloatingArrType_Type);  // NumPy float types
}

struct ConverterState {
  // The inferred tensor shape.
  gtl::InlinedVector<int64, 4> inferred_shape;

  // The inferred tensor data type.
  DataType inferred_dtype;

  // The following fields are used by ZeroDimArrayToScalar.
  // We cache the last result of the check for a zero dimensional array: the
  // function is called many times in a conversion, and most of the time is
  // to check for the same type. This cache can reduce the conversion time by
  // about 25%.
  PyTypeObject* last_zerodim_type;
  bool last_zerodim_check;

  ConverterState() : inferred_dtype(DT_INVALID), last_zerodim_type(nullptr) {}
};

// If the input is a zero dimensional PyArray return it converted to a scalar.
// Otherwise return the input and increment its reference count.
// Users must Py_DECREF the output of this method.
PyObject* ZeroDimArrayToScalar(PyObject* obj, ConverterState* state) {
  auto type = Py_TYPE(obj);
  auto pyarray_obj = reinterpret_cast<PyArrayObject*>(obj);
  if (type != state->last_zerodim_type) {
    state->last_zerodim_type = type;
    state->last_zerodim_check =
        PyObject_TypeCheck(obj, &PyArray_Type) &&
        !PyObject_TypeCheck(obj, &PyGenericArrType_Type);
  }

  if (state->last_zerodim_check && PyArray_NDIM(pyarray_obj) == 0) {
    obj = PyArray_ToScalar(PyArray_DATA(pyarray_obj), pyarray_obj);
  } else {
    Py_INCREF(obj);
  }
  return obj;
}

// Sets *elem to a NEW reference to an element in seq on success.
// REQUIRES: PySequence_Check(seq) && PySequence_Length(seq) > 0.
Status SampleElementFromSequence(PyObject* seq, PyObject** elem) {
  *elem = PySequence_GetItem(seq, 0);
  if (*elem != nullptr) return Status::OK();
  // seq may implement the sequence protocol (i.e., implement __getitem__)
  // but may legitimately not have a 0-th element (__getitem__(self, 0)
  // raises a KeyError). For example:
  // seq = pandas.Series([0, 1, 2], index=[2, 4, 6])
  //
  // We don't actually care for the element at key 0, any element will do
  // for inferring the element types. All elements are expected to
  // have the same type, and this will be validated when converting
  // to an EagerTensor.
  PyErr_Clear();
  Safe_PyObjectPtr iter(PyObject_GetIter(seq));
  if (PyErr_Occurred()) {
    return errors::InvalidArgument("Cannot infer dtype of a ",
                                   Py_TYPE(seq)->tp_name,
                                   " object: ", PyExceptionFetch());
  }
  *elem = PyIter_Next(iter.get());
  if (PyErr_Occurred()) {
    return errors::InvalidArgument(
        "Cannot infer dtype of a ", Py_TYPE(seq)->tp_name,
        " object, as iter(<object>).next() failed: ", PyExceptionFetch());
  }
  if (*elem == nullptr) {
    return errors::InvalidArgument("Cannot infer dtype of a ",
                                   Py_TYPE(seq)->tp_name,
                                   " object since it is an empty sequence");
  }
  return Status::OK();
}

tstring PyRepr(PyObject* obj);
bool IsPyDimension(PyObject* obj);

Status InferShapeAndType(PyObject* obj, ConverterState* state) {
  std::vector<Safe_PyObjectPtr> refs_to_clean;
  while (true) {
    // Convert any zero dimensional numpy arrays to scalars first of all.
    // We also have to make sure a reference to the safe_obj is kept.
    obj = ZeroDimArrayToScalar(obj, state);
    refs_to_clean.push_back(make_safe(obj));
    // We test strings first, in case a string is considered a sequence.
    if (IsPyString(obj)) {
      state->inferred_dtype = DT_STRING;
    } else if (PySequence_Check(obj)) {
      auto length = PySequence_Length(obj);
      if (length > 0) {
        state->inferred_shape.push_back(length);
        PyObject* elem = nullptr;
        TF_RETURN_IF_ERROR(SampleElementFromSequence(obj, &elem));
        obj = elem;
        refs_to_clean.push_back(make_safe(obj));
        continue;
      } else if (length == 0) {
        state->inferred_shape.push_back(length);
        state->inferred_dtype = DT_INVALID;  // Invalid dtype for empty tensors.
      } else {
        // The sequence does not have a valid length (PySequence_Length < 0).
        if (PyErr_Occurred()) {
          // PySequence_Length failed and set an exception. Fetch the message
          // and convert it to a failed status.
          return errors::InvalidArgument(PyExceptionFetch());
        } else {
          // This is almost certainly dead code: PySequence_Length failed but
          // did not set an exception.
          return errors::InvalidArgument(
              "Attempted to convert an invalid sequence to a Tensor.");
        }
      }
    } else if (IsPyDouble(obj)) {
      state->inferred_dtype = DT_DOUBLE;
    } else if (IsNumpyHalf(obj)) {
      state->inferred_dtype = DT_HALF;
    } else if (IsPyFloat(obj)) {
      state->inferred_dtype = DT_FLOAT;
    } else if (PyBool_Check(obj) || PyIsInstance(obj, &PyBoolArrType_Type)) {
      // Have to test for bool before int, since IsInt(True/False) == true.
      state->inferred_dtype = DT_BOOL;
    } else if (IsPyInt(obj)) {
      state->inferred_dtype = DT_INT64;
    } else if (IsPyDimension(obj)) {
      state->inferred_dtype = DT_INT64;
    } else if (PyComplex_Check(obj) ||
               PyIsInstance(obj, &PyComplexFloatingArrType_Type)) {  // NumPy
      state->inferred_dtype = DT_COMPLEX128;
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

// Defines a converter that recursively converts an object into
// an array of type T using the conversion function defined by the
// traits class in a ConvertScalar function.
//
// Note that these helper functions require shape.dims() >= 1.
template <class T>
struct ConverterTraits {
  static const tensorflow::DataType kTypeEnum;
  static const char* ConvertScalar(PyObject* v, T* out);
};

template <class T>
struct Converter {
  static const char* Helper(PyObject* obj, int depth, ConverterState* state,
                            T** buf) {
    if (TF_PREDICT_FALSE(obj == nullptr)) {
      return ErrorConverting;
    }

    Safe_PyObjectPtr seq = make_safe(PySequence_Fast(obj, ""));
    if (TF_PREDICT_FALSE(seq == nullptr)) return ErrorRectangular;

    const int64 s = state->inferred_shape[depth];
    if (TF_PREDICT_FALSE(s != PySequence_Fast_GET_SIZE(seq.get()))) {
      return ErrorRectangular;
    }

    if (state->inferred_shape.size() - depth > 1) {
      /* Iterate over outer dim, and recursively convert each element. */
      for (int64 i = 0; i < s; ++i) {
        const char* error = Helper(PySequence_Fast_GET_ITEM(seq.get(), i),
                                   depth + 1, state, buf);
        if (TF_PREDICT_FALSE(error != nullptr)) return error;
      }
    } else {
      PyObject** l = PySequence_Fast_ITEMS(seq.get());
      for (int64 i = 0; i < s; ++i) {
        auto scalar = ZeroDimArrayToScalar(l[i], state);
        const char* error = ConverterTraits<T>::ConvertScalar(scalar, *buf);
        Py_DECREF(scalar);
        if (TF_PREDICT_FALSE(error != nullptr)) return error;
        ++*buf;
      }
    }
    return nullptr;
  }

  static Status Convert(TFE_Context* ctx, PyObject* obj, ConverterState* state,
                        TFE_TensorHandle** h, const char** error) {
    // TODO(josh11b): Allocator & attributes
    AbstractTensorInterface* t;
    if (state->inferred_shape.empty()) { /* Scalar case */
      T value;
      auto scalar = ZeroDimArrayToScalar(obj, state);
      *error = ConverterTraits<T>::ConvertScalar(scalar, &value);
      Py_DECREF(scalar);
      if (*error != nullptr) return errors::InvalidArgument(*error);
      t = ConverterTraits<T>::CreateScalar(ctx, value);
      if (t == nullptr) {
        return errors::Internal("Cannot create tensor.");
      }
    } else {
      t = ConverterTraits<T>::CreateTensor(ctx, state->inferred_shape);
      if (t == nullptr) {
        return errors::Internal("Cannot create tensor.");
      }
      if (t->NumElements() > 0) {
        T* buf = static_cast<T*>(t->Data());
        *error = Helper(obj, 0, state, &buf);
        if (*error != nullptr) {
          t->Release();
          return errors::InvalidArgument(*error);
        }
      }
    }
    *h = tensorflow::wrap(tensorflow::unwrap(ctx)->CreateLocalHandle(t));
    t->Release();
    return Status::OK();
  }
};

// Int support

template <>
struct ConverterTraits<int64> {
  static AbstractTensorInterface* CreateScalar(TFE_Context* ctx, int64 value) {
    return tensorflow::unwrap(ctx)->CreateInt64Scalar(value);
  }

  static AbstractTensorInterface* CreateTensor(
      TFE_Context* ctx, absl::Span<const int64> dim_sizes) {
    return tensorflow::unwrap(ctx)->CreateTensor(DT_INT64, dim_sizes);
  }

  static const char* ConvertScalar(PyObject* v, int64* out) {
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
      return ConvertScalar(as_int.get(), out);
    }
    if (IsPyFloat(v)) return ErrorFoundFloat;
    return ErrorMixedTypes;
  }
};

typedef Converter<int64> Int64Converter;

template <>
struct ConverterTraits<uint64> {
  static AbstractTensorInterface* CreateScalar(TFE_Context* ctx, uint64 value) {
    return tensorflow::unwrap(ctx)->CreateUint64Scalar(value);
  }

  static AbstractTensorInterface* CreateTensor(
      TFE_Context* ctx, absl::Span<const int64> dim_sizes) {
    return tensorflow::unwrap(ctx)->CreateTensor(DT_UINT64, dim_sizes);
  }

  static const char* ConvertScalar(PyObject* v, uint64* out) {
#if PY_MAJOR_VERSION < 3
    if (TF_PREDICT_TRUE(PyInt_Check(v))) {
      *out = PyInt_AsUnsignedLongLongMask(v);
      return nullptr;
    }
#endif
    if (TF_PREDICT_TRUE(PyLong_Check(v) || IsPyDimension(v))) {
      *out = PyLong_AsUnsignedLongLong(v);
      return nullptr;
    }
    if (PyIsInstance(v, &PyIntegerArrType_Type)) {  // NumPy integers
#if PY_MAJOR_VERSION < 3
      Safe_PyObjectPtr as_int = make_safe(PyNumber_Int(v));
#else
      Safe_PyObjectPtr as_int = make_safe(PyNumber_Long(v));
#endif
      return ConvertScalar(as_int.get(), out);
    }
    if (IsPyFloat(v)) return ErrorFoundFloat;
    return ErrorMixedTypes;
  }
};

typedef Converter<uint64> UInt64Converter;

template <>
struct ConverterTraits<int32> {
  static AbstractTensorInterface* CreateScalar(TFE_Context* ctx, int32 value) {
    return tensorflow::unwrap(ctx)->CreateInt32Scalar(value);
  }

  static AbstractTensorInterface* CreateTensor(
      TFE_Context* ctx, absl::Span<const int64> dim_sizes) {
    return tensorflow::unwrap(ctx)->CreateTensor(DT_INT32, dim_sizes);
  }

  static const char* ConvertScalar(PyObject* v, int32* out) {
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
      return ConvertScalar(as_int.get(), out);
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
};

typedef Converter<int32> Int32Converter;

// Floating-point support

// Returns `true` if `out` overflows when converted from `as_double`.
template <class T>
static inline bool CheckForOverflow(double as_double, T* out) {
  return (sizeof(T) < sizeof(double) && std::isinf(*out) &&
          std::isfinite(as_double));
}

// There is no `std::isinf` that takes `Eigen::half` as argument but Eigen
// provides `Eigen::numext::isinf` instead.
template <>
inline bool CheckForOverflow<Eigen::half>(double as_double, Eigen::half* out) {
  return (Eigen::numext::isinf(*out) && std::isfinite(as_double));
}

template <class T>
static const char* ConvertOneFloat(PyObject* v, T* out) {
  if (PyErr_Occurred()) {
    return nullptr;
  }
  if (TF_PREDICT_TRUE(PyFloat_Check(v))) {
    const double as_double = PyFloat_AS_DOUBLE(v);
    *out = static_cast<T>(as_double);
    // Check for overflow
    if (TF_PREDICT_FALSE(CheckForOverflow<T>(as_double, out))) {
      return ErrorOutOfRangeDouble;
    }
    return nullptr;
  }
#if PY_MAJOR_VERSION < 3
  if (PyInt_Check(v)) {
    *out = static_cast<T>(PyInt_AS_LONG(v));
    return nullptr;
  }
#endif
  if (PyLong_Check(v)) {
    *out = static_cast<T>(PyLong_AsDouble(v));
    if (PyErr_Occurred()) return ErrorOutOfRangeDouble;
    return nullptr;
  }
  if (PyIsInstance(v, &PyFloatingArrType_Type)) {  // NumPy float types
    Safe_PyObjectPtr as_float = make_safe(PyNumber_Float(v));
    if (PyErr_Occurred()) {
      return nullptr;
    }
    return ConvertOneFloat<T>(as_float.get(), out);
  }
  if (PyIsInstance(v, &PyIntegerArrType_Type)) {  // NumPy integers
#if PY_MAJOR_VERSION < 3
    Safe_PyObjectPtr as_int = make_safe(PyNumber_Int(v));
#else
    Safe_PyObjectPtr as_int = make_safe(PyNumber_Long(v));
#endif
    if (PyErr_Occurred()) {
      return nullptr;
    }
    return ConvertOneFloat<T>(as_int.get(), out);
  }
  return ErrorMixedTypes;
}

template <>
struct ConverterTraits<float> {
  static AbstractTensorInterface* CreateScalar(TFE_Context* ctx, float value) {
    return tensorflow::unwrap(ctx)->CreateFloatScalar(value);
  }

  static AbstractTensorInterface* CreateTensor(
      TFE_Context* ctx, absl::Span<const int64> dim_sizes) {
    return tensorflow::unwrap(ctx)->CreateTensor(DT_FLOAT, dim_sizes);
  }

  static const char* ConvertScalar(PyObject* v, float* out) {
    return ConvertOneFloat<float>(v, out);
  }
};

template <>
struct ConverterTraits<double> {
  static AbstractTensorInterface* CreateScalar(TFE_Context* ctx, double value) {
    return tensorflow::unwrap(ctx)->CreateDoubleScalar(value);
  }

  static AbstractTensorInterface* CreateTensor(
      TFE_Context* ctx, absl::Span<const int64> dim_sizes) {
    return tensorflow::unwrap(ctx)->CreateTensor(DT_DOUBLE, dim_sizes);
  }

  static const char* ConvertScalar(PyObject* v, double* out) {
    return ConvertOneFloat<double>(v, out);
  }
};

typedef Converter<double> DoubleConverter;
typedef Converter<float> FloatConverter;

template <>
struct ConverterTraits<Eigen::half> {
  static AbstractTensorInterface* CreateScalar(TFE_Context* ctx,
                                               Eigen::half value) {
    return tensorflow::unwrap(ctx)->CreateHalfScalar(value);
  }

  static AbstractTensorInterface* CreateTensor(
      TFE_Context* ctx, absl::Span<const int64> dim_sizes) {
    return tensorflow::unwrap(ctx)->CreateTensor(DT_HALF, dim_sizes);
  }

  static const char* ConvertScalar(PyObject* v, Eigen::half* out) {
    return ConvertOneFloat<Eigen::half>(v, out);
  }
};

typedef Converter<Eigen::half> NumpyHalfConverter;

// String support

template <>
struct ConverterTraits<tstring> {
  static AbstractTensorInterface* CreateScalar(TFE_Context* ctx,
                                               tstring value) {
    return tensorflow::unwrap(ctx)->CreateStringScalar(value);
  }

  static AbstractTensorInterface* CreateTensor(
      TFE_Context* ctx, absl::Span<const int64> dim_sizes) {
    return tensorflow::unwrap(ctx)->CreateTensor(DT_STRING, dim_sizes);
  }

  static const char* ConvertScalar(PyObject* v, tstring* out) {
    if (PyBytes_Check(v)) {
      out->assign(PyBytes_AS_STRING(v), PyBytes_GET_SIZE(v));
      return nullptr;
    }
    if (PyUnicode_Check(v)) {
#if PY_MAJOR_VERSION >= 3
      Py_ssize_t size;
      const char* str = PyUnicode_AsUTF8AndSize(v, &size);
      if (str == nullptr) return ErrorConvertingUnicodeString;
      out->assign(str, size);
      return nullptr;
#else
      PyObject* py_str = PyUnicode_AsUTF8String(v);
      if (py_str == nullptr) return ErrorConvertingUnicodeString;
      out->assign(PyBytes_AS_STRING(py_str), PyBytes_GET_SIZE(py_str));
      Py_DECREF(py_str);
      return nullptr;
#endif
    }
    return ErrorMixedTypes;
  }
};

typedef Converter<tstring> StringConverter;

// Converts Python object `c` that should hold a Python string into a
// C++ string in *out.  Returns nullptr on success, or a message on error.
// Defined below, but forward declared here for use in PyRepr.
tstring PyRepr(PyObject* obj) {
  if (obj == nullptr) {
    return "<null>";
  }
  Safe_PyObjectPtr repr_obj = make_safe(PyObject_Repr(obj));
  if (repr_obj) {
    tstring repr_str;
    if (ConverterTraits<tstring>::ConvertScalar(repr_obj.get(), &repr_str) ==
        nullptr) {
      return repr_str;
    }
  }
  return "<error computing repr()>";
}

bool IsPyDimension(PyObject* obj) {
  const char* tp_name = obj->ob_type->tp_name;
  if (strcmp(tp_name, "Dimension") != 0) return false;
  bool ret = str_util::EndsWith(
      PyRepr(PyType(obj)),
      "tensorflow.python.framework.tensor_shape.Dimension'>");
  return ret;
}

// Complex support

template <>
struct ConverterTraits<complex128> {
  static AbstractTensorInterface* CreateScalar(TFE_Context* ctx,
                                               complex128 value) {
    return tensorflow::unwrap(ctx)->CreateComplex128Scalar(value);
  }

  static AbstractTensorInterface* CreateTensor(
      TFE_Context* ctx, absl::Span<const int64> dim_sizes) {
    return tensorflow::unwrap(ctx)->CreateTensor(DT_COMPLEX128, dim_sizes);
  }

  static const char* ConvertScalar(PyObject* v, complex128* out) {
    if (PyComplex_Check(v)) {
      *out = complex128(PyComplex_RealAsDouble(v), PyComplex_ImagAsDouble(v));
      return nullptr;
    } else if (PyIsInstance(v, &PyComplexFloatingArrType_Type)) {  // NumPy
      auto as_complex = PyComplex_AsCComplex(v);
      *out = complex128(as_complex.real, as_complex.imag);
      return nullptr;
    }
    double as_double;
    auto error = ConvertOneFloat<double>(v, &as_double);
    if (error != nullptr) return error;
    *out = complex128(as_double, 0.0);
    return nullptr;
  }
};

typedef Converter<complex128> Complex128Converter;

// Bool support

template <>
struct ConverterTraits<bool> {
  static AbstractTensorInterface* CreateScalar(TFE_Context* ctx, bool value) {
    return tensorflow::unwrap(ctx)->CreateBoolScalar(value);
  }

  static AbstractTensorInterface* CreateTensor(
      TFE_Context* ctx, absl::Span<const int64> dim_sizes) {
    return tensorflow::unwrap(ctx)->CreateTensor(DT_BOOL, dim_sizes);
  }

  static const char* ConvertScalar(PyObject* v, bool* out) {
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
};

typedef Converter<bool> BoolConverter;

// Convert a Python numpy.ndarray object to a TFE_TensorHandle.
// The two may share underlying storage so changes to one may reflect in the
// other.
TFE_TensorHandle* NumpyToTFE_TensorHandle(TFE_Context* ctx, PyObject* obj) {
  Safe_TF_TensorPtr tf_tensor = make_safe(static_cast<TF_Tensor*>(nullptr));
  Status status = tensorflow::NdarrayToTensor(ctx, obj, &tf_tensor);

  if (TF_PREDICT_FALSE(!status.ok())) {
    PyErr_SetString(PyExc_ValueError,
                    tensorflow::strings::StrCat(
                        "Failed to convert a NumPy array to a Tensor (",
                        status.error_message(), ").")
                        .c_str());
    return nullptr;
  }

  return tensorflow::wrap(
      tensorflow::unwrap(ctx)->CreateLocalHandle(tf_tensor->tensor));
}

}  // namespace

// TODO(b/147743551): This function handles enough conversions to justify
// promoting to something like PyObjectToTensorHandle.
// TODO(b/147828820): Handle Tensors properly.
TFE_TensorHandle* PySeqToTFE_TensorHandle(TFE_Context* ctx, PyObject* obj,
                                          DataType dtype) {
  // Shortcut: __array__ objects (such as Pandas data frames).
  // These objects are efficiently handled by Numpy. We transform them into
  // Numpy arrays and handle them in the Numpy case below. Note that Tensors
  // implement the __array__ function, and will be handled in this shortcut.
  Safe_PyObjectPtr array =
      make_safe(PyArray_FromArrayAttr(obj, nullptr, nullptr));
  if (array == nullptr) {
    return nullptr;
  }
  if (array.get() == Py_NotImplemented) {
    // The Py_NotImplemented returned from PyArray_FromArrayAttr is not
    // Py_INCREF'ed, so we don't want the Safe_PyObjectPtr to Py_DECREF it.
    array.release();
  } else {
    // PyArray_FromArrayAttr ensures that `array` is a PyArrayObject, so all
    // we have to do is replace `obj` with it and continue.
    obj = array.get();
  }

  // Shortcut: Numpy arrays.
  if (PyArray_Check(obj)) {
    int desired_np_dtype = -1;
    if (dtype != tensorflow::DT_INVALID) {
      if (!tensorflow::TF_DataType_to_PyArray_TYPE(
               static_cast<TF_DataType>(dtype), &desired_np_dtype)
               .ok()) {
        PyErr_SetString(
            PyExc_TypeError,
            tensorflow::strings::StrCat("Invalid dtype argument value ", dtype)
                .c_str());
        return nullptr;
      }
    }

    PyArrayObject* array = reinterpret_cast<PyArrayObject*>(obj);
    int array_dtype = PyArray_TYPE(array);

    Safe_PyObjectPtr safe_value(nullptr);
    // Use Numpy to convert between types if needed.
    if ((desired_np_dtype >= 0 && desired_np_dtype != array_dtype) ||
        !PyArray_ISCARRAY(array)) {
      int new_dtype = desired_np_dtype >= 0 ? desired_np_dtype : array_dtype;
      safe_value = tensorflow::make_safe(
          PyArray_FromAny(obj, PyArray_DescrFromType(new_dtype), 0, 0,
                          NPY_ARRAY_CARRAY_RO | NPY_ARRAY_FORCECAST, nullptr));
      if (PyErr_Occurred()) return nullptr;
      if (safe_value == nullptr) {
        PyErr_SetString(PyExc_ValueError, "Error while casting a numpy value");
      }
      obj = safe_value.get();
    }
    return NumpyToTFE_TensorHandle(ctx, obj);
  }

  ConverterState state;
  Status status = InferShapeAndType(obj, &state);
  if (!status.ok()) {
    PyErr_SetString(PyExc_ValueError, status.error_message().c_str());
    return nullptr;
  }
  DataType requested_dtype = DT_INVALID;
  if (dtype != DT_INVALID) {
    requested_dtype = dtype;
  }

  // NOTE(josh11b): If don't successfully convert to the requested type,
  // we just try instead to create a tensor of the inferred type and
  // let the caller convert it to the requested type using a cast
  // operation.
  const char* error = nullptr;
  TFE_TensorHandle* handle = nullptr;
  status = errors::Unimplemented("Missing Python -> Tensor conversion for ",
                                 DataTypeString(state.inferred_dtype));
  switch (requested_dtype) {
    case DT_FLOAT:
      status = FloatConverter::Convert(ctx, obj, &state, &handle, &error);
      break;

    case DT_DOUBLE:
      status = DoubleConverter::Convert(ctx, obj, &state, &handle, &error);
      break;

    case DT_HALF:
      status = NumpyHalfConverter::Convert(ctx, obj, &state, &handle, &error);
      break;

    case DT_INT64:
      status = Int64Converter::Convert(ctx, obj, &state, &handle, &error);
      break;

    case DT_INT32:
      status = Int32Converter::Convert(ctx, obj, &state, &handle, &error);
      break;

    case DT_UINT64:
      status = UInt64Converter::Convert(ctx, obj, &state, &handle, &error);
      break;

    case DT_COMPLEX128:
      status = Complex128Converter::Convert(ctx, obj, &state, &handle, &error);
      break;

    case DT_STRING:
      status = StringConverter::Convert(ctx, obj, &state, &handle, &error);
      break;

    case DT_BOOL:
      status = BoolConverter::Convert(ctx, obj, &state, &handle, &error);
      break;

    default:
      break;
  }
  if (status.ok()) return handle;

  switch (state.inferred_dtype) {
    case DT_FLOAT:
      // TODO(josh11b): Handle mixed floats and complex numbers?
      if (requested_dtype == DT_INVALID) {
        // TensorFlow uses float32s to represent floating point numbers
        // by default (for space and speed over using doubles).
        status = FloatConverter::Convert(ctx, obj, &state, &handle, &error);
      } else {
        // We are going to do a cast to the user's requested dtype
        // after this.  We use doubles for this intermediate result so
        // we don't lose precision that might be representable in the
        // final type.
        status = DoubleConverter::Convert(ctx, obj, &state, &handle, &error);
      }
      break;

    case DT_DOUBLE:
      status = DoubleConverter::Convert(ctx, obj, &state, &handle, &error);
      break;

    case DT_HALF:
      status = NumpyHalfConverter::Convert(ctx, obj, &state, &handle, &error);
      break;

    case DT_INT64:
      if (requested_dtype == DT_INVALID) {
        status = Int32Converter::Convert(ctx, obj, &state, &handle, &error);
        if (error == ErrorFoundInt64) {
          status = Int64Converter::Convert(ctx, obj, &state, &handle, &error);
        }
        if (error == ErrorFoundFloat) {
          status = FloatConverter::Convert(ctx, obj, &state, &handle, &error);
        }
        // TODO(josh11b): May also want to fall back to using doubles if
        // error == ErrorOutOfRange?
      } else {
        status = Int64Converter::Convert(ctx, obj, &state, &handle, &error);
        if (error == ErrorFoundFloat) {
          status = DoubleConverter::Convert(ctx, obj, &state, &handle, &error);
        }
      }
      break;

    case DT_STRING:
      status = StringConverter::Convert(ctx, obj, &state, &handle, &error);
      break;

    case DT_COMPLEX128:
      status = Complex128Converter::Convert(ctx, obj, &state, &handle, &error);
      break;

    case DT_BOOL:
      status = BoolConverter::Convert(ctx, obj, &state, &handle, &error);
      break;

    case DT_INVALID:  // Only occurs for empty tensors.
    {
      AbstractTensorInterface* t = tensorflow::unwrap(ctx)->CreateTensor(
          requested_dtype == DT_INVALID ? DT_FLOAT : requested_dtype,
          state.inferred_shape);
      auto* result =
          tensorflow::wrap(tensorflow::unwrap(ctx)->CreateLocalHandle(t));
      t->Release();
      return result;
    }

    default:
      break;
  }

  if (!status.ok()) {
    PyErr_SetString(PyExc_ValueError, status.error_message().c_str());
    return nullptr;
  }

  return handle;
}

}  // namespace tensorflow
