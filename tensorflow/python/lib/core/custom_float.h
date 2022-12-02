/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_PYTHON_LIB_CORE_CUSTOM_FLOAT_H_
#define TENSORFLOW_PYTHON_LIB_CORE_CUSTOM_FLOAT_H_

// Must be included first
// clang-format off
#include "tensorflow/tsl/python/lib/core/numpy.h" // NOLINT
// clang-format on

// Support utilities for adding custom floating-point dtypes to TensorFlow,
// such as bfloat16, and float8_*.

#include <array>   // NOLINT
#include <cmath>   // NOLINT
#include <limits>  // NOLINT
#include <locale>  // NOLINT
#include <memory>  // NOLINT

// Place `<locale>` before <Python.h> to avoid a build failure in macOS.
#include <Python.h>

#include "absl/strings/str_cat.h"
#include "third_party/eigen3/Eigen/Core"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"

#undef copysign  // TODO(ddunleavy): temporary fix for Windows bazel build
                 // Possible this has to do with numpy.h being included before
                 // system headers and in bfloat16.{cc,h}?

namespace tensorflow {
namespace custom_float_internal {

struct PyDecrefDeleter {
  void operator()(PyObject* p) const { Py_DECREF(p); }
};

// Safe container for an owned PyObject. On destruction, the reference count of
// the contained object will be decremented.
using Safe_PyObjectPtr = std::unique_ptr<PyObject, PyDecrefDeleter>;
inline Safe_PyObjectPtr make_safe(PyObject* object) {
  return Safe_PyObjectPtr(object);
}

inline bool PyLong_CheckNoOverflow(PyObject* object) {
  if (!PyLong_Check(object)) {
    return false;
  }
  int overflow = 0;
  PyLong_AsLongAndOverflow(object, &overflow);
  return (overflow == 0);
}

template <typename T, typename Enable = void>
struct TypeDescriptor {
  // typedef ... T;  // Representation type in memory for NumPy values of type
  // static int Dtype() { return NPY_...; }  // Numpy type number for T.
};

template <typename T>
struct CustomFloatTypeDescriptor {
  static int Dtype() { return npy_type; }

  // Registered numpy type ID. Global variable populated by the registration
  // code. Protected by the GIL.
  static int npy_type;

  static PyTypeObject type;
  // Pointer to the python type object we are using. This is either a pointer
  // to type, if we choose to register it, or to the python type
  // registered by another system into NumPy.
  static PyTypeObject* type_ptr;

  static PyNumberMethods number_methods;

  static PyArray_ArrFuncs arr_funcs;

  static PyArray_Descr npy_descr;
};
template <typename T>
int CustomFloatTypeDescriptor<T>::npy_type = NPY_NOTYPE;
template <typename T>
PyTypeObject* CustomFloatTypeDescriptor<T>::type_ptr = nullptr;

// Representation of a Python custom float object.
template <typename T>
struct PyCustomFloat {
  PyObject_HEAD;  // Python object header
  T value;
};

// Returns true if 'object' is a PyCustomFloat.
template <typename T>
bool PyCustomFloat_Check(PyObject* object) {
  return PyObject_IsInstance(
      object, reinterpret_cast<PyObject*>(&TypeDescriptor<T>::type));
}

// Extracts the value of a PyCustomFloat object.
template <typename T>
T PyCustomFloat_CustomFloat(PyObject* object) {
  return reinterpret_cast<PyCustomFloat<T>*>(object)->value;
}

// Constructs a PyCustomFloat object from PyCustomFloat<T>::T.
template <typename T>
Safe_PyObjectPtr PyCustomFloat_FromT(T x) {
  Safe_PyObjectPtr ref =
      make_safe(TypeDescriptor<T>::type.tp_alloc(&TypeDescriptor<T>::type, 0));
  PyCustomFloat<T>* p = reinterpret_cast<PyCustomFloat<T>*>(ref.get());
  if (p) {
    p->value = x;
  }
  return ref;
}

// Converts a Python object to a reduced float value. Returns true on success,
// returns false and reports a Python error on failure.
template <typename T>
bool CastToCustomFloat(PyObject* arg, T* output) {
  if (PyCustomFloat_Check<T>(arg)) {
    *output = PyCustomFloat_CustomFloat<T>(arg);
    return true;
  }
  if (PyFloat_Check(arg)) {
    double d = PyFloat_AsDouble(arg);
    if (PyErr_Occurred()) {
      return false;
    }
    // TODO(phawkins): check for overflow
    *output = T(d);
    return true;
  }
  if (PyLong_CheckNoOverflow(arg)) {
    long l = PyLong_AsLong(arg);  // NOLINT
    if (PyErr_Occurred()) {
      return false;
    }
    // TODO(phawkins): check for overflow
    *output = T(static_cast<float>(l));
    return true;
  }
  if (PyArray_IsScalar(arg, Half)) {
    Eigen::half f;
    PyArray_ScalarAsCtype(arg, &f);
    *output = T(f);
    return true;
  }
  if (PyArray_IsScalar(arg, Float)) {
    float f;
    PyArray_ScalarAsCtype(arg, &f);
    *output = T(f);
    return true;
  }
  if (PyArray_IsScalar(arg, Double)) {
    double f;
    PyArray_ScalarAsCtype(arg, &f);
    *output = T(f);
    return true;
  }
  if (PyArray_IsScalar(arg, LongDouble)) {
    long double f;
    PyArray_ScalarAsCtype(arg, &f);
    *output = T(f);
    return true;
  }
  if (PyArray_IsZeroDim(arg)) {
    Safe_PyObjectPtr ref;
    PyArrayObject* arr = reinterpret_cast<PyArrayObject*>(arg);
    if (PyArray_TYPE(arr) != TypeDescriptor<T>::Dtype()) {
      ref = make_safe(PyArray_Cast(arr, TypeDescriptor<T>::Dtype()));
      if (PyErr_Occurred()) {
        return false;
      }
      arg = ref.get();
      arr = reinterpret_cast<PyArrayObject*>(arg);
    }
    *output = *reinterpret_cast<T*>(PyArray_DATA(arr));
    return true;
  }
  return false;
}

template <typename T>
bool SafeCastToCustomFloat(PyObject* arg, T* output) {
  if (PyCustomFloat_Check<T>(arg)) {
    *output = PyCustomFloat_CustomFloat<T>(arg);
    return true;
  }
  return false;
}

// Converts a PyReduceFloat into a PyFloat.
template <typename T>
PyObject* PyCustomFloat_Float(PyObject* self) {
  T x = PyCustomFloat_CustomFloat<T>(self);
  return PyFloat_FromDouble(static_cast<double>(static_cast<float>(x)));
}

// Converts a PyReduceFloat into a PyInt.
template <typename T>
PyObject* PyCustomFloat_Int(PyObject* self) {
  T x = PyCustomFloat_CustomFloat<T>(self);
  long y = static_cast<long>(static_cast<float>(x));  // NOLINT
  return PyLong_FromLong(y);
}

// Negates a PyCustomFloat.
template <typename T>
PyObject* PyCustomFloat_Negative(PyObject* self) {
  T x = PyCustomFloat_CustomFloat<T>(self);
  return PyCustomFloat_FromT<T>(-x).release();
}

template <typename T>
PyObject* PyCustomFloat_Add(PyObject* a, PyObject* b) {
  T x, y;
  if (SafeCastToCustomFloat<T>(a, &x) && SafeCastToCustomFloat<T>(b, &y)) {
    return PyCustomFloat_FromT<T>(x + y).release();
  }
  return PyArray_Type.tp_as_number->nb_add(a, b);
}

template <typename T>
PyObject* PyCustomFloat_Subtract(PyObject* a, PyObject* b) {
  T x, y;
  if (SafeCastToCustomFloat<T>(a, &x) && SafeCastToCustomFloat<T>(b, &y)) {
    return PyCustomFloat_FromT<T>(x - y).release();
  }
  return PyArray_Type.tp_as_number->nb_subtract(a, b);
}

template <typename T>
PyObject* PyCustomFloat_Multiply(PyObject* a, PyObject* b) {
  T x, y;
  if (SafeCastToCustomFloat<T>(a, &x) && SafeCastToCustomFloat<T>(b, &y)) {
    return PyCustomFloat_FromT<T>(x * y).release();
  }
  return PyArray_Type.tp_as_number->nb_multiply(a, b);
}

template <typename T>
PyObject* PyCustomFloat_TrueDivide(PyObject* a, PyObject* b) {
  T x, y;
  if (SafeCastToCustomFloat<T>(a, &x) && SafeCastToCustomFloat<T>(b, &y)) {
    return PyCustomFloat_FromT<T>(x / y).release();
  }
  return PyArray_Type.tp_as_number->nb_true_divide(a, b);
}

// Python number methods for PyCustomFloat objects.
template <typename T>
PyNumberMethods CustomFloatTypeDescriptor<T>::number_methods = {
    PyCustomFloat_Add<T>,       // nb_add
    PyCustomFloat_Subtract<T>,  // nb_subtract
    PyCustomFloat_Multiply<T>,  // nb_multiply
    nullptr,                    // nb_remainder
    nullptr,                    // nb_divmod
    nullptr,                    // nb_power
    PyCustomFloat_Negative<T>,  // nb_negative
    nullptr,                    // nb_positive
    nullptr,                    // nb_absolute
    nullptr,                    // nb_nonzero
    nullptr,                    // nb_invert
    nullptr,                    // nb_lshift
    nullptr,                    // nb_rshift
    nullptr,                    // nb_and
    nullptr,                    // nb_xor
    nullptr,                    // nb_or
    PyCustomFloat_Int<T>,       // nb_int
    nullptr,                    // reserved
    PyCustomFloat_Float<T>,     // nb_float

    nullptr,  // nb_inplace_add
    nullptr,  // nb_inplace_subtract
    nullptr,  // nb_inplace_multiply
    nullptr,  // nb_inplace_remainder
    nullptr,  // nb_inplace_power
    nullptr,  // nb_inplace_lshift
    nullptr,  // nb_inplace_rshift
    nullptr,  // nb_inplace_and
    nullptr,  // nb_inplace_xor
    nullptr,  // nb_inplace_or

    nullptr,                      // nb_floor_divide
    PyCustomFloat_TrueDivide<T>,  // nb_true_divide
    nullptr,                      // nb_inplace_floor_divide
    nullptr,                      // nb_inplace_true_divide
    nullptr,                      // nb_index
};

// Constructs a new PyCustomFloat.
template <typename T>
PyObject* PyCustomFloat_New(PyTypeObject* type, PyObject* args,
                            PyObject* kwds) {
  if (kwds && PyDict_Size(kwds)) {
    PyErr_SetString(PyExc_TypeError, "constructor takes no keyword arguments");
    return nullptr;
  }
  Py_ssize_t size = PyTuple_Size(args);
  if (size != 1) {
    PyErr_Format(PyExc_TypeError,
                 "expected number as argument to %s constructor",
                 TypeDescriptor<T>::kTypeName);
    return nullptr;
  }
  PyObject* arg = PyTuple_GetItem(args, 0);

  T value;
  if (PyCustomFloat_Check<T>(arg)) {
    Py_INCREF(arg);
    return arg;
  } else if (CastToCustomFloat<T>(arg, &value)) {
    return PyCustomFloat_FromT<T>(value).release();
  } else if (PyArray_Check(arg)) {
    PyArrayObject* arr = reinterpret_cast<PyArrayObject*>(arg);
    if (PyArray_TYPE(arr) != TypeDescriptor<T>::Dtype()) {
      return PyArray_Cast(arr, TypeDescriptor<T>::Dtype());
    } else {
      Py_INCREF(arg);
      return arg;
    }
  } else if (PyUnicode_Check(arg) || PyBytes_Check(arg)) {
    // Parse float from string, then cast to T.
    PyObject* f = PyFloat_FromString(arg);
    if (CastToCustomFloat<T>(f, &value)) {
      return PyCustomFloat_FromT<T>(value).release();
    }
  }
  PyErr_Format(PyExc_TypeError, "expected number, got %s",
               Py_TYPE(arg)->tp_name);
  return nullptr;
}

// Comparisons on PyCustomFloats.
template <typename T>
PyObject* PyCustomFloat_RichCompare(PyObject* a, PyObject* b, int op) {
  T x, y;
  if (!SafeCastToCustomFloat<T>(a, &x) || !SafeCastToCustomFloat<T>(b, &y)) {
    return PyGenericArrType_Type.tp_richcompare(a, b, op);
  }
  bool result;
  switch (op) {
    case Py_LT:
      result = x < y;
      break;
    case Py_LE:
      result = x <= y;
      break;
    case Py_EQ:
      result = x == y;
      break;
    case Py_NE:
      result = x != y;
      break;
    case Py_GT:
      result = x > y;
      break;
    case Py_GE:
      result = x >= y;
      break;
    default:
      LOG(ERROR) << "Invalid op type " << op;
      result = false;
  }
  return PyBool_FromLong(result);
}

// Implementation of repr() for PyCustomFloat.
template <typename T>
PyObject* PyCustomFloat_Repr(PyObject* self) {
  T x = reinterpret_cast<PyCustomFloat<T>*>(self)->value;
  std::string v = absl::StrCat(static_cast<float>(x));
  return PyUnicode_FromString(v.c_str());
}

// Implementation of str() for PyCustomFloat.
template <typename T>
PyObject* PyCustomFloat_Str(PyObject* self) {
  T x = reinterpret_cast<PyCustomFloat<T>*>(self)->value;
  std::string v = absl::StrCat(static_cast<float>(x));
  return PyUnicode_FromString(v.c_str());
}

// _Py_HashDouble changed its prototype for Python 3.10 so we use an overload to
// handle the two possibilities.
// NOLINTNEXTLINE(clang-diagnostic-unused-function)
inline Py_hash_t HashImpl(Py_hash_t (*hash_double)(PyObject*, double),
                          PyObject* self, double value) {
  return hash_double(self, value);
}

// NOLINTNEXTLINE(clang-diagnostic-unused-function)
inline Py_hash_t HashImpl(Py_hash_t (*hash_double)(double), PyObject* self,
                          double value) {
  return hash_double(value);
}

// Hash function for PyCustomFloat.
template <typename T>
Py_hash_t PyCustomFloat_Hash(PyObject* self) {
  T x = reinterpret_cast<PyCustomFloat<T>*>(self)->value;
  return HashImpl(&_Py_HashDouble, self, static_cast<double>(x));
}

// Python type for PyCustomFloat objects.
template <typename T>
PyTypeObject CustomFloatTypeDescriptor<T>::type = {
    PyVarObject_HEAD_INIT(nullptr, 0) TypeDescriptor<T>::kTypeName,  // tp_name
    sizeof(PyCustomFloat<T>),  // tp_basicsize
    0,                         // tp_itemsize
    nullptr,                   // tp_dealloc
#if PY_VERSION_HEX < 0x03080000
    nullptr,  // tp_print
#else
    0,  // tp_vectorcall_offset
#endif
    nullptr,                                        // tp_getattr
    nullptr,                                        // tp_setattr
    nullptr,                                        // tp_compare / tp_reserved
    PyCustomFloat_Repr<T>,                          // tp_repr
    &CustomFloatTypeDescriptor<T>::number_methods,  // tp_as_number
    nullptr,                                        // tp_as_sequence
    nullptr,                                        // tp_as_mapping
    PyCustomFloat_Hash<T>,                          // tp_hash
    nullptr,                                        // tp_call
    PyCustomFloat_Str<T>,                           // tp_str
    nullptr,                                        // tp_getattro
    nullptr,                                        // tp_setattro
    nullptr,                                        // tp_as_buffer
                                                    // tp_flags
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    TypeDescriptor<T>::kTpDoc,     // tp_doc
    nullptr,                       // tp_traverse
    nullptr,                       // tp_clear
    PyCustomFloat_RichCompare<T>,  // tp_richcompare
    0,                             // tp_weaklistoffset
    nullptr,                       // tp_iter
    nullptr,                       // tp_iternext
    nullptr,                       // tp_methods
    nullptr,                       // tp_members
    nullptr,                       // tp_getset
    nullptr,                       // tp_base
    nullptr,                       // tp_dict
    nullptr,                       // tp_descr_get
    nullptr,                       // tp_descr_set
    0,                             // tp_dictoffset
    nullptr,                       // tp_init
    nullptr,                       // tp_alloc
    PyCustomFloat_New<T>,          // tp_new
    nullptr,                       // tp_free
    nullptr,                       // tp_is_gc
    nullptr,                       // tp_bases
    nullptr,                       // tp_mro
    nullptr,                       // tp_cache
    nullptr,                       // tp_subclasses
    nullptr,                       // tp_weaklist
    nullptr,                       // tp_del
    0,                             // tp_version_tag
};

// Numpy support
template <typename T>
PyArray_ArrFuncs CustomFloatTypeDescriptor<T>::arr_funcs;

template <typename T>
PyArray_Descr CustomFloatTypeDescriptor<T>::npy_descr = {
    PyObject_HEAD_INIT(nullptr)  //
                                 /*typeobj=*/
    (&TypeDescriptor<T>::type),
    /*kind=*/TypeDescriptor<T>::kNpyDescrKind,
    /*type=*/TypeDescriptor<T>::kNpyDescrType,
    /*byteorder=*/TypeDescriptor<T>::kNpyDescrByteorder,
    /*flags=*/NPY_NEEDS_PYAPI | NPY_USE_SETITEM,
    /*type_num=*/0,
    /*elsize=*/sizeof(T),
    /*alignment=*/alignof(T),
    /*subarray=*/nullptr,
    /*fields=*/nullptr,
    /*names=*/nullptr,
    /*f=*/&CustomFloatTypeDescriptor<T>::arr_funcs,
    /*metadata=*/nullptr,
    /*c_metadata=*/nullptr,
    /*hash=*/-1,  // -1 means "not computed yet".
};

// Implementations of NumPy array methods.

template <typename T>
PyObject* NPyCustomFloat_GetItem(void* data, void* arr) {
  T x;
  memcpy(&x, data, sizeof(T));
  return PyFloat_FromDouble(static_cast<float>(x));
}

template <typename T>
int NPyCustomFloat_SetItem(PyObject* item, void* data, void* arr) {
  T x;
  if (!CastToCustomFloat<T>(item, &x)) {
    PyErr_Format(PyExc_TypeError, "expected number, got %s",
                 Py_TYPE(item)->tp_name);
    return -1;
  }
  memcpy(data, &x, sizeof(T));
  return 0;
}

inline void ByteSwap16(void* value) {
  char* p = reinterpret_cast<char*>(value);
  std::swap(p[0], p[1]);
}

template <typename T>
int NPyCustomFloat_Compare(const void* a, const void* b, void* arr) {
  T x;
  memcpy(&x, a, sizeof(T));

  T y;
  memcpy(&y, b, sizeof(T));
  float fy(y);
  float fx(x);

  if (fx < fy) {
    return -1;
  }
  if (fy < fx) {
    return 1;
  }
  // NaNs sort to the end.
  if (!Eigen::numext::isnan(fx) && Eigen::numext::isnan(fy)) {
    return -1;
  }
  if (Eigen::numext::isnan(fx) && !Eigen::numext::isnan(fy)) {
    return 1;
  }
  return 0;
}

template <typename T>
void NPyCustomFloat_CopySwapN(void* dstv, npy_intp dstride, void* srcv,
                              npy_intp sstride, npy_intp n, int swap,
                              void* arr) {
  static_assert(sizeof(T) == sizeof(int16_t) || sizeof(T) == sizeof(int8_t),
                "Not supported");
  char* dst = reinterpret_cast<char*>(dstv);
  char* src = reinterpret_cast<char*>(srcv);
  if (!src) {
    return;
  }
  if (swap && sizeof(T) == sizeof(int16_t)) {
    for (npy_intp i = 0; i < n; i++) {
      char* r = dst + dstride * i;
      memcpy(r, src + sstride * i, sizeof(T));
      ByteSwap16(r);
    }
  } else if (dstride == sizeof(T) && sstride == sizeof(T)) {
    memcpy(dst, src, n * sizeof(T));
  } else {
    for (npy_intp i = 0; i < n; i++) {
      memcpy(dst + dstride * i, src + sstride * i, sizeof(T));
    }
  }
}

template <typename T>
void NPyCustomFloat_CopySwap(void* dst, void* src, int swap, void* arr) {
  if (!src) {
    return;
  }
  memcpy(dst, src, sizeof(T));
  static_assert(sizeof(T) == sizeof(int16_t) || sizeof(T) == sizeof(int8_t),
                "Not supported");
  if (swap && sizeof(T) == sizeof(int16_t)) {
    ByteSwap16(dst);
  }
}

template <typename T>
npy_bool NPyCustomFloat_NonZero(void* data, void* arr) {
  T x;
  memcpy(&x, data, sizeof(x));
  return x != static_cast<T>(0);
}

template <typename T>
int NPyCustomFloat_Fill(void* buffer_raw, npy_intp length, void* ignored) {
  T* const buffer = reinterpret_cast<T*>(buffer_raw);
  const float start(buffer[0]);
  const float delta = static_cast<float>(buffer[1]) - start;
  for (npy_intp i = 2; i < length; ++i) {
    buffer[i] = static_cast<T>(start + i * delta);
  }
  return 0;
}

template <typename T>
void NPyCustomFloat_DotFunc(void* ip1, npy_intp is1, void* ip2, npy_intp is2,
                            void* op, npy_intp n, void* arr) {
  char* c1 = reinterpret_cast<char*>(ip1);
  char* c2 = reinterpret_cast<char*>(ip2);
  float acc = 0.0f;
  for (npy_intp i = 0; i < n; ++i) {
    T* const b1 = reinterpret_cast<T*>(c1);
    T* const b2 = reinterpret_cast<T*>(c2);
    acc += static_cast<float>(*b1) * static_cast<float>(*b2);
    c1 += is1;
    c2 += is2;
  }
  T* out = reinterpret_cast<T*>(op);
  *out = static_cast<T>(acc);
}

template <typename T>
int NPyCustomFloat_CompareFunc(const void* v1, const void* v2, void* arr) {
  T b1 = *reinterpret_cast<const T*>(v1);
  T b2 = *reinterpret_cast<const T*>(v2);
  if (b1 < b2) {
    return -1;
  }
  if (b1 > b2) {
    return 1;
  }
  return 0;
}

template <typename T>
int NPyCustomFloat_ArgMaxFunc(void* data, npy_intp n, npy_intp* max_ind,
                              void* arr) {
  const T* bdata = reinterpret_cast<const T*>(data);
  // Start with a max_val of NaN, this results in the first iteration preferring
  // bdata[0].
  float max_val = std::numeric_limits<float>::quiet_NaN();
  for (npy_intp i = 0; i < n; ++i) {
    // This condition is chosen so that NaNs are always considered "max".
    if (!(static_cast<float>(bdata[i]) <= max_val)) {
      max_val = static_cast<float>(bdata[i]);
      *max_ind = i;
      // NumPy stops at the first NaN.
      if (Eigen::numext::isnan(max_val)) {
        break;
      }
    }
  }
  return 0;
}

template <typename T>
int NPyCustomFloat_ArgMinFunc(void* data, npy_intp n, npy_intp* min_ind,
                              void* arr) {
  const T* bdata = reinterpret_cast<const T*>(data);
  float min_val = std::numeric_limits<float>::quiet_NaN();
  // Start with a min_val of NaN, this results in the first iteration preferring
  // bdata[0].
  for (npy_intp i = 0; i < n; ++i) {
    // This condition is chosen so that NaNs are always considered "min".
    if (!(static_cast<float>(bdata[i]) >= min_val)) {
      min_val = static_cast<float>(bdata[i]);
      *min_ind = i;
      // NumPy stops at the first NaN.
      if (Eigen::numext::isnan(min_val)) {
        break;
      }
    }
  }
  return 0;
}

template <>
struct TypeDescriptor<unsigned char> {
  typedef unsigned char T;
  static int Dtype() { return NPY_UBYTE; }
};

template <>
struct TypeDescriptor<unsigned short> {  // NOLINT
  typedef unsigned short T;              // NOLINT
  static int Dtype() { return NPY_USHORT; }
};

// We register "int", "long", and "long long" types for portability across
// Linux, where "int" and "long" are the same type, and Windows, where "long"
// and "longlong" are the same type.
template <>
struct TypeDescriptor<unsigned int> {
  typedef unsigned int T;
  static int Dtype() { return NPY_UINT; }
};

template <>
struct TypeDescriptor<unsigned long> {  // NOLINT
  typedef unsigned long T;              // NOLINT
  static int Dtype() { return NPY_ULONG; }
};

template <>
struct TypeDescriptor<unsigned long long> {  // NOLINT
  typedef unsigned long long T;              // NOLINT
  static int Dtype() { return NPY_ULONGLONG; }
};

template <>
struct TypeDescriptor<signed char> {
  typedef signed char T;
  static int Dtype() { return NPY_BYTE; }
};

template <>
struct TypeDescriptor<short> {  // NOLINT
  typedef short T;              // NOLINT
  static int Dtype() { return NPY_SHORT; }
};

template <>
struct TypeDescriptor<int> {
  typedef int T;
  static int Dtype() { return NPY_INT; }
};

template <>
struct TypeDescriptor<long> {  // NOLINT
  typedef long T;              // NOLINT
  static int Dtype() { return NPY_LONG; }
};

template <>
struct TypeDescriptor<long long> {  // NOLINT
  typedef long long T;              // NOLINT
  static int Dtype() { return NPY_LONGLONG; }
};

template <>
struct TypeDescriptor<bool> {
  typedef unsigned char T;
  static int Dtype() { return NPY_BOOL; }
};

template <>
struct TypeDescriptor<Eigen::half> {
  typedef Eigen::half T;
  static int Dtype() { return NPY_HALF; }
};

template <>
struct TypeDescriptor<float> {
  typedef float T;
  static int Dtype() { return NPY_FLOAT; }
};

template <>
struct TypeDescriptor<double> {
  typedef double T;
  static int Dtype() { return NPY_DOUBLE; }
};

template <>
struct TypeDescriptor<long double> {
  typedef long double T;
  static int Dtype() { return NPY_LONGDOUBLE; }
};

template <>
struct TypeDescriptor<std::complex<float>> {
  typedef std::complex<float> T;
  static int Dtype() { return NPY_CFLOAT; }
};

template <>
struct TypeDescriptor<std::complex<double>> {
  typedef std::complex<double> T;
  static int Dtype() { return NPY_CDOUBLE; }
};

template <>
struct TypeDescriptor<std::complex<long double>> {
  typedef std::complex<long double> T;
  static int Dtype() { return NPY_CLONGDOUBLE; }
};

template <typename T>
float CastToFloat(T value) {
  return static_cast<float>(value);
}

template <typename T>
float CastToFloat(std::complex<T> value) {
  return CastToFloat(value.real());
}

// Performs a NumPy array cast from type 'From' to 'To'.
template <typename From, typename To>
void NPyCast(void* from_void, void* to_void, npy_intp n, void* fromarr,
             void* toarr) {
  const auto* from =
      reinterpret_cast<typename TypeDescriptor<From>::T*>(from_void);
  auto* to = reinterpret_cast<typename TypeDescriptor<To>::T*>(to_void);
  for (npy_intp i = 0; i < n; ++i) {
    to[i] = static_cast<typename TypeDescriptor<To>::T>(
        static_cast<To>(CastToFloat(from[i])));
  }
}

// Registers a cast between T (a reduced float) and type 'OtherT'. 'numpy_type'
// is the NumPy type corresponding to 'OtherT'.
template <typename T, typename OtherT>
bool RegisterCustomFloatCast(int numpy_type = TypeDescriptor<OtherT>::Dtype()) {
  PyArray_Descr* descr = PyArray_DescrFromType(numpy_type);
  if (PyArray_RegisterCastFunc(descr, TypeDescriptor<T>::Dtype(),
                               NPyCast<OtherT, T>) < 0) {
    return false;
  }
  if (PyArray_RegisterCastFunc(&CustomFloatTypeDescriptor<T>::npy_descr,
                               numpy_type, NPyCast<T, OtherT>) < 0) {
    return false;
  }
  return true;
}

template <typename T>
bool RegisterCasts() {
  if (!RegisterCustomFloatCast<T, Eigen::half>(NPY_HALF)) {
    return false;
  }

  if (!RegisterCustomFloatCast<T, float>(NPY_FLOAT)) {
    return false;
  }
  if (!RegisterCustomFloatCast<T, double>(NPY_DOUBLE)) {
    return false;
  }
  if (!RegisterCustomFloatCast<T, long double>(NPY_LONGDOUBLE)) {
    return false;
  }
  if (!RegisterCustomFloatCast<T, bool>(NPY_BOOL)) {
    return false;
  }
  if (!RegisterCustomFloatCast<T, unsigned char>(NPY_UBYTE)) {
    return false;
  }
  if (!RegisterCustomFloatCast<T, unsigned short>(NPY_USHORT)) {  // NOLINT
    return false;
  }
  if (!RegisterCustomFloatCast<T, unsigned int>(NPY_UINT)) {
    return false;
  }
  if (!RegisterCustomFloatCast<T, unsigned long>(NPY_ULONG)) {  // NOLINT
    return false;
  }
  if (!RegisterCustomFloatCast<T, unsigned long long>(  // NOLINT
          NPY_ULONGLONG)) {
    return false;
  }
  if (!RegisterCustomFloatCast<T, signed char>(NPY_BYTE)) {
    return false;
  }
  if (!RegisterCustomFloatCast<T, short>(NPY_SHORT)) {  // NOLINT
    return false;
  }
  if (!RegisterCustomFloatCast<T, int>(NPY_INT)) {
    return false;
  }
  if (!RegisterCustomFloatCast<T, long>(NPY_LONG)) {  // NOLINT
    return false;
  }
  if (!RegisterCustomFloatCast<T, long long>(NPY_LONGLONG)) {  // NOLINT
    return false;
  }
  // Following the numpy convention. imag part is dropped when converting to
  // float.
  if (!RegisterCustomFloatCast<T, std::complex<float>>(NPY_CFLOAT)) {
    return false;
  }
  if (!RegisterCustomFloatCast<T, std::complex<double>>(NPY_CDOUBLE)) {
    return false;
  }
  if (!RegisterCustomFloatCast<T, std::complex<long double>>(NPY_CLONGDOUBLE)) {
    return false;
  }

  // Safe casts from T to other types
  if (PyArray_RegisterCanCast(&TypeDescriptor<T>::npy_descr, NPY_FLOAT,
                              NPY_NOSCALAR) < 0) {
    return false;
  }
  if (PyArray_RegisterCanCast(&TypeDescriptor<T>::npy_descr, NPY_DOUBLE,
                              NPY_NOSCALAR) < 0) {
    return false;
  }
  if (PyArray_RegisterCanCast(&TypeDescriptor<T>::npy_descr, NPY_LONGDOUBLE,
                              NPY_NOSCALAR) < 0) {
    return false;
  }
  if (PyArray_RegisterCanCast(&TypeDescriptor<T>::npy_descr, NPY_CFLOAT,
                              NPY_NOSCALAR) < 0) {
    return false;
  }
  if (PyArray_RegisterCanCast(&TypeDescriptor<T>::npy_descr, NPY_CDOUBLE,
                              NPY_NOSCALAR) < 0) {
    return false;
  }
  if (PyArray_RegisterCanCast(&TypeDescriptor<T>::npy_descr, NPY_CLONGDOUBLE,
                              NPY_NOSCALAR) < 0) {
    return false;
  }

  // Safe casts to T from other types
  if (PyArray_RegisterCanCast(PyArray_DescrFromType(NPY_BOOL),
                              TypeDescriptor<T>::Dtype(), NPY_NOSCALAR) < 0) {
    return false;
  }
  if (PyArray_RegisterCanCast(PyArray_DescrFromType(NPY_UBYTE),
                              TypeDescriptor<T>::Dtype(), NPY_NOSCALAR) < 0) {
    return false;
  }
  if (PyArray_RegisterCanCast(PyArray_DescrFromType(NPY_BYTE),
                              TypeDescriptor<T>::Dtype(), NPY_NOSCALAR) < 0) {
    return false;
  }

  return true;
}

template <typename InType, typename OutType, typename Functor>
struct UnaryUFunc {
  static std::vector<int> Types() {
    return {TypeDescriptor<InType>::Dtype(), TypeDescriptor<OutType>::Dtype()};
  }
  static void Call(char** args, const npy_intp* dimensions,
                   const npy_intp* steps, void* data) {
    const char* i0 = args[0];
    char* o = args[1];
    for (npy_intp k = 0; k < *dimensions; k++) {
      auto x = *reinterpret_cast<const typename TypeDescriptor<InType>::T*>(i0);
      *reinterpret_cast<typename TypeDescriptor<OutType>::T*>(o) = Functor()(x);
      i0 += steps[0];
      o += steps[1];
    }
  }
};

template <typename InType, typename OutType, typename OutType2,
          typename Functor>
struct UnaryUFunc2 {
  static std::vector<int> Types() {
    return {TypeDescriptor<InType>::Dtype(), TypeDescriptor<OutType>::Dtype(),
            TypeDescriptor<OutType2>::Dtype()};
  }
  static void Call(char** args, const npy_intp* dimensions,
                   const npy_intp* steps, void* data) {
    const char* i0 = args[0];
    char* o0 = args[1];
    char* o1 = args[2];
    for (npy_intp k = 0; k < *dimensions; k++) {
      auto x = *reinterpret_cast<const typename TypeDescriptor<InType>::T*>(i0);
      std::tie(*reinterpret_cast<typename TypeDescriptor<OutType>::T*>(o0),
               *reinterpret_cast<typename TypeDescriptor<OutType2>::T*>(o1)) =
          Functor()(x);
      i0 += steps[0];
      o0 += steps[1];
      o1 += steps[2];
    }
  }
};

template <typename InType, typename OutType, typename Functor>
struct BinaryUFunc {
  static std::vector<int> Types() {
    return {TypeDescriptor<InType>::Dtype(), TypeDescriptor<InType>::Dtype(),
            TypeDescriptor<OutType>::Dtype()};
  }
  static void Call(char** args, const npy_intp* dimensions,
                   const npy_intp* steps, void* data) {
    const char* i0 = args[0];
    const char* i1 = args[1];
    char* o = args[2];
    for (npy_intp k = 0; k < *dimensions; k++) {
      auto x = *reinterpret_cast<const typename TypeDescriptor<InType>::T*>(i0);
      auto y = *reinterpret_cast<const typename TypeDescriptor<InType>::T*>(i1);
      *reinterpret_cast<typename TypeDescriptor<OutType>::T*>(o) =
          Functor()(x, y);
      i0 += steps[0];
      i1 += steps[1];
      o += steps[2];
    }
  }
};

template <typename InType, typename InType2, typename OutType, typename Functor>
struct BinaryUFunc2 {
  static std::vector<int> Types() {
    return {TypeDescriptor<InType>::Dtype(), TypeDescriptor<InType2>::Dtype(),
            TypeDescriptor<OutType>::Dtype()};
  }
  static void Call(char** args, const npy_intp* dimensions,
                   const npy_intp* steps, void* data) {
    const char* i0 = args[0];
    const char* i1 = args[1];
    char* o = args[2];
    for (npy_intp k = 0; k < *dimensions; k++) {
      auto x = *reinterpret_cast<const typename TypeDescriptor<InType>::T*>(i0);
      auto y =
          *reinterpret_cast<const typename TypeDescriptor<InType2>::T*>(i1);
      *reinterpret_cast<typename TypeDescriptor<OutType>::T*>(o) =
          Functor()(x, y);
      i0 += steps[0];
      i1 += steps[1];
      o += steps[2];
    }
  }
};

template <typename UFunc, typename CustomFloatT>
bool RegisterUFunc(PyObject* numpy, const char* name) {
  std::vector<int> types = UFunc::Types();
  PyUFuncGenericFunction fn =
      reinterpret_cast<PyUFuncGenericFunction>(UFunc::Call);
  Safe_PyObjectPtr ufunc_obj = make_safe(PyObject_GetAttrString(numpy, name));
  if (!ufunc_obj) {
    return false;
  }
  PyUFuncObject* ufunc = reinterpret_cast<PyUFuncObject*>(ufunc_obj.get());
  if (static_cast<int>(types.size()) != ufunc->nargs) {
    PyErr_Format(PyExc_AssertionError,
                 "ufunc %s takes %d arguments, loop takes %lu", name,
                 ufunc->nargs, types.size());
    return false;
  }
  if (PyUFunc_RegisterLoopForType(ufunc, TypeDescriptor<CustomFloatT>::Dtype(),
                                  fn, const_cast<int*>(types.data()),
                                  nullptr) < 0) {
    return false;
  }
  return true;
}

namespace ufuncs {

template <typename T>
struct Add {
  T operator()(T a, T b) { return a + b; }
};
template <typename T>
struct Subtract {
  T operator()(T a, T b) { return a - b; }
};
template <typename T>
struct Multiply {
  T operator()(T a, T b) { return a * b; }
};
template <typename T>
struct TrueDivide {
  T operator()(T a, T b) { return a / b; }
};

inline std::pair<float, float> divmod(float a, float b) {
  if (b == 0.0f) {
    float nan = std::numeric_limits<float>::quiet_NaN();
    return {nan, nan};
  }
  float mod = std::fmod(a, b);
  float div = (a - mod) / b;
  if (mod != 0.0f) {
    if ((b < 0.0f) != (mod < 0.0f)) {
      mod += b;
      div -= 1.0f;
    }
  } else {
    mod = std::copysign(0.0f, b);
  }

  float floordiv;
  if (div != 0.0f) {
    floordiv = std::floor(div);
    if (div - floordiv > 0.5f) {
      floordiv += 1.0f;
    }
  } else {
    floordiv = std::copysign(0.0f, a / b);
  }
  return {floordiv, mod};
}

template <typename T>
struct FloorDivide {
  T operator()(T a, T b) {
    return T(divmod(static_cast<float>(a), static_cast<float>(b)).first);
  }
};
template <typename T>
struct Remainder {
  T operator()(T a, T b) {
    return T(divmod(static_cast<float>(a), static_cast<float>(b)).second);
  }
};
template <typename T>
struct DivmodUFunc {
  static std::vector<int> Types() {
    return {TypeDescriptor<T>::Dtype(), TypeDescriptor<T>::Dtype(),
            TypeDescriptor<T>::Dtype(), TypeDescriptor<T>::Dtype()};
  }
  static void Call(char** args, npy_intp* dimensions, npy_intp* steps,
                   void* data) {
    const char* i0 = args[0];
    const char* i1 = args[1];
    char* o0 = args[2];
    char* o1 = args[3];
    for (npy_intp k = 0; k < *dimensions; k++) {
      T x = *reinterpret_cast<const T*>(i0);
      T y = *reinterpret_cast<const T*>(i1);
      float floordiv, mod;
      std::tie(floordiv, mod) =
          divmod(static_cast<float>(x), static_cast<float>(y));
      *reinterpret_cast<T*>(o0) = T(floordiv);
      *reinterpret_cast<T*>(o1) = T(mod);
      i0 += steps[0];
      i1 += steps[1];
      o0 += steps[2];
      o1 += steps[3];
    }
  }
};
template <typename T>
struct Fmod {
  T operator()(T a, T b) {
    return T(std::fmod(static_cast<float>(a), static_cast<float>(b)));
  }
};
template <typename T>
struct Negative {
  T operator()(T a) { return -a; }
};
template <typename T>
struct Positive {
  T operator()(T a) { return a; }
};
template <typename T>
struct Power {
  T operator()(T a, T b) {
    return T(std::pow(static_cast<float>(a), static_cast<float>(b)));
  }
};
template <typename T>
struct Abs {
  T operator()(T a) { return T(std::abs(static_cast<float>(a))); }
};
template <typename T>
struct Cbrt {
  T operator()(T a) { return T(std::cbrt(static_cast<float>(a))); }
};
template <typename T>
struct Ceil {
  T operator()(T a) { return T(std::ceil(static_cast<float>(a))); }
};
template <typename T>
struct CopySign;

template <typename T>
struct Exp {
  T operator()(T a) { return T(std::exp(static_cast<float>(a))); }
};
template <typename T>
struct Exp2 {
  T operator()(T a) { return T(std::exp2(static_cast<float>(a))); }
};
template <typename T>
struct Expm1 {
  T operator()(T a) { return T(std::expm1(static_cast<float>(a))); }
};
template <typename T>
struct Floor {
  T operator()(T a) { return T(std::floor(static_cast<float>(a))); }
};
template <typename T>
struct Frexp {
  std::pair<T, int> operator()(T a) {
    int exp;
    float f = std::frexp(static_cast<float>(a), &exp);
    return {T(f), exp};
  }
};
template <typename T>
struct Heaviside {
  T operator()(T bx, T h0) {
    float x = static_cast<float>(bx);
    if (Eigen::numext::isnan(x)) {
      return bx;
    }
    if (x < 0) {
      return T(0.0f);
    }
    if (x > 0) {
      return T(1.0f);
    }
    return h0;  // x == 0
  }
};
template <typename T>
struct Conjugate {
  T operator()(T a) { return a; }
};
template <typename T>
struct IsFinite {
  bool operator()(T a) { return std::isfinite(static_cast<float>(a)); }
};
template <typename T>
struct IsInf {
  bool operator()(T a) { return std::isinf(static_cast<float>(a)); }
};
template <typename T>
struct IsNan {
  bool operator()(T a) { return Eigen::numext::isnan(static_cast<float>(a)); }
};
template <typename T>
struct Ldexp {
  T operator()(T a, int exp) {
    return T(std::ldexp(static_cast<float>(a), exp));
  }
};
template <typename T>
struct Log {
  T operator()(T a) { return T(std::log(static_cast<float>(a))); }
};
template <typename T>
struct Log2 {
  T operator()(T a) { return T(std::log2(static_cast<float>(a))); }
};
template <typename T>
struct Log10 {
  T operator()(T a) { return T(std::log10(static_cast<float>(a))); }
};
template <typename T>
struct Log1p {
  T operator()(T a) { return T(std::log1p(static_cast<float>(a))); }
};
template <typename T>
struct LogAddExp {
  T operator()(T bx, T by) {
    float x = static_cast<float>(bx);
    float y = static_cast<float>(by);
    if (x == y) {
      // Handles infinities of the same sign.
      return T(x + std::log(2.0f));
    }
    float out = std::numeric_limits<float>::quiet_NaN();
    if (x > y) {
      out = x + std::log1p(std::exp(y - x));
    } else if (x < y) {
      out = y + std::log1p(std::exp(x - y));
    }
    return T(out);
  }
};
template <typename T>
struct LogAddExp2 {
  T operator()(T bx, T by) {
    float x = static_cast<float>(bx);
    float y = static_cast<float>(by);
    if (x == y) {
      // Handles infinities of the same sign.
      return T(x + 1.0f);
    }
    float out = std::numeric_limits<float>::quiet_NaN();
    if (x > y) {
      out = x + std::log1p(std::exp2(y - x)) / std::log(2.0f);
    } else if (x < y) {
      out = y + std::log1p(std::exp2(x - y)) / std::log(2.0f);
    }
    return T(out);
  }
};
template <typename T>
struct Modf {
  std::pair<T, T> operator()(T a) {
    float integral;
    float f = std::modf(static_cast<float>(a), &integral);
    return {T(f), T(integral)};
  }
};

template <typename T>
struct Reciprocal {
  T operator()(T a) { return T(1.f / static_cast<float>(a)); }
};
template <typename T>
struct Rint {
  T operator()(T a) { return T(std::rint(static_cast<float>(a))); }
};
template <typename T>
struct Sign {
  T operator()(T a) {
    float f(a);
    if (f < 0) {
      return T(-1);
    }
    if (f > 0) {
      return T(1);
    }
    return a;
  }
};
template <typename T>
struct SignBit {
  bool operator()(T a) { return std::signbit(static_cast<float>(a)); }
};
template <typename T>
struct Sqrt {
  T operator()(T a) { return T(std::sqrt(static_cast<float>(a))); }
};
template <typename T>
struct Square {
  T operator()(T a) {
    float f(a);
    return T(f * f);
  }
};
template <typename T>
struct Trunc {
  T operator()(T a) { return T(std::trunc(static_cast<float>(a))); }
};

// Trigonometric functions
template <typename T>
struct Sin {
  T operator()(T a) { return T(std::sin(static_cast<float>(a))); }
};
template <typename T>
struct Cos {
  T operator()(T a) { return T(std::cos(static_cast<float>(a))); }
};
template <typename T>
struct Tan {
  T operator()(T a) { return T(std::tan(static_cast<float>(a))); }
};
template <typename T>
struct Arcsin {
  T operator()(T a) { return T(std::asin(static_cast<float>(a))); }
};
template <typename T>
struct Arccos {
  T operator()(T a) { return T(std::acos(static_cast<float>(a))); }
};
template <typename T>
struct Arctan {
  T operator()(T a) { return T(std::atan(static_cast<float>(a))); }
};
template <typename T>
struct Arctan2 {
  T operator()(T a, T b) {
    return T(std::atan2(static_cast<float>(a), static_cast<float>(b)));
  }
};
template <typename T>
struct Hypot {
  T operator()(T a, T b) {
    return T(std::hypot(static_cast<float>(a), static_cast<float>(b)));
  }
};
template <typename T>
struct Sinh {
  T operator()(T a) { return T(std::sinh(static_cast<float>(a))); }
};
template <typename T>
struct Cosh {
  T operator()(T a) { return T(std::cosh(static_cast<float>(a))); }
};
template <typename T>
struct Tanh {
  T operator()(T a) { return T(std::tanh(static_cast<float>(a))); }
};
template <typename T>
struct Arcsinh {
  T operator()(T a) { return T(std::asinh(static_cast<float>(a))); }
};
template <typename T>
struct Arccosh {
  T operator()(T a) { return T(std::acosh(static_cast<float>(a))); }
};
template <typename T>
struct Arctanh {
  T operator()(T a) { return T(std::atanh(static_cast<float>(a))); }
};
template <typename T>
struct Deg2rad {
  T operator()(T a) {
    static constexpr float radians_per_degree = M_PI / 180.0f;
    return T(static_cast<float>(a) * radians_per_degree);
  }
};
template <typename T>
struct Rad2deg {
  T operator()(T a) {
    static constexpr float degrees_per_radian = 180.0f / M_PI;
    return T(static_cast<float>(a) * degrees_per_radian);
  }
};

template <typename T>
struct Eq {
  npy_bool operator()(T a, T b) { return a == b; }
};
template <typename T>
struct Ne {
  npy_bool operator()(T a, T b) { return a != b; }
};
template <typename T>
struct Lt {
  npy_bool operator()(T a, T b) { return a < b; }
};
template <typename T>
struct Gt {
  npy_bool operator()(T a, T b) { return a > b; }
};
template <typename T>
struct Le {
  npy_bool operator()(T a, T b) { return a <= b; }
};
template <typename T>
struct Ge {
  npy_bool operator()(T a, T b) { return a >= b; }
};
template <typename T>
struct Maximum {
  T operator()(T a, T b) {
    float fa(a), fb(b);
    return Eigen::numext::isnan(fa) || fa > fb ? a : b;
  }
};
template <typename T>
struct Minimum {
  T operator()(T a, T b) {
    float fa(a), fb(b);
    return Eigen::numext::isnan(fa) || fa < fb ? a : b;
  }
};
template <typename T>
struct Fmax {
  T operator()(T a, T b) {
    float fa(a), fb(b);
    return Eigen::numext::isnan(fb) || fa > fb ? a : b;
  }
};
template <typename T>
struct Fmin {
  T operator()(T a, T b) {
    float fa(a), fb(b);
    return Eigen::numext::isnan(fb) || fa < fb ? a : b;
  }
};

template <typename T>
struct LogicalNot {
  npy_bool operator()(T a) { return !static_cast<bool>(a); }
};
template <typename T>
struct LogicalAnd {
  npy_bool operator()(T a, T b) {
    return static_cast<bool>(a) && static_cast<bool>(b);
  }
};
template <typename T>
struct LogicalOr {
  npy_bool operator()(T a, T b) {
    return static_cast<bool>(a) || static_cast<bool>(b);
  }
};
template <typename T>
struct LogicalXor {
  npy_bool operator()(T a, T b) {
    return static_cast<bool>(a) ^ static_cast<bool>(b);
  }
};

template <typename T>
struct NextAfter;

template <typename T>
struct Spacing {
  T operator()(T x) {
    // Compute the distance between the input and the next number with greater
    // magnitude. The result should have the sign of the input.
    T away(std::copysign(std::numeric_limits<float>::infinity(),
                         static_cast<float>(x)));
    return NextAfter<T>()(x, away) - x;
  }
};

template <typename T>
bool RegisterUFuncs(PyObject* numpy) {
  bool ok =
      RegisterUFunc<BinaryUFunc<T, T, ufuncs::Add<T>>, T>(numpy, "add") &&
      RegisterUFunc<BinaryUFunc<T, T, ufuncs::Subtract<T>>, T>(numpy,
                                                               "subtract") &&
      RegisterUFunc<BinaryUFunc<T, T, ufuncs::Multiply<T>>, T>(numpy,
                                                               "multiply") &&
      RegisterUFunc<BinaryUFunc<T, T, ufuncs::TrueDivide<T>>, T>(numpy,
                                                                 "divide") &&
      RegisterUFunc<BinaryUFunc<T, T, ufuncs::LogAddExp<T>>, T>(numpy,
                                                                "logaddexp") &&
      RegisterUFunc<BinaryUFunc<T, T, ufuncs::LogAddExp2<T>>, T>(
          numpy, "logaddexp2") &&
      RegisterUFunc<UnaryUFunc<T, T, ufuncs::Negative<T>>, T>(numpy,
                                                              "negative") &&
      RegisterUFunc<UnaryUFunc<T, T, ufuncs::Positive<T>>, T>(numpy,
                                                              "positive") &&
      RegisterUFunc<BinaryUFunc<T, T, ufuncs::TrueDivide<T>>, T>(
          numpy, "true_divide") &&
      RegisterUFunc<BinaryUFunc<T, T, ufuncs::FloorDivide<T>>, T>(
          numpy, "floor_divide") &&
      RegisterUFunc<BinaryUFunc<T, T, ufuncs::Power<T>>, T>(numpy, "power") &&
      RegisterUFunc<BinaryUFunc<T, T, ufuncs::Remainder<T>>, T>(numpy,
                                                                "remainder") &&
      RegisterUFunc<BinaryUFunc<T, T, ufuncs::Remainder<T>>, T>(numpy, "mod") &&
      RegisterUFunc<BinaryUFunc<T, T, ufuncs::Fmod<T>>, T>(numpy, "fmod") &&
      RegisterUFunc<ufuncs::DivmodUFunc<T>, T>(numpy, "divmod") &&
      RegisterUFunc<UnaryUFunc<T, T, ufuncs::Abs<T>>, T>(numpy, "absolute") &&
      RegisterUFunc<UnaryUFunc<T, T, ufuncs::Abs<T>>, T>(numpy, "fabs") &&
      RegisterUFunc<UnaryUFunc<T, T, ufuncs::Rint<T>>, T>(numpy, "rint") &&
      RegisterUFunc<UnaryUFunc<T, T, ufuncs::Sign<T>>, T>(numpy, "sign") &&
      RegisterUFunc<BinaryUFunc<T, T, ufuncs::Heaviside<T>>, T>(numpy,
                                                                "heaviside") &&
      RegisterUFunc<UnaryUFunc<T, T, ufuncs::Conjugate<T>>, T>(numpy,
                                                               "conjugate") &&
      RegisterUFunc<UnaryUFunc<T, T, ufuncs::Exp<T>>, T>(numpy, "exp") &&
      RegisterUFunc<UnaryUFunc<T, T, ufuncs::Exp2<T>>, T>(numpy, "exp2") &&
      RegisterUFunc<UnaryUFunc<T, T, ufuncs::Expm1<T>>, T>(numpy, "expm1") &&
      RegisterUFunc<UnaryUFunc<T, T, ufuncs::Log<T>>, T>(numpy, "log") &&
      RegisterUFunc<UnaryUFunc<T, T, ufuncs::Log2<T>>, T>(numpy, "log2") &&
      RegisterUFunc<UnaryUFunc<T, T, ufuncs::Log10<T>>, T>(numpy, "log10") &&
      RegisterUFunc<UnaryUFunc<T, T, ufuncs::Log1p<T>>, T>(numpy, "log1p") &&
      RegisterUFunc<UnaryUFunc<T, T, ufuncs::Sqrt<T>>, T>(numpy, "sqrt") &&
      RegisterUFunc<UnaryUFunc<T, T, ufuncs::Square<T>>, T>(numpy, "square") &&
      RegisterUFunc<UnaryUFunc<T, T, ufuncs::Cbrt<T>>, T>(numpy, "cbrt") &&
      RegisterUFunc<UnaryUFunc<T, T, ufuncs::Reciprocal<T>>, T>(numpy,
                                                                "reciprocal") &&

      // Trigonometric functions
      RegisterUFunc<UnaryUFunc<T, T, ufuncs::Sin<T>>, T>(numpy, "sin") &&
      RegisterUFunc<UnaryUFunc<T, T, ufuncs::Cos<T>>, T>(numpy, "cos") &&
      RegisterUFunc<UnaryUFunc<T, T, ufuncs::Tan<T>>, T>(numpy, "tan") &&
      RegisterUFunc<UnaryUFunc<T, T, ufuncs::Arcsin<T>>, T>(numpy, "arcsin") &&
      RegisterUFunc<UnaryUFunc<T, T, ufuncs::Arccos<T>>, T>(numpy, "arccos") &&
      RegisterUFunc<UnaryUFunc<T, T, ufuncs::Arctan<T>>, T>(numpy, "arctan") &&
      RegisterUFunc<BinaryUFunc<T, T, ufuncs::Arctan2<T>>, T>(numpy,
                                                              "arctan2") &&
      RegisterUFunc<BinaryUFunc<T, T, ufuncs::Hypot<T>>, T>(numpy, "hypot") &&
      RegisterUFunc<UnaryUFunc<T, T, ufuncs::Sinh<T>>, T>(numpy, "sinh") &&
      RegisterUFunc<UnaryUFunc<T, T, ufuncs::Cosh<T>>, T>(numpy, "cosh") &&
      RegisterUFunc<UnaryUFunc<T, T, ufuncs::Tanh<T>>, T>(numpy, "tanh") &&
      RegisterUFunc<UnaryUFunc<T, T, ufuncs::Arcsinh<T>>, T>(numpy,
                                                             "arcsinh") &&
      RegisterUFunc<UnaryUFunc<T, T, ufuncs::Arccosh<T>>, T>(numpy,
                                                             "arccosh") &&
      RegisterUFunc<UnaryUFunc<T, T, ufuncs::Arctanh<T>>, T>(numpy,
                                                             "arctanh") &&
      RegisterUFunc<UnaryUFunc<T, T, ufuncs::Deg2rad<T>>, T>(numpy,
                                                             "deg2rad") &&
      RegisterUFunc<UnaryUFunc<T, T, ufuncs::Rad2deg<T>>, T>(numpy,
                                                             "rad2deg") &&

      // Comparison functions
      RegisterUFunc<BinaryUFunc<T, bool, ufuncs::Eq<T>>, T>(numpy, "equal") &&
      RegisterUFunc<BinaryUFunc<T, bool, ufuncs::Ne<T>>, T>(numpy,
                                                            "not_equal") &&
      RegisterUFunc<BinaryUFunc<T, bool, ufuncs::Lt<T>>, T>(numpy, "less") &&
      RegisterUFunc<BinaryUFunc<T, bool, ufuncs::Gt<T>>, T>(numpy, "greater") &&
      RegisterUFunc<BinaryUFunc<T, bool, ufuncs::Le<T>>, T>(numpy,
                                                            "less_equal") &&
      RegisterUFunc<BinaryUFunc<T, bool, ufuncs::Ge<T>>, T>(numpy,
                                                            "greater_equal") &&
      RegisterUFunc<BinaryUFunc<T, T, ufuncs::Maximum<T>>, T>(numpy,
                                                              "maximum") &&
      RegisterUFunc<BinaryUFunc<T, T, ufuncs::Minimum<T>>, T>(numpy,
                                                              "minimum") &&
      RegisterUFunc<BinaryUFunc<T, T, ufuncs::Fmax<T>>, T>(numpy, "fmax") &&
      RegisterUFunc<BinaryUFunc<T, T, ufuncs::Fmin<T>>, T>(numpy, "fmin") &&
      RegisterUFunc<BinaryUFunc<T, bool, ufuncs::LogicalAnd<T>>, T>(
          numpy, "logical_and") &&
      RegisterUFunc<BinaryUFunc<T, bool, ufuncs::LogicalOr<T>>, T>(
          numpy, "logical_or") &&
      RegisterUFunc<BinaryUFunc<T, bool, ufuncs::LogicalXor<T>>, T>(
          numpy, "logical_xor") &&
      RegisterUFunc<UnaryUFunc<T, bool, ufuncs::LogicalNot<T>>, T>(
          numpy, "logical_not") &&

      // Floating point functions
      RegisterUFunc<UnaryUFunc<T, bool, ufuncs::IsFinite<T>>, T>(numpy,
                                                                 "isfinite") &&
      RegisterUFunc<UnaryUFunc<T, bool, ufuncs::IsInf<T>>, T>(numpy, "isinf") &&
      RegisterUFunc<UnaryUFunc<T, bool, ufuncs::IsNan<T>>, T>(numpy, "isnan") &&
      RegisterUFunc<UnaryUFunc<T, bool, ufuncs::SignBit<T>>, T>(numpy,
                                                                "signbit") &&
      RegisterUFunc<BinaryUFunc<T, T, ufuncs::CopySign<T>>, T>(numpy,
                                                               "copysign") &&
      RegisterUFunc<UnaryUFunc2<T, T, T, ufuncs::Modf<T>>, T>(numpy, "modf") &&
      RegisterUFunc<BinaryUFunc2<T, int, T, ufuncs::Ldexp<T>>, T>(numpy,
                                                                  "ldexp") &&
      RegisterUFunc<UnaryUFunc2<T, T, int, ufuncs::Frexp<T>>, T>(numpy,
                                                                 "frexp") &&
      RegisterUFunc<UnaryUFunc<T, T, ufuncs::Floor<T>>, T>(numpy, "floor") &&
      RegisterUFunc<UnaryUFunc<T, T, ufuncs::Ceil<T>>, T>(numpy, "ceil") &&
      RegisterUFunc<UnaryUFunc<T, T, ufuncs::Trunc<T>>, T>(numpy, "trunc") &&
      RegisterUFunc<BinaryUFunc<T, T, ufuncs::NextAfter<T>>, T>(numpy,
                                                                "nextafter") &&
      RegisterUFunc<UnaryUFunc<T, T, ufuncs::Spacing<T>>, T>(numpy, "spacing");

  return ok;
}

}  // namespace ufuncs

template <typename T>
bool RegisterNumpyDtype(PyObject* numpy) {
  // If another module (presumably either TF or JAX) has registered a bfloat16
  // type, use it. We don't want two bfloat16 types if we can avoid it since it
  // leads to confusion if we have two different types with the same name. This
  // assumes that the other module has a sufficiently complete bfloat16
  // implementation. The only known NumPy bfloat16 extension at the time of
  // writing is this one (distributed in TF and JAX).
  // TODO(phawkins): distribute the bfloat16 extension as its own pip package,
  // so we can unambiguously refer to a single canonical definition of bfloat16.
  int typenum =
      PyArray_TypeNumFromName(const_cast<char*>(TypeDescriptor<T>::kTypeName));
  if (typenum != NPY_NOTYPE) {
    PyArray_Descr* descr = PyArray_DescrFromType(typenum);
    // The test for an argmax function here is to verify that the
    // bfloat16 implementation is sufficiently new, and, say, not from
    // an older version of TF or JAX.
    if (descr && descr->f && descr->f->argmax) {
      TypeDescriptor<T>::npy_type = typenum;
      TypeDescriptor<T>::type_ptr = descr->typeobj;
      return true;
    }
  }

  TypeDescriptor<T>::type.tp_base = &PyGenericArrType_Type;

  if (PyType_Ready(&TypeDescriptor<T>::type) < 0) {
    return false;
  }

  // Initializes the NumPy descriptor.
  PyArray_ArrFuncs& arr_funcs = CustomFloatTypeDescriptor<T>::arr_funcs;
  PyArray_InitArrFuncs(&arr_funcs);
  arr_funcs.getitem = NPyCustomFloat_GetItem<T>;
  arr_funcs.setitem = NPyCustomFloat_SetItem<T>;
  arr_funcs.compare = NPyCustomFloat_Compare<T>;
  arr_funcs.copyswapn = NPyCustomFloat_CopySwapN<T>;
  arr_funcs.copyswap = NPyCustomFloat_CopySwap<T>;
  arr_funcs.nonzero = NPyCustomFloat_NonZero<T>;
  arr_funcs.fill = NPyCustomFloat_Fill<T>;
  arr_funcs.dotfunc = NPyCustomFloat_DotFunc<T>;
  arr_funcs.compare = NPyCustomFloat_CompareFunc<T>;
  arr_funcs.argmax = NPyCustomFloat_ArgMaxFunc<T>;
  arr_funcs.argmin = NPyCustomFloat_ArgMinFunc<T>;

#if PY_VERSION_HEX < 0x030900A4 && !defined(Py_SET_TYPE)
  Py_TYPE(&CustomFloatTypeDescriptor<T>::npy_descr) = &PyArrayDescr_Type;
#else
  Py_SET_TYPE(&CustomFloatTypeDescriptor<T>::npy_descr, &PyArrayDescr_Type);
#endif
  TypeDescriptor<T>::npy_type =
      PyArray_RegisterDataType(&CustomFloatTypeDescriptor<T>::npy_descr);
  TypeDescriptor<T>::type_ptr = &TypeDescriptor<T>::type;
  if (TypeDescriptor<T>::Dtype() < 0) {
    return false;
  }

  Safe_PyObjectPtr typeDict_obj =
      make_safe(PyObject_GetAttrString(numpy, "sctypeDict"));
  if (!typeDict_obj) return false;
  // Add the type object to `numpy.typeDict`: that makes
  // `numpy.dtype(type_name)` work.
  if (PyDict_SetItemString(
          typeDict_obj.get(), TypeDescriptor<T>::kTypeName,
          reinterpret_cast<PyObject*>(&TypeDescriptor<T>::type)) < 0) {
    return false;
  }

  // Support dtype(type_name)
  if (PyDict_SetItemString(TypeDescriptor<T>::type.tp_dict, "dtype",
                           reinterpret_cast<PyObject*>(
                               &CustomFloatTypeDescriptor<T>::npy_descr)) < 0) {
    return false;
  }

  return RegisterCasts<T>() && ufuncs::RegisterUFuncs<T>(numpy);
}

}  // namespace custom_float_internal
}  // namespace tensorflow

#endif  // TENSORFLOW_PYTHON_LIB_CORE_CUSTOM_FLOAT_H_
