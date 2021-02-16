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

#include "tensorflow/compiler/xla/python/cus.h"

#include <array>
#include <locale>
// Place `<locale>` before <Python.h> to avoid a build failure in macOS.
#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include "numpy/arrayobject.h"
#include "numpy/ufuncobject.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/platform/cus.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {
namespace {

namespace py = pybind11;

struct PyDecrefDeleter {
  void operator()(PyObject* p) const { Py_DECREF(p); }
};

// Safe container for an owned PyObject. On destruction, the reference count of
// the contained object will be decremented.
using Safe_PyObjectPtr = std::unique_ptr<PyObject, PyDecrefDeleter>;
Safe_PyObjectPtr make_safe(PyObject* object) {
  return Safe_PyObjectPtr(object);
}

bool PyLong_CheckNoOverflow(PyObject* object) {
  if (!PyLong_Check(object)) {
    return false;
  }
  int overflow = 0;
  PyLong_AsLongAndOverflow(object, &overflow);
  return (overflow == 0);
}

// Registered numpy type ID. Global variable populated by the registration code.
// Protected by the GIL.
int npy_cus = -1;

// Forward declaration.
extern PyTypeObject PyCus_Type;

// Representation of a Python cus object.
struct PyCus {
  PyObject_HEAD;  // Python object header
  cus value;
};

// Returns true if 'object' is a PyCus.
bool PyCus_Check(PyObject* object) {
  return PyObject_IsInstance(object,
                             reinterpret_cast<PyObject*>(&PyCus_Type));
}

// Extracts the value of a PyCus object.
cus PyCus_Cus(PyObject* object) {
  return reinterpret_cast<PyCus*>(object)->value;
}

// Constructs a PyCus object from a cus.
Safe_PyObjectPtr PyCus_FromCus(cus x) {
  Safe_PyObjectPtr ref =
      make_safe(PyCus_Type.tp_alloc(&PyCus_Type, 0));
  PyCus* p = reinterpret_cast<PyCus*>(ref.get());
  if (p) {
    p->value = x;
  }
  return ref;
}

// Converts a Python object to a cus value. Returns true on success,
// returns false and reports a Python error on failure.
bool CastToCus(PyObject* arg, cus* output) {
  if (PyCus_Check(arg)) {
    *output = PyCus_Cus(arg);
    return true;
  }
  if (PyFloat_Check(arg)) {
    double d = PyFloat_AsDouble(arg);
    if (PyErr_Occurred()) {
      return false;
    }
    // TODO(phawkins): check for overflow
    *output = cus(d);
    return true;
  }
  if (PyLong_CheckNoOverflow(arg)) {
    long l = PyLong_AsLong(arg);  // NOLINT
    if (PyErr_Occurred()) {
      return false;
    }
    // TODO(phawkins): check for overflow
    *output = cus(static_cast<float>(l));
    return true;
  }
  if (PyArray_IsScalar(arg, Half)) {
    Eigen::half f;
    PyArray_ScalarAsCtype(arg, &f);
    *output = cus(f);
    return true;
  }
  if (PyArray_IsScalar(arg, Float)) {
    float f;
    PyArray_ScalarAsCtype(arg, &f);
    *output = cus(f);
    return true;
  }
  if (PyArray_IsScalar(arg, Double)) {
    double f;
    PyArray_ScalarAsCtype(arg, &f);
    *output = cus(f);
    return true;
  }
  if (PyArray_IsZeroDim(arg)) {
    Safe_PyObjectPtr ref;
    PyArrayObject* arr = reinterpret_cast<PyArrayObject*>(arg);
    if (PyArray_TYPE(arr) != npy_cus) {
      ref = make_safe(PyArray_Cast(arr, npy_cus));
      if (PyErr_Occurred()) {
        return false;
      }
      arg = ref.get();
      arr = reinterpret_cast<PyArrayObject*>(arg);
    }
    *output = *reinterpret_cast<cus*>(PyArray_DATA(arr));
    return true;
  }
  return false;
}

bool SafeCastToCus(PyObject* arg, cus* output) {
  if (PyCus_Check(arg)) {
    *output = PyCus_Cus(arg);
    return true;
  }
  return false;
}

// Converts a PyCus into a PyFloat.
PyObject* PyCus_Float(PyObject* self) {
  cus x = PyCus_Cus(self);
  return PyFloat_FromDouble(static_cast<double>(x));
}

// Converts a PyCus into a PyInt.
PyObject* PyCus_Int(PyObject* self) {
  cus x = PyCus_Cus(self);
  long y = static_cast<long>(x);  // NOLINT
  return PyLong_FromLong(y);
}

// Negates a PyCus.
PyObject* PyCus_Negative(PyObject* self) {
  cus x = PyCus_Cus(self);
  return PyCus_FromCus(-x).release();
}

PyObject* PyCus_Add(PyObject* a, PyObject* b) {
  cus x, y;
  if (SafeCastToCus(a, &x) && SafeCastToCus(b, &y)) {
    return PyCus_FromCus(x + y).release();
  }
  return PyArray_Type.tp_as_number->nb_add(a, b);
}

PyObject* PyCus_Subtract(PyObject* a, PyObject* b) {
  cus x, y;
  if (SafeCastToCus(a, &x) && SafeCastToCus(b, &y)) {
    return PyCus_FromCus(x - y).release();
  }
  return PyArray_Type.tp_as_number->nb_subtract(a, b);
}

PyObject* PyCus_Multiply(PyObject* a, PyObject* b) {
  cus x, y;
  if (SafeCastToCus(a, &x) && SafeCastToCus(b, &y)) {
    return PyCus_FromCus(x * y).release();
  }
  return PyArray_Type.tp_as_number->nb_multiply(a, b);
}

PyObject* PyCus_TrueDivide(PyObject* a, PyObject* b) {
  cus x, y;
  if (SafeCastToCus(a, &x) && SafeCastToCus(b, &y)) {
    return PyCus_FromCus(x / y).release();
  }
  return PyArray_Type.tp_as_number->nb_true_divide(a, b);
}

// Python number methods for PyCus objects.
PyNumberMethods PyCus_AsNumber = {
    PyCus_Add,       // nb_add
    PyCus_Subtract,  // nb_subtract
    PyCus_Multiply,  // nb_multiply
    nullptr,              // nb_remainder
    nullptr,              // nb_divmod
    nullptr,              // nb_power
    PyCus_Negative,  // nb_negative
    nullptr,              // nb_positive
    nullptr,              // nb_absolute
    nullptr,              // nb_nonzero
    nullptr,              // nb_invert
    nullptr,              // nb_lshift
    nullptr,              // nb_rshift
    nullptr,              // nb_and
    nullptr,              // nb_xor
    nullptr,              // nb_or
    PyCus_Int,       // nb_int
    nullptr,              // reserved
    PyCus_Float,     // nb_float

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

    nullptr,                // nb_floor_divide
    PyCus_TrueDivide,  // nb_true_divide
    nullptr,                // nb_inplace_floor_divide
    nullptr,                // nb_inplace_true_divide
    nullptr,                // nb_index
};

// Constructs a new PyCus.
PyObject* PyCus_New(PyTypeObject* type, PyObject* args, PyObject* kwds) {
  if (kwds && PyDict_Size(kwds)) {
    PyErr_SetString(PyExc_TypeError, "constructor takes no keyword arguments");
    return nullptr;
  }
  Py_ssize_t size = PyTuple_Size(args);
  if (size != 1) {
    PyErr_SetString(PyExc_TypeError,
                    "expected number as argument to cus constructor");
    return nullptr;
  }
  PyObject* arg = PyTuple_GetItem(args, 0);

  cus value;
  if (PyCus_Check(arg)) {
    Py_INCREF(arg);
    return arg;
  } else if (CastToCus(arg, &value)) {
    return PyCus_FromCus(value).release();
  } else if (PyArray_Check(arg)) {
    PyArrayObject* arr = reinterpret_cast<PyArrayObject*>(arg);
    if (PyArray_TYPE(arr) != npy_cus) {
      return PyArray_Cast(arr, npy_cus);
    } else {
      Py_INCREF(arg);
      return arg;
    }
  }
  PyErr_Format(PyExc_TypeError, "expected number, got %s",
               arg->ob_type->tp_name);
  return nullptr;
}

// Comparisons on PyCuss.
PyObject* PyCus_RichCompare(PyObject* a, PyObject* b, int op) {
  cus x, y;
  if (!SafeCastToCus(a, &x) || !SafeCastToCus(b, &y)) {
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
      LOG(FATAL) << "Invalid op type " << op;
  }
  return PyBool_FromLong(result);
}

// Implementation of repr() for PyCus.
PyObject* PyCus_Repr(PyObject* self) {
  cus x = reinterpret_cast<PyCus*>(self)->value;
  std::string v = absl::StrCat(static_cast<float>(x));
  return PyUnicode_FromString(v.c_str());
}

// Implementation of str() for PyCus.
PyObject* PyCus_Str(PyObject* self) {
  cus x = reinterpret_cast<PyCus*>(self)->value;
  std::string v = absl::StrCat(static_cast<float>(x));
  return PyUnicode_FromString(v.c_str());
}

// Hash function for PyCus. We use the identity function, which is a weak
// hash function.
Py_hash_t PyCus_Hash(PyObject* self) {
  cus x = reinterpret_cast<PyCus*>(self)->value;
  return x.value;
}

// Python type for PyCus objects.
PyTypeObject PyCus_Type = {
    PyVarObject_HEAD_INIT(nullptr, 0) "cus",  // tp_name
    sizeof(PyCus),                            // tp_basicsize
    0,                                             // tp_itemsize
    nullptr,                                       // tp_dealloc
#if PY_VERSION_HEX < 0x03080000
    nullptr,  // tp_print
#else
    0,  // tp_vectorcall_offset
#endif
    nullptr,               // tp_getattr
    nullptr,               // tp_setattr
    nullptr,               // tp_compare / tp_reserved
    PyCus_Repr,       // tp_repr
    &PyCus_AsNumber,  // tp_as_number
    nullptr,               // tp_as_sequence
    nullptr,               // tp_as_mapping
    PyCus_Hash,       // tp_hash
    nullptr,               // tp_call
    PyCus_Str,        // tp_str
    nullptr,               // tp_getattro
    nullptr,               // tp_setattro
    nullptr,               // tp_as_buffer
                           // tp_flags
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    "cus floating-point values",  // tp_doc
    nullptr,                           // tp_traverse
    nullptr,                           // tp_clear
    PyCus_RichCompare,            // tp_richcompare
    0,                                 // tp_weaklistoffset
    nullptr,                           // tp_iter
    nullptr,                           // tp_iternext
    nullptr,                           // tp_methods
    nullptr,                           // tp_members
    nullptr,                           // tp_getset
    nullptr,                           // tp_base
    nullptr,                           // tp_dict
    nullptr,                           // tp_descr_get
    nullptr,                           // tp_descr_set
    0,                                 // tp_dictoffset
    nullptr,                           // tp_init
    nullptr,                           // tp_alloc
    PyCus_New,                    // tp_new
    nullptr,                           // tp_free
    nullptr,                           // tp_is_gc
    nullptr,                           // tp_bases
    nullptr,                           // tp_mro
    nullptr,                           // tp_cache
    nullptr,                           // tp_subclasses
    nullptr,                           // tp_weaklist
    nullptr,                           // tp_del
    0,                                 // tp_version_tag
};

// Numpy support

PyArray_ArrFuncs NPyCus_ArrFuncs;

PyArray_Descr NPyCus_Descr = {
    PyObject_HEAD_INIT(nullptr)  //
                                 /*typeobj=*/
    (&PyCus_Type),
    // We must register cus with a kind other than "f", because numpy
    // considers two types with the same kind and size to be equal, but
    // float16 != cus.
    // The downside of this is that NumPy scalar promotion does not work with
    // cus values.
    /*kind=*/'V',
    // TODO(phawkins): there doesn't seem to be a way of guaranteeing a type
    // character is unique.
    /*type=*/'E',
    /*byteorder=*/'=',
    /*flags=*/NPY_NEEDS_PYAPI | NPY_USE_GETITEM | NPY_USE_SETITEM,
    /*type_num=*/0,
    /*elsize=*/sizeof(cus),
    /*alignment=*/alignof(cus),
    /*subarray=*/nullptr,
    /*fields=*/nullptr,
    /*names=*/nullptr,
    /*f=*/&NPyCus_ArrFuncs,
    /*metadata=*/nullptr,
    /*c_metadata=*/nullptr,
    /*hash=*/-1,  // -1 means "not computed yet".
};

// Implementations of NumPy array methods.

PyObject* NPyCus_GetItem(void* data, void* arr) {
  cus x;
  memcpy(&x, data, sizeof(cus));
  return PyCus_FromCus(x).release();
}

int NPyCus_SetItem(PyObject* item, void* data, void* arr) {
  cus x;
  if (!CastToCus(item, &x)) {
    PyErr_Format(PyExc_TypeError, "expected number, got %s",
                 item->ob_type->tp_name);
    return -1;
  }
  memcpy(data, &x, sizeof(cus));
  return 0;
}

void ByteSwap16(void* value) {
  char* p = reinterpret_cast<char*>(value);
  std::swap(p[0], p[1]);
}

int NPyCus_Compare(const void* a, const void* b, void* arr) {
  cus x;
  memcpy(&x, a, sizeof(cus));

  cus y;
  memcpy(&y, b, sizeof(cus));

  if (x < y) {
    return -1;
  }
  if (y < x) {
    return 1;
  }
  // NaNs sort to the end.
  if (!Eigen::numext::isnan(x) && Eigen::numext::isnan(y)) {
    return -1;
  }
  if (Eigen::numext::isnan(x) && !Eigen::numext::isnan(y)) {
    return 1;
  }
  return 0;
}

void NPyCus_CopySwapN(void* dstv, npy_intp dstride, void* srcv,
                           npy_intp sstride, npy_intp n, int swap, void* arr) {
  char* dst = reinterpret_cast<char*>(dstv);
  char* src = reinterpret_cast<char*>(srcv);
  if (!src) {
    return;
  }
  if (swap) {
    for (npy_intp i = 0; i < n; i++) {
      char* r = dst + dstride * i;
      memcpy(r, src + sstride * i, sizeof(uint16_t));
      ByteSwap16(r);
    }
  } else if (dstride == sizeof(uint16_t) && sstride == sizeof(uint16_t)) {
    memcpy(dst, src, n * sizeof(uint16_t));
  } else {
    for (npy_intp i = 0; i < n; i++) {
      memcpy(dst + dstride * i, src + sstride * i, sizeof(uint16_t));
    }
  }
}

void NPyCus_CopySwap(void* dst, void* src, int swap, void* arr) {
  if (!src) {
    return;
  }
  memcpy(dst, src, sizeof(uint16_t));
  if (swap) {
    ByteSwap16(dst);
  }
}

npy_bool NPyCus_NonZero(void* data, void* arr) {
  cus x;
  memcpy(&x, data, sizeof(x));
  return x != static_cast<cus>(0);
}

int NPyCus_Fill(void* buffer_raw, npy_intp length, void* ignored) {
  cus* const buffer = reinterpret_cast<cus*>(buffer_raw);
  const float start(buffer[0]);
  const float delta = static_cast<float>(buffer[1]) - start;
  for (npy_intp i = 2; i < length; ++i) {
    buffer[i] = static_cast<cus>(start + i * delta);
  }
  return 0;
}

void NPyCus_DotFunc(void* ip1, npy_intp is1, void* ip2, npy_intp is2,
                         void* op, npy_intp n, void* arr) {
  char* c1 = reinterpret_cast<char*>(ip1);
  char* c2 = reinterpret_cast<char*>(ip2);
  float acc = 0.0f;
  for (npy_intp i = 0; i < n; ++i) {
    cus* const b1 = reinterpret_cast<cus*>(c1);
    cus* const b2 = reinterpret_cast<cus*>(c2);
    acc += static_cast<float>(*b1) * static_cast<float>(*b2);
    c1 += is1;
    c2 += is2;
  }
  cus* out = reinterpret_cast<cus*>(op);
  *out = static_cast<cus>(acc);
}

int NPyCus_CompareFunc(const void* v1, const void* v2, void* arr) {
  cus b1 = *reinterpret_cast<const cus*>(v1);
  cus b2 = *reinterpret_cast<const cus*>(v2);
  if (b1 < b2) {
    return -1;
  }
  if (b1 > b2) {
    return 1;
  }
  return 0;
}

int NPyCus_ArgMaxFunc(void* data, npy_intp n, npy_intp* max_ind,
                           void* arr) {
  const cus* bdata = reinterpret_cast<const cus*>(data);
  float max_val = -std::numeric_limits<float>::infinity();
  for (npy_intp i = 0; i < n; ++i) {
    if (static_cast<float>(bdata[i]) > max_val) {
      max_val = static_cast<float>(bdata[i]);
      *max_ind = i;
    }
  }
  return 0;
}

int NPyCus_ArgMinFunc(void* data, npy_intp n, npy_intp* min_ind,
                           void* arr) {
  const cus* bdata = reinterpret_cast<const cus*>(data);
  float min_val = std::numeric_limits<float>::infinity();
  for (npy_intp i = 0; i < n; ++i) {
    if (static_cast<float>(bdata[i]) < min_val) {
      min_val = static_cast<float>(bdata[i]);
      *min_ind = i;
    }
  }
  return 0;
}

// NumPy casts

template <typename T, typename Enable = void>
struct TypeDescriptor {
  // typedef ... T;  // Representation type in memory for NumPy values of type
  // static int Dtype() { return NPY_...; }  // Numpy type number for T.
};

template <>
struct TypeDescriptor<cus> {
  typedef cus T;
  static int Dtype() { return npy_cus; }
};

template <>
struct TypeDescriptor<uint8> {
  typedef uint8 T;
  static int Dtype() { return NPY_UINT8; }
};

template <>
struct TypeDescriptor<uint16> {
  typedef uint16 T;
  static int Dtype() { return NPY_UINT16; }
};

template <>
struct TypeDescriptor<uint32> {
  typedef uint32 T;
  static int Dtype() { return NPY_UINT32; }
};

template <typename Uint64Type>
struct TypeDescriptor<
    Uint64Type, typename std::enable_if<std::is_integral<Uint64Type>::value &&
                                        !std::is_signed<Uint64Type>::value &&
                                        sizeof(Uint64Type) == 8>::type> {
  typedef Uint64Type T;
  static int Dtype() { return NPY_UINT64; }
};

template <>
struct TypeDescriptor<int8> {
  typedef int8 T;
  static int Dtype() { return NPY_INT8; }
};

template <>
struct TypeDescriptor<int16> {
  typedef int16 T;
  static int Dtype() { return NPY_INT16; }
};

template <>
struct TypeDescriptor<int32> {
  typedef int32 T;
  static int Dtype() { return NPY_INT32; }
};

template <typename Int64Type>
struct TypeDescriptor<
    Int64Type, typename std::enable_if<std::is_integral<Int64Type>::value &&
                                       std::is_signed<Int64Type>::value &&
                                       sizeof(Int64Type) == 8>::type> {
  typedef Int64Type T;
  static int Dtype() { return NPY_INT64; }
};

template <>
struct TypeDescriptor<bool> {
  typedef int8 T;
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
struct TypeDescriptor<complex64> {
  typedef complex64 T;
  static int Dtype() { return NPY_COMPLEX64; }
};

template <>
struct TypeDescriptor<complex128> {
  typedef complex128 T;
  static int Dtype() { return NPY_COMPLEX128; }
};

// Performs a NumPy array cast from type 'From' to 'To'.
template <typename From, typename To>
void NPyCast(void* from_void, void* to_void, npy_intp n, void* fromarr,
             void* toarr) {
  const auto* from =
      reinterpret_cast<typename TypeDescriptor<From>::T*>(from_void);
  auto* to = reinterpret_cast<typename TypeDescriptor<To>::T*>(to_void);
  for (npy_intp i = 0; i < n; ++i) {
    to[i] =
        static_cast<typename TypeDescriptor<To>::T>(static_cast<To>(from[i]));
  }
}

// Registers a cast between cus and type 'T'. 'numpy_type' is the NumPy
// type corresponding to 'T'. If 'cast_is_safe', registers that cus can be
// safely coerced to T.
template <typename T>
bool RegisterCusCast(int numpy_type, bool cast_is_safe) {
  if (PyArray_RegisterCastFunc(PyArray_DescrFromType(numpy_type), npy_cus,
                               NPyCast<T, cus>) < 0) {
    return false;
  }
  if (PyArray_RegisterCastFunc(&NPyCus_Descr, numpy_type,
                               NPyCast<cus, T>) < 0) {
    return false;
  }
  if (cast_is_safe && PyArray_RegisterCanCast(&NPyCus_Descr, numpy_type,
                                              NPY_NOSCALAR) < 0) {
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

template <typename UFunc>
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
  if (PyUFunc_RegisterLoopForType(ufunc, npy_cus, fn,
                                  const_cast<int*>(types.data()),
                                  nullptr) < 0) {
    return false;
  }
  return true;
}

namespace ufuncs {

struct Add {
  cus operator()(cus a, cus b) { return a + b; }
};
struct Subtract {
  cus operator()(cus a, cus b) { return a - b; }
};
struct Multiply {
  cus operator()(cus a, cus b) { return a * b; }
};
struct TrueDivide {
  cus operator()(cus a, cus b) { return a / b; }
};

std::pair<float, float> divmod(float a, float b) {
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

struct FloorDivide {
  cus operator()(cus a, cus b) {
    return cus(divmod(static_cast<float>(a), static_cast<float>(b)).first);
  }
};
struct Remainder {
  cus operator()(cus a, cus b) {
    return cus(
        divmod(static_cast<float>(a), static_cast<float>(b)).second);
  }
};
struct DivmodUFunc {
  static std::vector<int> Types() {
    return {npy_cus, npy_cus, npy_cus, npy_cus};
  }
  static void Call(char** args, npy_intp* dimensions, npy_intp* steps,
                   void* data) {
    const char* i0 = args[0];
    const char* i1 = args[1];
    char* o0 = args[2];
    char* o1 = args[3];
    for (npy_intp k = 0; k < *dimensions; k++) {
      cus x = *reinterpret_cast<const cus*>(i0);
      cus y = *reinterpret_cast<const cus*>(i1);
      float floordiv, mod;
      std::tie(floordiv, mod) =
          divmod(static_cast<float>(x), static_cast<float>(y));
      *reinterpret_cast<cus*>(o0) = cus(floordiv);
      *reinterpret_cast<cus*>(o1) = cus(mod);
      i0 += steps[0];
      i1 += steps[1];
      o0 += steps[2];
      o1 += steps[3];
    }
  }
};
struct Fmod {
  cus operator()(cus a, cus b) {
    return cus(std::fmod(static_cast<float>(a), static_cast<float>(b)));
  }
};
struct Negative {
  cus operator()(cus a) { return -a; }
};
struct Positive {
  cus operator()(cus a) { return a; }
};
struct Power {
  cus operator()(cus a, cus b) {
    return cus(std::pow(static_cast<float>(a), static_cast<float>(b)));
  }
};
struct Abs {
  cus operator()(cus a) {
    return cus(std::abs(static_cast<float>(a)));
  }
};
struct Cbrt {
  cus operator()(cus a) {
    return cus(std::cbrt(static_cast<float>(a)));
  }
};
struct Ceil {
  cus operator()(cus a) {
    return cus(std::ceil(static_cast<float>(a)));
  }
};
struct CopySign {
  cus operator()(cus a, cus b) {
    return cus(
        std::copysign(static_cast<float>(a), static_cast<float>(b)));
  }
};
struct Exp {
  cus operator()(cus a) {
    return cus(std::exp(static_cast<float>(a)));
  }
};
struct Exp2 {
  cus operator()(cus a) {
    return cus(std::exp2(static_cast<float>(a)));
  }
};
struct Expm1 {
  cus operator()(cus a) {
    return cus(std::expm1(static_cast<float>(a)));
  }
};
struct Floor {
  cus operator()(cus a) {
    return cus(std::floor(static_cast<float>(a)));
  }
};
struct Frexp {
  std::pair<cus, int> operator()(cus a) {
    int exp;
    float f = std::frexp(static_cast<float>(a), &exp);
    return {cus(f), exp};
  }
};
struct Heaviside {
  cus operator()(cus bx, cus h0) {
    float x = static_cast<float>(bx);
    if (Eigen::numext::isnan(x)) {
      return bx;
    }
    if (x < 0) {
      return cus(0.0f);
    }
    if (x > 0) {
      return cus(1.0f);
    }
    return h0;  // x == 0
  }
};
struct Conjugate {
  cus operator()(cus a) { return a; }
};
struct IsFinite {
  bool operator()(cus a) { return std::isfinite(static_cast<float>(a)); }
};
struct IsInf {
  bool operator()(cus a) { return std::isinf(static_cast<float>(a)); }
};
struct IsNan {
  bool operator()(cus a) {
    return Eigen::numext::isnan(static_cast<float>(a));
  }
};
struct Ldexp {
  cus operator()(cus a, int exp) {
    return cus(std::ldexp(static_cast<float>(a), exp));
  }
};
struct Log {
  cus operator()(cus a) {
    return cus(std::log(static_cast<float>(a)));
  }
};
struct Log2 {
  cus operator()(cus a) {
    return cus(std::log2(static_cast<float>(a)));
  }
};
struct Log10 {
  cus operator()(cus a) {
    return cus(std::log10(static_cast<float>(a)));
  }
};
struct Log1p {
  cus operator()(cus a) {
    return cus(std::log1p(static_cast<float>(a)));
  }
};
struct LogAddExp {
  cus operator()(cus bx, cus by) {
    float x = static_cast<float>(bx);
    float y = static_cast<float>(by);
    if (x == y) {
      // Handles infinities of the same sign.
      return cus(x + std::log(2.0f));
    }
    float out = std::numeric_limits<float>::quiet_NaN();
    if (x > y) {
      out = x + std::log1p(std::exp(y - x));
    } else if (x < y) {
      out = y + std::log1p(std::exp(x - y));
    }
    return cus(out);
  }
};
struct LogAddExp2 {
  cus operator()(cus bx, cus by) {
    float x = static_cast<float>(bx);
    float y = static_cast<float>(by);
    if (x == y) {
      // Handles infinities of the same sign.
      return cus(x + 1.0f);
    }
    float out = std::numeric_limits<float>::quiet_NaN();
    if (x > y) {
      out = x + std::log1p(std::exp2(y - x)) / std::log(2.0f);
    } else if (x < y) {
      out = y + std::log1p(std::exp2(x - y)) / std::log(2.0f);
    }
    return cus(out);
  }
};
struct Modf {
  std::pair<cus, cus> operator()(cus a) {
    float integral;
    float f = std::modf(static_cast<float>(a), &integral);
    return {cus(f), cus(integral)};
  }
};

struct Reciprocal {
  cus operator()(cus a) {
    return cus(1.f / static_cast<float>(a));
  }
};
struct Rint {
  cus operator()(cus a) {
    return cus(std::rint(static_cast<float>(a)));
  }
};
struct Sign {
  cus operator()(cus a) {
    float f(a);
    if (f < 0) {
      return cus(-1);
    }
    if (f > 0) {
      return cus(1);
    }
    return a;
  }
};
struct SignBit {
  bool operator()(cus a) { return std::signbit(static_cast<float>(a)); }
};
struct Sqrt {
  cus operator()(cus a) {
    return cus(std::sqrt(static_cast<float>(a)));
  }
};
struct Square {
  cus operator()(cus a) {
    float f(a);
    return cus(f * f);
  }
};
struct Trunc {
  cus operator()(cus a) {
    return cus(std::trunc(static_cast<float>(a)));
  }
};

// Trigonometric functions
struct Sin {
  cus operator()(cus a) {
    return cus(std::sin(static_cast<float>(a)));
  }
};
struct Cos {
  cus operator()(cus a) {
    return cus(std::cos(static_cast<float>(a)));
  }
};
struct Tan {
  cus operator()(cus a) {
    return cus(std::tan(static_cast<float>(a)));
  }
};
struct Arcsin {
  cus operator()(cus a) {
    return cus(std::asin(static_cast<float>(a)));
  }
};
struct Arccos {
  cus operator()(cus a) {
    return cus(std::acos(static_cast<float>(a)));
  }
};
struct Arctan {
  cus operator()(cus a) {
    return cus(std::atan(static_cast<float>(a)));
  }
};
struct Arctan2 {
  cus operator()(cus a, cus b) {
    return cus(std::atan2(static_cast<float>(a), static_cast<float>(b)));
  }
};
struct Hypot {
  cus operator()(cus a, cus b) {
    return cus(std::hypot(static_cast<float>(a), static_cast<float>(b)));
  }
};
struct Sinh {
  cus operator()(cus a) {
    return cus(std::sinh(static_cast<float>(a)));
  }
};
struct Cosh {
  cus operator()(cus a) {
    return cus(std::cosh(static_cast<float>(a)));
  }
};
struct Tanh {
  cus operator()(cus a) {
    return cus(std::tanh(static_cast<float>(a)));
  }
};
struct Arcsinh {
  cus operator()(cus a) {
    return cus(std::asinh(static_cast<float>(a)));
  }
};
struct Arccosh {
  cus operator()(cus a) {
    return cus(std::acosh(static_cast<float>(a)));
  }
};
struct Arctanh {
  cus operator()(cus a) {
    return cus(std::atanh(static_cast<float>(a)));
  }
};
struct Deg2rad {
  cus operator()(cus a) {
    static constexpr float radians_per_degree = M_PI / 180.0f;
    return cus(static_cast<float>(a) * radians_per_degree);
  }
};
struct Rad2deg {
  cus operator()(cus a) {
    static constexpr float degrees_per_radian = 180.0f / M_PI;
    return cus(static_cast<float>(a) * degrees_per_radian);
  }
};

struct Eq {
  npy_bool operator()(cus a, cus b) { return a == b; }
};
struct Ne {
  npy_bool operator()(cus a, cus b) { return a != b; }
};
struct Lt {
  npy_bool operator()(cus a, cus b) { return a < b; }
};
struct Gt {
  npy_bool operator()(cus a, cus b) { return a > b; }
};
struct Le {
  npy_bool operator()(cus a, cus b) { return a <= b; }
};
struct Ge {
  npy_bool operator()(cus a, cus b) { return a >= b; }
};
struct Maximum {
  cus operator()(cus a, cus b) {
    float fa(a), fb(b);
    return Eigen::numext::isnan(fa) || fa > fb ? a : b;
  }
};
struct Minimum {
  cus operator()(cus a, cus b) {
    float fa(a), fb(b);
    return Eigen::numext::isnan(fa) || fa < fb ? a : b;
  }
};
struct Fmax {
  cus operator()(cus a, cus b) {
    float fa(a), fb(b);
    return Eigen::numext::isnan(fb) || fa > fb ? a : b;
  }
};
struct Fmin {
  cus operator()(cus a, cus b) {
    float fa(a), fb(b);
    return Eigen::numext::isnan(fb) || fa < fb ? a : b;
  }
};

struct LogicalNot {
  npy_bool operator()(cus a) { return !a; }
};
struct LogicalAnd {
  npy_bool operator()(cus a, cus b) { return a && b; }
};
struct LogicalOr {
  npy_bool operator()(cus a, cus b) { return a || b; }
};
struct LogicalXor {
  npy_bool operator()(cus a, cus b) {
    return static_cast<bool>(a) ^ static_cast<bool>(b);
  }
};

struct NextAfter {
  cus operator()(cus from, cus to) {
    uint16_t from_as_int, to_as_int;
    const uint16_t sign_mask = 1 << 15;
    float from_as_float(from), to_as_float(to);
    memcpy(&from_as_int, &from, sizeof(cus));
    memcpy(&to_as_int, &to, sizeof(cus));
    if (Eigen::numext::isnan(from_as_float) ||
        Eigen::numext::isnan(to_as_float)) {
      return cus(std::numeric_limits<float>::quiet_NaN());
    }
    if (from_as_int == to_as_int) {
      return to;
    }
    if (from_as_float == 0) {
      if (to_as_float == 0) {
        return to;
      } else {
        // Smallest subnormal signed like `to`.
        uint16_t out_int = (to_as_int & sign_mask) | 1;
        cus out;
        memcpy(&out, &out_int, sizeof(cus));
        return out;
      }
    }
    uint16_t from_sign = from_as_int & sign_mask;
    uint16_t to_sign = to_as_int & sign_mask;
    uint16_t from_abs = from_as_int & ~sign_mask;
    uint16_t to_abs = to_as_int & ~sign_mask;
    uint16_t magnitude_adjustment =
        (from_abs > to_abs || from_sign != to_sign) ? 0xFFFF : 0x0001;
    uint16_t out_int = from_as_int + magnitude_adjustment;
    cus out;
    memcpy(&out, &out_int, sizeof(cus));
    return out;
  }
};

// TODO(phawkins): implement spacing

}  // namespace ufuncs

}  // namespace

// Initializes the module.
bool Initialize() {
  import_array1(false);
  import_umath1(false);

  Safe_PyObjectPtr numpy_str = make_safe(PyUnicode_FromString("numpy"));
  if (!numpy_str) {
    return false;
  }
  Safe_PyObjectPtr numpy = make_safe(PyImport_Import(numpy_str.get()));
  if (!numpy) {
    return false;
  }

  PyCus_Type.tp_base = &PyGenericArrType_Type;

  if (PyType_Ready(&PyCus_Type) < 0) {
    return false;
  }

  // Initializes the NumPy descriptor.
  PyArray_InitArrFuncs(&NPyCus_ArrFuncs);
  NPyCus_ArrFuncs.getitem = NPyCus_GetItem;
  NPyCus_ArrFuncs.setitem = NPyCus_SetItem;
  NPyCus_ArrFuncs.compare = NPyCus_Compare;
  NPyCus_ArrFuncs.copyswapn = NPyCus_CopySwapN;
  NPyCus_ArrFuncs.copyswap = NPyCus_CopySwap;
  NPyCus_ArrFuncs.nonzero = NPyCus_NonZero;
  NPyCus_ArrFuncs.fill = NPyCus_Fill;
  NPyCus_ArrFuncs.dotfunc = NPyCus_DotFunc;
  NPyCus_ArrFuncs.compare = NPyCus_CompareFunc;
  NPyCus_ArrFuncs.argmax = NPyCus_ArgMaxFunc;
  NPyCus_ArrFuncs.argmin = NPyCus_ArgMinFunc;

  Py_TYPE(&NPyCus_Descr) = &PyArrayDescr_Type;
  npy_cus = PyArray_RegisterDataType(&NPyCus_Descr);
  if (npy_cus < 0) {
    return false;
  }

  // Support dtype(cus)
  if (PyDict_SetItemString(PyCus_Type.tp_dict, "dtype",
                           reinterpret_cast<PyObject*>(&NPyCus_Descr)) <
      0) {
    return false;
  }

  // Register casts
  if (!RegisterCusCast<Eigen::half>(NPY_HALF, /*cast_is_safe=*/false)) {
    return false;
  }
  if (!RegisterCusCast<float>(NPY_FLOAT, /*cast_is_safe=*/true)) {
    return false;
  }
  if (!RegisterCusCast<double>(NPY_DOUBLE, /*cast_is_safe=*/true)) {
    return false;
  }
  if (!RegisterCusCast<bool>(NPY_BOOL, /*cast_is_safe=*/false)) {
    return false;
  }
  if (!RegisterCusCast<uint8>(NPY_UINT8, /*cast_is_safe=*/false)) {
    return false;
  }
  if (!RegisterCusCast<uint16>(NPY_UINT16, /*cast_is_safe=*/false)) {
    return false;
  }
  if (!RegisterCusCast<uint32>(NPY_UINT32, /*cast_is_safe=*/false)) {
    return false;
  }
  if (!RegisterCusCast<uint64>(NPY_UINT64, /*cast_is_safe=*/false)) {
    return false;
  }
  if (!RegisterCusCast<int8>(NPY_INT8, /*cast_is_safe=*/false)) {
    return false;
  }
  if (!RegisterCusCast<int16>(NPY_INT16, /*cast_is_safe=*/false)) {
    return false;
  }
  if (!RegisterCusCast<int32>(NPY_INT32, /*cast_is_safe=*/false)) {
    return false;
  }
  if (!RegisterCusCast<int64>(NPY_INT64, /*cast_is_safe=*/false)) {
    return false;
  }
  if (!RegisterCusCast<npy_longlong>(NPY_LONGLONG,
                                          /*cast_is_safe=*/false)) {
    return false;
  }
  // Following the numpy convention. imag part is dropped when converting to
  // float.
  if (!RegisterCusCast<complex64>(NPY_COMPLEX64, /*cast_is_safe=*/true)) {
    return false;
  }
  if (!RegisterCusCast<complex128>(NPY_COMPLEX128,
                                        /*cast_is_safe=*/true)) {
    return false;
  }

  bool ok =
      RegisterUFunc<BinaryUFunc<cus, cus, ufuncs::Add>>(numpy.get(),
                                                                  "add") &&
      RegisterUFunc<BinaryUFunc<cus, cus, ufuncs::Subtract>>(
          numpy.get(), "subtract") &&
      RegisterUFunc<BinaryUFunc<cus, cus, ufuncs::Multiply>>(
          numpy.get(), "multiply") &&
      RegisterUFunc<BinaryUFunc<cus, cus, ufuncs::TrueDivide>>(
          numpy.get(), "divide") &&
      RegisterUFunc<BinaryUFunc<cus, cus, ufuncs::LogAddExp>>(
          numpy.get(), "logaddexp") &&
      RegisterUFunc<BinaryUFunc<cus, cus, ufuncs::LogAddExp2>>(
          numpy.get(), "logaddexp2") &&
      RegisterUFunc<UnaryUFunc<cus, cus, ufuncs::Negative>>(
          numpy.get(), "negative") &&
      RegisterUFunc<UnaryUFunc<cus, cus, ufuncs::Positive>>(
          numpy.get(), "positive") &&
      RegisterUFunc<BinaryUFunc<cus, cus, ufuncs::TrueDivide>>(
          numpy.get(), "true_divide") &&
      RegisterUFunc<BinaryUFunc<cus, cus, ufuncs::FloorDivide>>(
          numpy.get(), "floor_divide") &&
      RegisterUFunc<BinaryUFunc<cus, cus, ufuncs::Power>>(numpy.get(),
                                                                    "power") &&
      RegisterUFunc<BinaryUFunc<cus, cus, ufuncs::Remainder>>(
          numpy.get(), "remainder") &&
      RegisterUFunc<BinaryUFunc<cus, cus, ufuncs::Remainder>>(
          numpy.get(), "mod") &&
      RegisterUFunc<BinaryUFunc<cus, cus, ufuncs::Fmod>>(numpy.get(),
                                                                   "fmod") &&
      RegisterUFunc<ufuncs::DivmodUFunc>(numpy.get(), "divmod") &&
      RegisterUFunc<UnaryUFunc<cus, cus, ufuncs::Abs>>(numpy.get(),
                                                                 "absolute") &&
      RegisterUFunc<UnaryUFunc<cus, cus, ufuncs::Abs>>(numpy.get(),
                                                                 "fabs") &&
      RegisterUFunc<UnaryUFunc<cus, cus, ufuncs::Rint>>(numpy.get(),
                                                                  "rint") &&
      RegisterUFunc<UnaryUFunc<cus, cus, ufuncs::Sign>>(numpy.get(),
                                                                  "sign") &&
      RegisterUFunc<BinaryUFunc<cus, cus, ufuncs::Heaviside>>(
          numpy.get(), "heaviside") &&
      RegisterUFunc<UnaryUFunc<cus, cus, ufuncs::Conjugate>>(
          numpy.get(), "conjugate") &&
      RegisterUFunc<UnaryUFunc<cus, cus, ufuncs::Exp>>(numpy.get(),
                                                                 "exp") &&
      RegisterUFunc<UnaryUFunc<cus, cus, ufuncs::Exp2>>(numpy.get(),
                                                                  "exp2") &&
      RegisterUFunc<UnaryUFunc<cus, cus, ufuncs::Expm1>>(numpy.get(),
                                                                   "expm1") &&
      RegisterUFunc<UnaryUFunc<cus, cus, ufuncs::Log>>(numpy.get(),
                                                                 "log") &&
      RegisterUFunc<UnaryUFunc<cus, cus, ufuncs::Log2>>(numpy.get(),
                                                                  "log2") &&
      RegisterUFunc<UnaryUFunc<cus, cus, ufuncs::Log10>>(numpy.get(),
                                                                   "log10") &&
      RegisterUFunc<UnaryUFunc<cus, cus, ufuncs::Log1p>>(numpy.get(),
                                                                   "log1p") &&
      RegisterUFunc<UnaryUFunc<cus, cus, ufuncs::Sqrt>>(numpy.get(),
                                                                  "sqrt") &&
      RegisterUFunc<UnaryUFunc<cus, cus, ufuncs::Square>>(numpy.get(),
                                                                    "square") &&
      RegisterUFunc<UnaryUFunc<cus, cus, ufuncs::Cbrt>>(numpy.get(),
                                                                  "cbrt") &&
      RegisterUFunc<UnaryUFunc<cus, cus, ufuncs::Reciprocal>>(
          numpy.get(), "reciprocal") &&

      // Trigonometric functions
      RegisterUFunc<UnaryUFunc<cus, cus, ufuncs::Sin>>(numpy.get(),
                                                                 "sin") &&
      RegisterUFunc<UnaryUFunc<cus, cus, ufuncs::Cos>>(numpy.get(),
                                                                 "cos") &&
      RegisterUFunc<UnaryUFunc<cus, cus, ufuncs::Tan>>(numpy.get(),
                                                                 "tan") &&
      RegisterUFunc<UnaryUFunc<cus, cus, ufuncs::Arcsin>>(numpy.get(),
                                                                    "arcsin") &&
      RegisterUFunc<UnaryUFunc<cus, cus, ufuncs::Arccos>>(numpy.get(),
                                                                    "arccos") &&
      RegisterUFunc<UnaryUFunc<cus, cus, ufuncs::Arctan>>(numpy.get(),
                                                                    "arctan") &&
      RegisterUFunc<BinaryUFunc<cus, cus, ufuncs::Arctan2>>(
          numpy.get(), "arctan2") &&
      RegisterUFunc<BinaryUFunc<cus, cus, ufuncs::Hypot>>(numpy.get(),
                                                                    "hypot") &&
      RegisterUFunc<UnaryUFunc<cus, cus, ufuncs::Sinh>>(numpy.get(),
                                                                  "sinh") &&
      RegisterUFunc<UnaryUFunc<cus, cus, ufuncs::Cosh>>(numpy.get(),
                                                                  "cosh") &&
      RegisterUFunc<UnaryUFunc<cus, cus, ufuncs::Tanh>>(numpy.get(),
                                                                  "tanh") &&
      RegisterUFunc<UnaryUFunc<cus, cus, ufuncs::Arcsinh>>(
          numpy.get(), "arcsinh") &&
      RegisterUFunc<UnaryUFunc<cus, cus, ufuncs::Arccosh>>(
          numpy.get(), "arccosh") &&
      RegisterUFunc<UnaryUFunc<cus, cus, ufuncs::Arctanh>>(
          numpy.get(), "arctanh") &&
      RegisterUFunc<UnaryUFunc<cus, cus, ufuncs::Deg2rad>>(
          numpy.get(), "deg2rad") &&
      RegisterUFunc<UnaryUFunc<cus, cus, ufuncs::Rad2deg>>(
          numpy.get(), "rad2deg") &&

      // Comparison functions
      RegisterUFunc<BinaryUFunc<cus, bool, ufuncs::Eq>>(numpy.get(),
                                                             "equal") &&
      RegisterUFunc<BinaryUFunc<cus, bool, ufuncs::Ne>>(numpy.get(),
                                                             "not_equal") &&
      RegisterUFunc<BinaryUFunc<cus, bool, ufuncs::Lt>>(numpy.get(),
                                                             "less") &&
      RegisterUFunc<BinaryUFunc<cus, bool, ufuncs::Gt>>(numpy.get(),
                                                             "greater") &&
      RegisterUFunc<BinaryUFunc<cus, bool, ufuncs::Le>>(numpy.get(),
                                                             "less_equal") &&
      RegisterUFunc<BinaryUFunc<cus, bool, ufuncs::Ge>>(numpy.get(),
                                                             "greater_equal") &&
      RegisterUFunc<BinaryUFunc<cus, cus, ufuncs::Maximum>>(
          numpy.get(), "maximum") &&
      RegisterUFunc<BinaryUFunc<cus, cus, ufuncs::Minimum>>(
          numpy.get(), "minimum") &&
      RegisterUFunc<BinaryUFunc<cus, cus, ufuncs::Fmax>>(numpy.get(),
                                                                   "fmax") &&
      RegisterUFunc<BinaryUFunc<cus, cus, ufuncs::Fmin>>(numpy.get(),
                                                                   "fmin") &&
      RegisterUFunc<BinaryUFunc<cus, bool, ufuncs::LogicalAnd>>(
          numpy.get(), "logical_and") &&
      RegisterUFunc<BinaryUFunc<cus, bool, ufuncs::LogicalOr>>(
          numpy.get(), "logical_or") &&
      RegisterUFunc<BinaryUFunc<cus, bool, ufuncs::LogicalXor>>(
          numpy.get(), "logical_xor") &&
      RegisterUFunc<UnaryUFunc<cus, bool, ufuncs::LogicalNot>>(
          numpy.get(), "logical_not") &&

      // Floating point functions
      RegisterUFunc<UnaryUFunc<cus, bool, ufuncs::IsFinite>>(numpy.get(),
                                                                  "isfinite") &&
      RegisterUFunc<UnaryUFunc<cus, bool, ufuncs::IsInf>>(numpy.get(),
                                                               "isinf") &&
      RegisterUFunc<UnaryUFunc<cus, bool, ufuncs::IsNan>>(numpy.get(),
                                                               "isnan") &&
      RegisterUFunc<UnaryUFunc<cus, bool, ufuncs::SignBit>>(numpy.get(),
                                                                 "signbit") &&
      RegisterUFunc<BinaryUFunc<cus, cus, ufuncs::CopySign>>(
          numpy.get(), "copysign") &&
      RegisterUFunc<UnaryUFunc2<cus, cus, cus, ufuncs::Modf>>(
          numpy.get(), "modf") &&
      RegisterUFunc<BinaryUFunc2<cus, int, cus, ufuncs::Ldexp>>(
          numpy.get(), "ldexp") &&
      RegisterUFunc<UnaryUFunc2<cus, cus, int, ufuncs::Frexp>>(
          numpy.get(), "frexp") &&
      RegisterUFunc<UnaryUFunc<cus, cus, ufuncs::Floor>>(numpy.get(),
                                                                   "floor") &&
      RegisterUFunc<UnaryUFunc<cus, cus, ufuncs::Ceil>>(numpy.get(),
                                                                  "ceil") &&
      RegisterUFunc<UnaryUFunc<cus, cus, ufuncs::Trunc>>(numpy.get(),
                                                                   "trunc") &&
      RegisterUFunc<BinaryUFunc<cus, cus, ufuncs::NextAfter>>(
          numpy.get(), "nextafter");

  return ok;
}

StatusOr<py::object> CusDtype() {
  if (npy_cus < 0) {
    // Not yet initialized. We assume the GIL protects npy_cus.
    if (!Initialize()) {
      return InternalError("Cus numpy type initialization failed.");
    }
  }
  return py::object(reinterpret_cast<PyObject*>(&PyCus_Type),
                    /*is_borrowed=*/true);
}

}  // namespace xla
