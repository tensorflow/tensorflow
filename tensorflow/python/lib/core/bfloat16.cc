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

#include <array>

#include "tensorflow/python/lib/core/bfloat16.h"

#include "tensorflow/core/framework/numeric_types.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/python/lib/core/numpy.h"
#include "tensorflow/python/lib/core/safe_ptr.h"

namespace tensorflow {
namespace {

// Workarounds for Python 2 vs 3 API differences.
#if PY_MAJOR_VERSION < 3

PyObject* MakePyString(const string& s) {
  return PyString_FromString(s.c_str());
}

typedef long HashType;  // NOLINT

bool TfPyInt_Check(PyObject* object) { return PyInt_Check(object); }

PyObject* TfPyInt_FromLong(long x) {  // NOLINT
  return PyInt_FromLong(x);
}

long TfPyInt_AsLong(PyObject* x) {  // NOLINT
  return PyInt_AsLong(x);
}

#else  // PY_MAJOR_VERSION < 3

PyObject* MakePyString(const string& s) {
  return PyUnicode_FromString(s.c_str());
}

bool TfPyInt_Check(PyObject* object) {
  if (!PyLong_Check(object)) {
    return 0;
  }
  int overflow = 0;
  PyLong_AsLongAndOverflow(object, &overflow);
  return (overflow == 0);
}

PyObject* TfPyInt_FromLong(long x) {  // NOLINT
  return PyLong_FromLong(x);
}

long TfPyInt_AsLong(PyObject* x) {  // NOLINT
  return PyLong_AsLong(x);
}

typedef Py_hash_t HashType;

#endif  // PY_MAJOR_VERSION < 3

// Forward declaration.
extern PyTypeObject PyBfloat16_Type;

// Representation of a Python bfloat16 object.
struct PyBfloat16 {
  PyObject_HEAD;  // Python object header
  bfloat16 value;
};

// Returns true if 'object' is a PyBfloat16.
bool PyBfloat16_Check(PyObject* object) {
  return PyObject_IsInstance(object,
                             reinterpret_cast<PyObject*>(&PyBfloat16_Type));
}

// Extracts the value of a PyBfloat16 object.
bfloat16 PyBfloat16_Bfloat16(PyObject* object) {
  return reinterpret_cast<PyBfloat16*>(object)->value;
}

// Constructs a PyBfloat16 object from a bfloat16.
Safe_PyObjectPtr PyBfloat16_FromBfloat16(bfloat16 x) {
  Safe_PyObjectPtr ref =
      make_safe(PyBfloat16_Type.tp_alloc(&PyBfloat16_Type, 0));
  PyBfloat16* p = reinterpret_cast<PyBfloat16*>(ref.get());
  if (p) {
    p->value = x;
  }
  return ref;
}

// Converts a Python object to a bfloat16 value. Returns true on success,
// returns false and reports a Python error on failure.
bool AsBfloat16(PyObject* arg, bfloat16* output) {
  if (PyBfloat16_Check(arg)) {
    *output = PyBfloat16_Bfloat16(arg);
    return true;
  }
  if (PyFloat_Check(arg)) {
    double d = PyFloat_AsDouble(arg);
    if (PyErr_Occurred()) {
      return false;
    }
    // TODO(phawkins): check for overflow
    *output = bfloat16(d);
    return true;
  }
  if (TfPyInt_Check(arg)) {
    long l = TfPyInt_AsLong(arg);  // NOLINT
    if (PyErr_Occurred()) {
      return false;
    }
    // TODO(phawkins): check for overflow
    *output = bfloat16(static_cast<float>(l));
    return true;
  }
  if (PyArray_IsScalar(arg, Float)) {
    float f;
    PyArray_ScalarAsCtype(arg, &f);
    *output = bfloat16(f);
    return true;
  }
  PyErr_Format(PyExc_TypeError, "expected number, got %s",
               arg->ob_type->tp_name);
  return false;
}

// Converts a PyBfloat16 into a PyFloat.
PyObject* PyBfloat16_Float(PyObject* self) {
  bfloat16 x = PyBfloat16_Bfloat16(self);
  return PyFloat_FromDouble(static_cast<double>(x));
}

// Converts a PyBfloat16 into a PyInt.
PyObject* PyBfloat16_Int(PyObject* self) {
  bfloat16 x = PyBfloat16_Bfloat16(self);
  long y = static_cast<long>(x);  // NOLINT
  return TfPyInt_FromLong(y);
}

// Negates a PyBfloat16.
PyObject* PyBfloat16_Negative(PyObject* self) {
  bfloat16 x = PyBfloat16_Bfloat16(self);
  return PyBfloat16_FromBfloat16(-x).release();
}

// Binary arithmetic operators on PyBfloat16 values.
#define BFLOAT16_BINOP(name, op)                                  \
  PyObject* PyBfloat16_##name(PyObject* a, PyObject* b) {         \
    bfloat16 x, y;                                                \
    if (!AsBfloat16(a, &x) || !AsBfloat16(b, &y)) return nullptr; \
    bfloat16 z = x op y;                                          \
    return PyBfloat16_FromBfloat16(z).release();                  \
  }
BFLOAT16_BINOP(Add, +)
BFLOAT16_BINOP(Subtract, -)
BFLOAT16_BINOP(Multiply, *)
BFLOAT16_BINOP(Divide, /)
#undef BFLOAT16_BINOP

// Python number methods for PyBfloat16 objects.
PyNumberMethods PyBfloat16_AsNumber = {
    PyBfloat16_Add,       // nb_add
    PyBfloat16_Subtract,  // nb_subtract
    PyBfloat16_Multiply,  // nb_multiply
#if PY_MAJOR_VERSION < 3
    PyBfloat16_Divide,  // nb_divide
#endif
    nullptr,              // nb_remainder
    nullptr,              // nb_divmod
    nullptr,              // nb_power
    PyBfloat16_Negative,  // nb_negative
    nullptr,              // nb_positive
    nullptr,              // nb_absolute
    nullptr,              // nb_nonzero
    nullptr,              // nb_invert
    nullptr,              // nb_lshift
    nullptr,              // nb_rshift
    nullptr,              // nb_and
    nullptr,              // nb_xor
    nullptr,              // nb_or
#if PY_MAJOR_VERSION < 3
    nullptr,  // nb_coerce
#endif
    PyBfloat16_Int,  // nb_int
#if PY_MAJOR_VERSION < 3
    PyBfloat16_Int,  // nb_long
#else
    nullptr,  // reserved
#endif
    PyBfloat16_Float,  // nb_float
#if PY_MAJOR_VERSION < 3
    nullptr,  // nb_oct
    nullptr,  // nb_hex
#endif

    nullptr,  // nb_inplace_add
    nullptr,  // nb_inplace_subtract
    nullptr,  // nb_inplace_multiply
#if PY_MAJOR_VERSION < 3
    nullptr,  // nb_inplace_divide
#endif
    nullptr,  // nb_inplace_remainder
    nullptr,  // nb_inplace_power
    nullptr,  // nb_inplace_lshift
    nullptr,  // nb_inplace_rshift
    nullptr,  // nb_inplace_and
    nullptr,  // nb_inplace_xor
    nullptr,  // nb_inplace_or

    nullptr,            // nb_floor_divide
    PyBfloat16_Divide,  // nb_true_divide
    nullptr,            // nb_inplace_floor_divide
    nullptr,            // nb_inplace_true_divide
    nullptr,            // nb_index
};

// Constructs a new PyBfloat16.
PyObject* PyBfloat16_New(PyTypeObject* type, PyObject* args, PyObject* kwds) {
  if (kwds && PyDict_Size(kwds)) {
    PyErr_SetString(PyExc_TypeError, "constructor takes no keyword arguments");
    return nullptr;
  }
  Py_ssize_t size = PyTuple_Size(args);
  if (size != 1) {
    PyErr_SetString(PyExc_TypeError,
                    "expected number as argument to bfloat16 constructor");
    return nullptr;
  }
  PyObject* arg = PyTuple_GetItem(args, 0);

  if (PyBfloat16_Check(arg)) {
    Py_INCREF(arg);
    return arg;
  } else {
    bfloat16 value;
    if (!AsBfloat16(arg, &value)) {
      return nullptr;
    }
    return PyBfloat16_FromBfloat16(value).release();
  }
}

// Comparisons on PyBfloat16s.
PyObject* PyBfloat16_RichCompare(PyObject* a, PyObject* b, int op) {
  bfloat16 x, y;
  if (!AsBfloat16(a, &x) || !AsBfloat16(b, &y)) return nullptr;
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

// Implementation of repr() for PyBfloat16.
PyObject* PyBfloat16_Repr(PyObject* self) {
  bfloat16 x = reinterpret_cast<PyBfloat16*>(self)->value;
  string v = strings::StrCat("bfloat16(", static_cast<float>(x), ")");
  return MakePyString(v);
}

// Implementation of str() for PyBfloat16.
PyObject* PyBfloat16_Str(PyObject* self) {
  bfloat16 x = reinterpret_cast<PyBfloat16*>(self)->value;
  string v = strings::StrCat(static_cast<float>(x));
  return MakePyString(v);
}

// Hash function for PyBfloat16. We use the identity function, which is a weak
// hash function.
HashType PyBfloat16_Hash(PyObject* self) {
  bfloat16 x = reinterpret_cast<PyBfloat16*>(self)->value;
  return x.value;
}

// Python type for PyBfloat16 objects.
PyTypeObject PyBfloat16_Type = {
#if PY_MAJOR_VERSION < 3
    PyObject_HEAD_INIT(nullptr) 0,  // ob_size
#else
    PyVarObject_HEAD_INIT(nullptr, 0)
#endif
    "bfloat16",                                // tp_name
    sizeof(PyBfloat16),                        // tp_basicsize
    0,                                         // tp_itemsize
    nullptr,                                   // tp_dealloc
    0,                                         // tp_print
    nullptr,                                   // tp_getattr
    nullptr,                                   // tp_setattr
    nullptr,                                   // tp_compare / tp_reserved
    PyBfloat16_Repr,                           // tp_repr
    &PyBfloat16_AsNumber,                      // tp_as_number
    nullptr,                                   // tp_as_sequence
    nullptr,                                   // tp_as_mapping
    PyBfloat16_Hash,                           // tp_hash
    nullptr,                                   // tp_call
    PyBfloat16_Str,                            // tp_str
    nullptr,                                   // tp_getattro
    nullptr,                                   // tp_setattro
    nullptr,                                   // tp_as_buffer
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,  // tp_flags
    "bfloat16 floating-point values",          // tp_doc
    nullptr,                                   // tp_traverse
    nullptr,                                   // tp_clear
    PyBfloat16_RichCompare,                    // tp_richcompare
    0,                                         // tp_weaklistoffset
    nullptr,                                   // tp_iter
    nullptr,                                   // tp_iternext
    nullptr,                                   // tp_methods
    nullptr,                                   // tp_members
    nullptr,                                   // tp_getset
    nullptr,                                   // tp_base
    nullptr,                                   // tp_dict
    nullptr,                                   // tp_descr_get
    nullptr,                                   // tp_descr_set
    0,                                         // tp_dictoffset
    nullptr,                                   // tp_init
    nullptr,                                   // tp_alloc
    PyBfloat16_New,                            // tp_new
    nullptr,                                   // tp_free
    nullptr,                                   // tp_is_gc
    nullptr,                                   // tp_bases
    nullptr,                                   // tp_mro
    nullptr,                                   // tp_cache
    nullptr,                                   // tp_subclasses
    nullptr,                                   // tp_weaklist
    nullptr,                                   // tp_del
    0,                                         // tp_version_tag
};

// Numpy support

PyArray_ArrFuncs NPyBfloat16_ArrFuncs;

PyArray_Descr NPyBfloat16_Descr = {
    PyObject_HEAD_INIT(nullptr) & PyBfloat16_Type,  // typeobj
    // We must register bfloat16 with a kind other than "f", because numpy
    // considers two types with the same kind and size to be equal, but
    // float16 != bfloat16.
    'V',  // kind
    // TODO(phawkins): there doesn't seem to be a way of guaranteeing a type
    // character is unique.
    'E',                                                  // type
    '=',                                                  // byteorder
    NPY_NEEDS_PYAPI | NPY_USE_GETITEM | NPY_USE_SETITEM,  // hasobject
    0,                                                    // type_num
    sizeof(bfloat16),                                     // elsize
    alignof(bfloat16),                                    // alignment
    nullptr,                                              // subarray
    nullptr,                                              // fields
    nullptr,                                              // names
    &NPyBfloat16_ArrFuncs,                                // f
};

// Registered numpy type ID. Global variable populated by the registration code.
int npy_bfloat16_ = -1;

// Implementations of NumPy array methods.

PyObject* NPyBfloat16_GetItem(void* data, void* arr) {
  bfloat16 x;
  memcpy(&x, data, sizeof(bfloat16));
  return PyBfloat16_FromBfloat16(x).release();
}

int NPyBfloat16_SetItem(PyObject* item, void* data, void* arr) {
  bfloat16 x;
  if (!AsBfloat16(item, &x)) return -1;
  memcpy(data, &x, sizeof(bfloat16));
  return 0;
}

void ByteSwap16(void* value) {
  char* p = reinterpret_cast<char*>(value);
  std::swap(p[0], p[1]);
}

void NPyBfloat16_CopySwapN(void* dstv, npy_intp dstride, void* srcv,
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

void NPyBfloat16_CopySwap(void* dst, void* src, int swap, void* arr) {
  if (!src) {
    return;
  }
  memcpy(dst, src, sizeof(uint16_t));
  if (swap) {
    ByteSwap16(dst);
  }
}

npy_bool NPyBfloat16_NonZero(void* data, void* arr) {
  bfloat16 x;
  memcpy(&x, data, sizeof(x));
  return x != static_cast<bfloat16>(0);
}

int NPyBfloat16_Fill(void* buffer_raw, npy_intp length, void* ignored) {
  bfloat16* const buffer = reinterpret_cast<bfloat16*>(buffer_raw);
  const float start(buffer[0]);
  const float delta = static_cast<float>(buffer[1]) - start;
  for (npy_intp i = 2; i < length; ++i) {
    buffer[i] = static_cast<bfloat16>(start + i * delta);
  }
  return 0;
}

// NumPy casts

// Performs a NumPy array cast from type 'From' to 'To'.
template <typename From, typename To>
void NPyCast(void* from_void, void* to_void, npy_intp n, void* fromarr,
             void* toarr) {
  const From* from = reinterpret_cast<From*>(from_void);
  To* to = reinterpret_cast<To*>(to_void);
  for (npy_intp i = 0; i < n; ++i) {
    to[i] = static_cast<To>(from[i]);
  }
}

// Registers a cast between bfloat16 and type 'T'. 'numpy_type' is the NumPy
// type corresponding to 'T'. If 'cast_is_safe', registers that bfloat16 can be
// safely coerced to T.
template <typename T>
bool RegisterBfloat16Cast(int numpy_type, bool cast_is_safe) {
  if (PyArray_RegisterCastFunc(PyArray_DescrFromType(numpy_type), npy_bfloat16_,
                               NPyCast<T, bfloat16>) < 0) {
    return false;
  }
  if (PyArray_RegisterCastFunc(&NPyBfloat16_Descr, numpy_type,
                               NPyCast<bfloat16, T>) < 0) {
    return false;
  }
  if (cast_is_safe && PyArray_RegisterCanCast(&NPyBfloat16_Descr, numpy_type,
                                              NPY_NOSCALAR) < 0) {
    return false;
  }
  return true;
}

template <typename InType, typename OutType, typename Functor>
void BinaryUFunc(char** args, npy_intp* dimensions, npy_intp* steps,
                 void* data) {
  const char* i0 = args[0];
  const char* i1 = args[1];
  char* o = args[2];
  for (npy_intp k = 0; k < *dimensions; k++) {
    InType x = *reinterpret_cast<const InType*>(i0);
    InType y = *reinterpret_cast<const InType*>(i1);
    *reinterpret_cast<OutType*>(o) = Functor()(x, y);
    i0 += steps[0];
    i1 += steps[1];
    o += steps[2];
  }
}

template <typename Functor>
void CompareUFunc(char** args, npy_intp* dimensions, npy_intp* steps,
                  void* data) {
  BinaryUFunc<bfloat16, npy_bool, Functor>(args, dimensions, steps, data);
}

struct Bfloat16EqFunctor {
  npy_bool operator()(bfloat16 a, bfloat16 b) { return a == b; }
};
struct Bfloat16NeFunctor {
  npy_bool operator()(bfloat16 a, bfloat16 b) { return a != b; }
};
struct Bfloat16LtFunctor {
  npy_bool operator()(bfloat16 a, bfloat16 b) { return a < b; }
};
struct Bfloat16GtFunctor {
  npy_bool operator()(bfloat16 a, bfloat16 b) { return a > b; }
};
struct Bfloat16LeFunctor {
  npy_bool operator()(bfloat16 a, bfloat16 b) { return a <= b; }
};
struct Bfloat16GeFunctor {
  npy_bool operator()(bfloat16 a, bfloat16 b) { return a >= b; }
};

// Initializes the module.
bool Initialize() {
  // It's critical to import umath to avoid crash in open source build.
  import_umath1(false);

  Safe_PyObjectPtr numpy_str = make_safe(MakePyString("numpy"));
  if (!numpy_str) {
    return false;
  }
  Safe_PyObjectPtr numpy = make_safe(PyImport_Import(numpy_str.get()));
  if (!numpy) {
    return false;
  }

  // We hit a mysterious crash if we haven't initialized numpy before this:
  PyBfloat16_Type.tp_base = &PyGenericArrType_Type;

  if (PyType_Ready(&PyBfloat16_Type) < 0) {
    return false;
  }

  // Initializes the NumPy descriptor.
  PyArray_InitArrFuncs(&NPyBfloat16_ArrFuncs);
  NPyBfloat16_ArrFuncs.getitem = NPyBfloat16_GetItem;
  NPyBfloat16_ArrFuncs.setitem = NPyBfloat16_SetItem;
  NPyBfloat16_ArrFuncs.copyswapn = NPyBfloat16_CopySwapN;
  NPyBfloat16_ArrFuncs.copyswap = NPyBfloat16_CopySwap;
  NPyBfloat16_ArrFuncs.nonzero = NPyBfloat16_NonZero;
  NPyBfloat16_ArrFuncs.fill = NPyBfloat16_Fill;

  Py_TYPE(&NPyBfloat16_Descr) = &PyArrayDescr_Type;
  npy_bfloat16_ = PyArray_RegisterDataType(&NPyBfloat16_Descr);
  if (npy_bfloat16_ < 0) return false;

  // Support dtype(bfloat16)
  if (PyDict_SetItemString(PyBfloat16_Type.tp_dict, "dtype",
                           reinterpret_cast<PyObject*>(&NPyBfloat16_Descr)) <
      0) {
    return false;
  }

  // Register casts

  // We lie shamelessly and say that a cast from half to bfloat16 is safe.
  // Numpy frequently uses the smallest legal representation type for small
  // float constants (e.g., 1.0), which is often float16. Things break if these
  // cannot be converted transparently to bfloat16.
  if (!RegisterBfloat16Cast<Eigen::half>(NPY_HALF, /*cast_is_safe=*/true)) {
    return false;
  }

  if (!RegisterBfloat16Cast<float>(NPY_FLOAT, /*cast_is_safe=*/true)) {
    return false;
  }
  if (!RegisterBfloat16Cast<double>(NPY_DOUBLE, /*cast_is_safe=*/true)) {
    return false;
  }
  if (!RegisterBfloat16Cast<int32>(NPY_INT32, /*cast_is_safe=*/false)) {
    return false;
  }
  if (!RegisterBfloat16Cast<int64>(NPY_INT64, /*cast_is_safe=*/false)) {
    return false;
  }
  // Following the numpy convention. imag part is dropped when converting to
  // float.
  if (!RegisterBfloat16Cast<complex64>(NPY_COMPLEX64, /*cast_is_safe=*/true)) {
    return false;
  }
  if (!RegisterBfloat16Cast<complex128>(NPY_COMPLEX128,
                                        /*cast_is_safe=*/true)) {
    return false;
  }

  // Register ufuncs
  auto register_ufunc = [&](const char* name, PyUFuncGenericFunction fn,
                            const std::array<int, 3>& types) {
    Safe_PyObjectPtr ufunc_obj =
        make_safe(PyObject_GetAttrString(numpy.get(), name));
    if (!ufunc_obj) {
      return false;
    }
    PyUFuncObject* ufunc = reinterpret_cast<PyUFuncObject*>(ufunc_obj.get());
    if (types.size() != ufunc->nargs) {
      PyErr_Format(PyExc_AssertionError,
                   "ufunc %s takes %d arguments, loop takes %lu", name,
                   ufunc->nargs, types.size());
      return false;
    }
    if (PyUFunc_RegisterLoopForType(ufunc, npy_bfloat16_, fn,
                                    const_cast<int*>(types.data()),
                                    nullptr) < 0) {
      return false;
    }
    return true;
  };

  // Comparisons
  const std::array<int, 3> compare_types = {
      {npy_bfloat16_, npy_bfloat16_, NPY_BOOL}};

  if (!register_ufunc("equal", CompareUFunc<Bfloat16EqFunctor>,
                      compare_types)) {
    return false;
  }
  if (!register_ufunc("not_equal", CompareUFunc<Bfloat16NeFunctor>,
                      compare_types)) {
    return false;
  }
  if (!register_ufunc("less", CompareUFunc<Bfloat16LtFunctor>, compare_types)) {
    return false;
  }
  if (!register_ufunc("greater", CompareUFunc<Bfloat16GtFunctor>,
                      compare_types)) {
    return false;
  }
  if (!register_ufunc("less_equal", CompareUFunc<Bfloat16LeFunctor>,
                      compare_types)) {
    return false;
  }
  if (!register_ufunc("greater_equal", CompareUFunc<Bfloat16GeFunctor>,
                      compare_types)) {
    return false;
  }
  return true;
}

}  // namespace

void RegisterNumpyBfloat16() {
  if (npy_bfloat16_ >= 0) {
    // Already initialized.
    return;
  }
  if (!Initialize()) {
    if (!PyErr_Occurred()) {
      PyErr_SetString(PyExc_RuntimeError, "cannot load bfloat16 module.");
    }
    PyErr_Print();
  }
}

PyObject* Bfloat16PyType() {
  CHECK(PyBfloat16_Type.tp_base != nullptr);
  Py_INCREF(&PyBfloat16_Type);
  return reinterpret_cast<PyObject*>(&PyBfloat16_Type);
}

int Bfloat16NumpyType() {
  CHECK_GE(npy_bfloat16_, 0);
  return npy_bfloat16_;
}

}  // namespace tensorflow
