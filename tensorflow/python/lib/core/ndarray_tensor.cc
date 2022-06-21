/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/python/lib/core/ndarray_tensor.h"

#include <cstring>
#include <optional>

#include "tensorflow/c/eager/tfe_context_internal.h"
#include "tensorflow/c/tf_tensor_internal.h"
#include "tensorflow/core/lib/core/coding.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/python/lib/core/bfloat16.h"
#include "tensorflow/python/lib/core/ndarray_tensor_bridge.h"
#include "tensorflow/python/lib/core/numpy.h"

namespace tensorflow {
namespace {

char const* numpy_type_name(int numpy_type) {
  switch (numpy_type) {
#define TYPE_CASE(s) \
  case s:            \
    return #s

    TYPE_CASE(NPY_BOOL);
    TYPE_CASE(NPY_BYTE);
    TYPE_CASE(NPY_UBYTE);
    TYPE_CASE(NPY_SHORT);
    TYPE_CASE(NPY_USHORT);
    TYPE_CASE(NPY_INT);
    TYPE_CASE(NPY_UINT);
    TYPE_CASE(NPY_LONG);
    TYPE_CASE(NPY_ULONG);
    TYPE_CASE(NPY_LONGLONG);
    TYPE_CASE(NPY_ULONGLONG);
    TYPE_CASE(NPY_FLOAT);
    TYPE_CASE(NPY_DOUBLE);
    TYPE_CASE(NPY_LONGDOUBLE);
    TYPE_CASE(NPY_CFLOAT);
    TYPE_CASE(NPY_CDOUBLE);
    TYPE_CASE(NPY_CLONGDOUBLE);
    TYPE_CASE(NPY_OBJECT);
    TYPE_CASE(NPY_STRING);
    TYPE_CASE(NPY_UNICODE);
    TYPE_CASE(NPY_VOID);
    TYPE_CASE(NPY_DATETIME);
    TYPE_CASE(NPY_TIMEDELTA);
    TYPE_CASE(NPY_HALF);
    TYPE_CASE(NPY_NTYPES);
    TYPE_CASE(NPY_NOTYPE);
    TYPE_CASE(NPY_CHAR);
    TYPE_CASE(NPY_USERDEF);
    default:
      return "not a numpy type";
  }
}

Status PyArrayDescr_to_TF_DataType(PyArray_Descr* descr,
                                   TF_DataType* out_tf_datatype) {
  PyObject* key;
  PyObject* value;
  Py_ssize_t pos = 0;

  // Return an error if the fields attribute is null.
  // Occurs with an improper conversion attempt to resource.
  if (descr->fields == nullptr) {
    return errors::Internal("Unexpected numpy data type");
  }

  if (PyDict_Next(descr->fields, &pos, &key, &value)) {
    // In Python 3, the keys of numpy custom struct types are unicode, unlike
    // Python 2, where the keys are bytes.
    const char* key_string =
        PyBytes_Check(key) ? PyBytes_AsString(key)
                           : PyBytes_AsString(PyUnicode_AsASCIIString(key));
    if (!key_string) {
      return errors::Internal("Corrupt numpy type descriptor");
    }
    tensorflow::string key = key_string;
    // The typenames here should match the field names in the custom struct
    // types constructed in test_util.py.
    // TODO(mrry,keveman): Investigate Numpy type registration to replace this
    // hard-coding of names.
    if (key == "quint8") {
      *out_tf_datatype = TF_QUINT8;
    } else if (key == "qint8") {
      *out_tf_datatype = TF_QINT8;
    } else if (key == "qint16") {
      *out_tf_datatype = TF_QINT16;
    } else if (key == "quint16") {
      *out_tf_datatype = TF_QUINT16;
    } else if (key == "qint32") {
      *out_tf_datatype = TF_QINT32;
    } else if (key == "resource") {
      *out_tf_datatype = TF_RESOURCE;
    } else {
      return errors::Internal("Unsupported numpy data type");
    }
    return OkStatus();
  }
  return errors::Internal("Unsupported numpy data type");
}

Status PyArray_TYPE_to_TF_DataType(PyArrayObject* array,
                                   TF_DataType* out_tf_datatype) {
  int pyarray_type = PyArray_TYPE(array);
  PyArray_Descr* descr = PyArray_DESCR(array);
  switch (pyarray_type) {
    case NPY_FLOAT16:
      *out_tf_datatype = TF_HALF;
      break;
    case NPY_FLOAT32:
      *out_tf_datatype = TF_FLOAT;
      break;
    case NPY_FLOAT64:
      *out_tf_datatype = TF_DOUBLE;
      break;
    case NPY_INT32:
      *out_tf_datatype = TF_INT32;
      break;
    case NPY_UINT8:
      *out_tf_datatype = TF_UINT8;
      break;
    case NPY_UINT16:
      *out_tf_datatype = TF_UINT16;
      break;
    case NPY_UINT32:
      *out_tf_datatype = TF_UINT32;
      break;
    case NPY_UINT64:
      *out_tf_datatype = TF_UINT64;
      break;
    case NPY_INT8:
      *out_tf_datatype = TF_INT8;
      break;
    case NPY_INT16:
      *out_tf_datatype = TF_INT16;
      break;
    case NPY_INT64:
      *out_tf_datatype = TF_INT64;
      break;
    case NPY_BOOL:
      *out_tf_datatype = TF_BOOL;
      break;
    case NPY_COMPLEX64:
      *out_tf_datatype = TF_COMPLEX64;
      break;
    case NPY_COMPLEX128:
      *out_tf_datatype = TF_COMPLEX128;
      break;
    case NPY_OBJECT:
    case NPY_STRING:
    case NPY_UNICODE:
      *out_tf_datatype = TF_STRING;
      break;
    case NPY_VOID:
      // Quantized types are currently represented as custom struct types.
      // PyArray_TYPE returns NPY_VOID for structs, and we should look into
      // descr to derive the actual type.
      // Direct feeds of certain types of ResourceHandles are represented as a
      // custom struct type.
      return PyArrayDescr_to_TF_DataType(descr, out_tf_datatype);
    default:
      if (pyarray_type == Bfloat16NumpyType()) {
        *out_tf_datatype = TF_BFLOAT16;
        break;
      } else if (pyarray_type == NPY_ULONGLONG) {
        // NPY_ULONGLONG is equivalent to NPY_UINT64, while their enum values
        // might be different on certain platforms.
        *out_tf_datatype = TF_UINT64;
        break;
      } else if (pyarray_type == NPY_LONGLONG) {
        // NPY_LONGLONG is equivalent to NPY_INT64, while their enum values
        // might be different on certain platforms.
        *out_tf_datatype = TF_INT64;
        break;
      } else if (pyarray_type == NPY_INT) {
        // NPY_INT is equivalent to NPY_INT32, while their enum values might be
        // different on certain platforms.
        *out_tf_datatype = TF_INT32;
        break;
      } else if (pyarray_type == NPY_UINT) {
        // NPY_UINT is equivalent to NPY_UINT32, while their enum values might
        // be different on certain platforms.
        *out_tf_datatype = TF_UINT32;
        break;
      }
      return errors::Internal("Unsupported numpy type: ",
                              numpy_type_name(pyarray_type));
  }
  return OkStatus();
}

Status PyObjectToString(PyObject* obj, const char** ptr, Py_ssize_t* len,
                        PyObject** ptr_owner) {
  *ptr_owner = nullptr;
  if (PyBytes_Check(obj)) {
    char* buf;
    if (PyBytes_AsStringAndSize(obj, &buf, len) != 0) {
      return errors::Internal("Unable to get element as bytes.");
    }
    *ptr = buf;
    return OkStatus();
  } else if (PyUnicode_Check(obj)) {
#if (PY_MAJOR_VERSION > 3 || (PY_MAJOR_VERSION == 3 && PY_MINOR_VERSION >= 3))
    *ptr = PyUnicode_AsUTF8AndSize(obj, len);
    if (*ptr != nullptr) return OkStatus();
#else
    PyObject* utemp = PyUnicode_AsUTF8String(obj);
    char* buf;
    if (utemp != nullptr && PyBytes_AsStringAndSize(utemp, &buf, len) != -1) {
      *ptr = buf;
      *ptr_owner = utemp;
      return Status::OK();
    }
    Py_XDECREF(utemp);
#endif
    return errors::Internal("Unable to convert element to UTF-8");
  } else {
    return errors::Internal("Unsupported object type ", obj->ob_type->tp_name);
  }
}

// Iterate over the string array 'array', extract the ptr and len of each string
// element and call f(ptr, len).
template <typename F>
Status PyBytesArrayMap(PyArrayObject* array, F f) {
  Safe_PyObjectPtr iter = tensorflow::make_safe(
      PyArray_IterNew(reinterpret_cast<PyObject*>(array)));
  while (PyArray_ITER_NOTDONE(iter.get())) {
    auto item = tensorflow::make_safe(PyArray_GETITEM(
        array, static_cast<char*>(PyArray_ITER_DATA(iter.get()))));
    if (!item) {
      return errors::Internal("Unable to get element from the feed - no item.");
    }
    Py_ssize_t len;
    const char* ptr;
    PyObject* ptr_owner = nullptr;
    TF_RETURN_IF_ERROR(PyObjectToString(item.get(), &ptr, &len, &ptr_owner));
    f(ptr, len);
    Py_XDECREF(ptr_owner);
    PyArray_ITER_NEXT(iter.get());
  }
  return OkStatus();
}

// Encode the strings in 'array' into a contiguous buffer and return the base of
// the buffer. The caller takes ownership of the buffer.
Status EncodePyBytesArray(PyArrayObject* array, int64_t nelems, size_t* size,
                          void** buffer) {
  // Encode all strings.
  *size = nelems * sizeof(tensorflow::tstring);
  std::unique_ptr<tensorflow::tstring[]> base_ptr(
      new tensorflow::tstring[nelems]);
  tensorflow::tstring* dst = base_ptr.get();

  TF_RETURN_IF_ERROR(
      PyBytesArrayMap(array, [&dst](const char* ptr, Py_ssize_t len) {
        dst->assign(ptr, len);
        dst++;
      }));
  *buffer = base_ptr.release();
  return OkStatus();
}

Status CopyTF_TensorStringsToPyArray(const TF_Tensor* src, uint64 nelems,
                                     PyArrayObject* dst) {
  const void* tensor_data = TF_TensorData(src);
  DCHECK(tensor_data != nullptr);
  DCHECK_EQ(TF_STRING, TF_TensorType(src));

  const tstring* tstr = static_cast<const tstring*>(tensor_data);

  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(
      TF_NewStatus(), TF_DeleteStatus);
  auto iter = make_safe(PyArray_IterNew(reinterpret_cast<PyObject*>(dst)));
  for (int64_t i = 0; i < static_cast<int64_t>(nelems); ++i) {
    const tstring& tstr_i = tstr[i];
    auto py_string =
        make_safe(PyBytes_FromStringAndSize(tstr_i.data(), tstr_i.size()));
    if (py_string == nullptr) {
      return errors::Internal(
          "failed to create a python byte array when converting element #", i,
          " of a TF_STRING tensor to a numpy ndarray");
    }

    if (PyArray_SETITEM(dst, static_cast<char*>(PyArray_ITER_DATA(iter.get())),
                        py_string.get()) != 0) {
      return errors::Internal("Error settings element #", i,
                              " in the numpy ndarray");
    }
    PyArray_ITER_NEXT(iter.get());
  }
  return OkStatus();
}

// Determine the dimensions of a numpy ndarray to be created to represent an
// output Tensor.
Status GetPyArrayDimensionsForTensor(const TF_Tensor* tensor,
                                     gtl::InlinedVector<npy_intp, 4>* dims,
                                     int64_t* nelems) {
  dims->clear();
  const int ndims = TF_NumDims(tensor);
  if (TF_TensorType(tensor) == TF_RESOURCE) {
    if (ndims != 0) {
      return errors::InvalidArgument(
          "Fetching of non-scalar resource tensors is not supported.");
    }
    dims->push_back(TF_TensorByteSize(tensor));
    *nelems = dims->back();
  } else {
    *nelems = 1;
    for (int i = 0; i < ndims; ++i) {
      dims->push_back(TF_Dim(tensor, i));
      *nelems *= dims->back();
    }
  }
  return OkStatus();
}

// Determine the type description (PyArray_Descr) of a numpy ndarray to be
// created to represent an output Tensor.
Status GetPyArrayDescrForTensor(const TF_Tensor* tensor,
                                PyArray_Descr** descr) {
  if (TF_TensorType(tensor) == TF_RESOURCE) {
    PyObject* field = PyTuple_New(3);
#if PY_MAJOR_VERSION < 3
    PyTuple_SetItem(field, 0, PyBytes_FromString("resource"));
#else
    PyTuple_SetItem(field, 0, PyUnicode_FromString("resource"));
#endif
    PyTuple_SetItem(field, 1, PyArray_TypeObjectFromType(NPY_UBYTE));
    PyTuple_SetItem(field, 2, PyLong_FromLong(1));
    PyObject* fields = PyList_New(1);
    PyList_SetItem(fields, 0, field);
    int convert_result = PyArray_DescrConverter(fields, descr);
    Py_CLEAR(fields);
    if (convert_result != 1) {
      return errors::Internal("Failed to create numpy array description for ",
                              "TF_RESOURCE-type tensor");
    }
  } else {
    int type_num = -1;
    TF_RETURN_IF_ERROR(
        TF_DataType_to_PyArray_TYPE(TF_TensorType(tensor), &type_num));
    *descr = PyArray_DescrFromType(type_num);
  }

  return OkStatus();
}

inline void FastMemcpy(void* dst, const void* src, size_t size) {
  // clang-format off
  switch (size) {
    // Most compilers will generate inline code for fixed sizes,
    // which is significantly faster for small copies.
    case  1: memcpy(dst, src, 1); break;
    case  2: memcpy(dst, src, 2); break;
    case  3: memcpy(dst, src, 3); break;
    case  4: memcpy(dst, src, 4); break;
    case  5: memcpy(dst, src, 5); break;
    case  6: memcpy(dst, src, 6); break;
    case  7: memcpy(dst, src, 7); break;
    case  8: memcpy(dst, src, 8); break;
    case  9: memcpy(dst, src, 9); break;
    case 10: memcpy(dst, src, 10); break;
    case 11: memcpy(dst, src, 11); break;
    case 12: memcpy(dst, src, 12); break;
    case 13: memcpy(dst, src, 13); break;
    case 14: memcpy(dst, src, 14); break;
    case 15: memcpy(dst, src, 15); break;
    case 16: memcpy(dst, src, 16); break;
#if defined(PLATFORM_GOOGLE) || defined(PLATFORM_POSIX) && \
    !defined(IS_MOBILE_PLATFORM)
    // On Linux, memmove appears to be faster than memcpy for
    // large sizes, strangely enough.
    default: memmove(dst, src, size); break;
#else
    default: memcpy(dst, src, size); break;
#endif
  }
  // clang-format on
}

}  // namespace

// TODO(slebedev): revise TF_TensorToPyArray usages and switch to the
// aliased version where appropriate.
Status TF_TensorToMaybeAliasedPyArray(Safe_TF_TensorPtr tensor,
                                      PyObject** out_ndarray) {
  auto dtype = TF_TensorType(tensor.get());
  if (dtype == TF_STRING || dtype == TF_RESOURCE) {
    return TF_TensorToPyArray(std::move(tensor), out_ndarray);
  }

  TF_Tensor* moved = tensor.release();
  int64_t nelems = -1;
  gtl::InlinedVector<npy_intp, 4> dims;
  TF_RETURN_IF_ERROR(GetPyArrayDimensionsForTensor(moved, &dims, &nelems));
  return ArrayFromMemory(
      dims.size(), dims.data(), TF_TensorData(moved),
      static_cast<DataType>(dtype), [moved] { TF_DeleteTensor(moved); },
      out_ndarray);
}

// Converts the given TF_Tensor to a numpy ndarray.
// If the returned status is OK, the caller becomes the owner of *out_array.
Status TF_TensorToPyArray(Safe_TF_TensorPtr tensor, PyObject** out_ndarray) {
  // A fetched operation will correspond to a null tensor, and a None
  // in Python.
  if (tensor == nullptr) {
    Py_INCREF(Py_None);
    *out_ndarray = Py_None;
    return OkStatus();
  }
  int64_t nelems = -1;
  gtl::InlinedVector<npy_intp, 4> dims;
  TF_RETURN_IF_ERROR(
      GetPyArrayDimensionsForTensor(tensor.get(), &dims, &nelems));

  // If the type is neither string nor resource we can reuse the Tensor memory.
  TF_Tensor* original = tensor.get();
  TF_Tensor* moved = TF_TensorMaybeMove(tensor.release());
  if (moved != nullptr) {
    if (ArrayFromMemory(
            dims.size(), dims.data(), TF_TensorData(moved),
            static_cast<DataType>(TF_TensorType(moved)),
            [moved] { TF_DeleteTensor(moved); }, out_ndarray)
            .ok()) {
      return OkStatus();
    }
  }
  tensor.reset(original);

  // Copy the TF_TensorData into a newly-created ndarray and return it.
  PyArray_Descr* descr = nullptr;
  TF_RETURN_IF_ERROR(GetPyArrayDescrForTensor(tensor.get(), &descr));
  Safe_PyObjectPtr safe_out_array =
      tensorflow::make_safe(PyArray_Empty(dims.size(), dims.data(), descr, 0));
  if (!safe_out_array) {
    return errors::Internal("Could not allocate ndarray");
  }
  PyArrayObject* py_array =
      reinterpret_cast<PyArrayObject*>(safe_out_array.get());
  if (TF_TensorType(tensor.get()) == TF_STRING) {
    Status s = CopyTF_TensorStringsToPyArray(tensor.get(), nelems, py_array);
    if (!s.ok()) {
      return s;
    }
  } else if (static_cast<size_t>(PyArray_NBYTES(py_array)) !=
             TF_TensorByteSize(tensor.get())) {
    return errors::Internal("ndarray was ", PyArray_NBYTES(py_array),
                            " bytes but TF_Tensor was ",
                            TF_TensorByteSize(tensor.get()), " bytes");
  } else {
    FastMemcpy(PyArray_DATA(py_array), TF_TensorData(tensor.get()),
               PyArray_NBYTES(py_array));
  }

  *out_ndarray = safe_out_array.release();
  return OkStatus();
}

Status NdarrayToTensor(TFE_Context* ctx, PyObject* ndarray,
                       Safe_TF_TensorPtr* ret) {
  DCHECK(ret != nullptr);

  // Make sure we dereference this array object in case of error, etc.
  Safe_PyObjectPtr array_safe(make_safe(
      PyArray_FromAny(ndarray, nullptr, 0, 0, NPY_ARRAY_CARRAY_RO, nullptr)));
  if (!array_safe) return errors::InvalidArgument("Not a ndarray.");
  PyArrayObject* array = reinterpret_cast<PyArrayObject*>(array_safe.get());

  // Convert numpy dtype to TensorFlow dtype.
  TF_DataType dtype = TF_FLOAT;
  TF_RETURN_IF_ERROR(PyArray_TYPE_to_TF_DataType(array, &dtype));

  int64_t nelems = 1;
  gtl::InlinedVector<int64_t, 4> dims;
  for (int i = 0; i < PyArray_NDIM(array); ++i) {
    dims.push_back(PyArray_SHAPE(array)[i]);
    nelems *= dims[i];
  }

  // Create a TF_Tensor based on the fed data. In the case of non-string data
  // type, this steals a reference to array, which will be relinquished when
  // the underlying buffer is deallocated. For string, a new temporary buffer
  // is allocated into which the strings are encoded.
  if (dtype == TF_RESOURCE) {
    size_t size = PyArray_NBYTES(array);
    array_safe.release();

    if (ctx) {
      *ret = make_safe(new TF_Tensor{tensorflow::unwrap(ctx)->CreateTensor(
          static_cast<tensorflow::DataType>(dtype), {}, 0, PyArray_DATA(array),
          size, &DelayedNumpyDecref, array)});
    } else {
      *ret = make_safe(TF_NewTensor(dtype, {}, 0, PyArray_DATA(array), size,
                                    &DelayedNumpyDecref, array));
    }

  } else if (dtype != TF_STRING) {
    size_t size = PyArray_NBYTES(array);
    array_safe.release();
    if (ctx) {
      *ret = make_safe(new TF_Tensor{tensorflow::unwrap(ctx)->CreateTensor(
          static_cast<tensorflow::DataType>(dtype), dims.data(), dims.size(),
          PyArray_DATA(array), size, &DelayedNumpyDecref, array)});
    } else {
      *ret = make_safe(TF_NewTensor(dtype, dims.data(), dims.size(),
                                    PyArray_DATA(array), size,
                                    &DelayedNumpyDecref, array));
    }

  } else {
    size_t size = 0;
    void* encoded = nullptr;
    TF_RETURN_IF_ERROR(EncodePyBytesArray(array, nelems, &size, &encoded));
    if (ctx) {
      *ret = make_safe(new TF_Tensor{tensorflow::unwrap(ctx)->CreateTensor(
          static_cast<tensorflow::DataType>(dtype), dims.data(), dims.size(),
          encoded, size,
          [](void* data, size_t len, void* arg) {
            delete[] reinterpret_cast<tensorflow::tstring*>(data);
          },
          nullptr)});
    } else {
      *ret = make_safe(TF_NewTensor(
          dtype, dims.data(), dims.size(), encoded, size,
          [](void* data, size_t len, void* arg) {
            delete[] reinterpret_cast<tensorflow::tstring*>(data);
          },
          nullptr));
    }
  }

  return OkStatus();
}

Status TF_TensorToTensor(const TF_Tensor* src, Tensor* dst);
TF_Tensor* TF_TensorFromTensor(const tensorflow::Tensor& src, Status* status);

Status NdarrayToTensor(PyObject* obj, Tensor* ret) {
  Safe_TF_TensorPtr tf_tensor = make_safe(static_cast<TF_Tensor*>(nullptr));
  Status s = NdarrayToTensor(nullptr /*ctx*/, obj, &tf_tensor);
  if (!s.ok()) {
    return s;
  }
  return TF_TensorToTensor(tf_tensor.get(), ret);
}

Status TensorToNdarray(const Tensor& t, PyObject** ret) {
  Status status;
  Safe_TF_TensorPtr tf_tensor = make_safe(TF_TensorFromTensor(t, &status));
  if (!status.ok()) {
    return status;
  }
  return TF_TensorToMaybeAliasedPyArray(std::move(tf_tensor), ret);
}

}  // namespace tensorflow
