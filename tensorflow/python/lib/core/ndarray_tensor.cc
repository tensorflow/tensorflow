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

#include "tensorflow/core/lib/core/coding.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/python/lib/core/ndarray_tensor_bridge.h"

namespace tensorflow {
namespace {

Status PyArrayDescr_to_TF_DataType(PyArray_Descr* descr,
                                   TF_DataType* out_tf_datatype) {
  PyObject* key;
  PyObject* value;
  Py_ssize_t pos = 0;
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
    return Status::OK();
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
      // TODO(mrry): Support these.
      return errors::Internal("Unsupported feed type");
  }
  return Status::OK();
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
    if (!item.get()) {
      return errors::Internal("Unable to get element from the feed - no item.");
    }
    char* ptr;
    Py_ssize_t len;

    if (PyUnicode_Check(item.get())) {
#if PY_VERSION_HEX >= 0x03030000
      // Accept unicode by converting to UTF-8 bytes.
      ptr = PyUnicode_AsUTF8AndSize(item.get(), &len);
      if (!ptr) {
        return errors::Internal("Unable to get element as UTF-8.");
      }
      f(ptr, len);
#else
      PyObject* utemp = PyUnicode_AsUTF8String(item.get());
      if (!utemp || PyBytes_AsStringAndSize(utemp, &ptr, &len) == -1) {
        Py_XDECREF(utemp);
        return errors::Internal("Unable to convert element to UTF-8.");
      }
      f(ptr, len);
      Py_DECREF(utemp);
#endif
    } else {
      int success = PyBytes_AsStringAndSize(item.get(), &ptr, &len);
      if (success != 0) {
        return errors::Internal("Unable to get element as bytes.");
      }
      f(ptr, len);
    }
    PyArray_ITER_NEXT(iter.get());
  }
  return Status::OK();
}

// Encode the strings in 'array' into a contiguous buffer and return the base of
// the buffer. The caller takes ownership of the buffer.
Status EncodePyBytesArray(PyArrayObject* array, tensorflow::int64 nelems,
                          size_t* size, void** buffer) {
  // Compute bytes needed for encoding.
  *size = 0;
  TF_RETURN_IF_ERROR(PyBytesArrayMap(array, [&size](char* ptr, Py_ssize_t len) {
    *size +=
        sizeof(tensorflow::uint64) + tensorflow::core::VarintLength(len) + len;
  }));
  // Encode all strings.
  std::unique_ptr<char[]> base_ptr(new char[*size]);
  char* base = base_ptr.get();
  char* data_start = base + sizeof(tensorflow::uint64) * nelems;
  char* dst = data_start;  // Where next string is encoded.
  tensorflow::uint64* offsets = reinterpret_cast<tensorflow::uint64*>(base);

  TF_RETURN_IF_ERROR(PyBytesArrayMap(
      array, [&base, &data_start, &dst, &offsets](char* ptr, Py_ssize_t len) {
        *offsets = (dst - data_start);
        offsets++;
        dst = tensorflow::core::EncodeVarint64(dst, len);
        memcpy(dst, ptr, len);
        dst += len;
      }));
  CHECK_EQ(dst, base + *size);
  *buffer = base_ptr.release();
  return Status::OK();
}

Status CopyTF_TensorStringsToPyArray(const TF_Tensor* src, uint64 nelems,
                                     PyArrayObject* dst) {
  const void* tensor_data = TF_TensorData(src);
  const size_t tensor_size = TF_TensorByteSize(src);
  const char* limit = static_cast<const char*>(tensor_data) + tensor_size;
  DCHECK(tensor_data != nullptr);
  DCHECK_EQ(TF_STRING, TF_TensorType(src));

  const uint64* offsets = static_cast<const uint64*>(tensor_data);
  const size_t offsets_size = sizeof(uint64) * nelems;
  const char* data = static_cast<const char*>(tensor_data) + offsets_size;

  const size_t expected_tensor_size =
      (limit - static_cast<const char*>(tensor_data));
  if (expected_tensor_size - tensor_size) {
    return errors::InvalidArgument(
        "Invalid/corrupt TF_STRING tensor: expected ", expected_tensor_size,
        " bytes of encoded strings for the tensor containing ", nelems,
        " strings, but the tensor is encoded in ", tensor_size, " bytes");
  }
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(
      TF_NewStatus(), TF_DeleteStatus);
  auto iter = make_safe(PyArray_IterNew(reinterpret_cast<PyObject*>(dst)));
  for (int64 i = 0; i < nelems; ++i) {
    const char* start = data + offsets[i];
    const char* ptr = nullptr;
    size_t len = 0;

    TF_StringDecode(start, limit - start, &ptr, &len, status.get());
    if (TF_GetCode(status.get()) != TF_OK) {
      return errors::InvalidArgument(TF_Message(status.get()));
    }

    auto py_string = make_safe(PyBytes_FromStringAndSize(ptr, len));
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
  return Status::OK();
}

// Determine the dimensions of a numpy ndarray to be created to represent an
// output Tensor.
gtl::InlinedVector<npy_intp, 4> GetPyArrayDimensionsForTensor(
    const TF_Tensor* tensor, tensorflow::int64* nelems) {
  const int ndims = TF_NumDims(tensor);
  gtl::InlinedVector<npy_intp, 4> dims(ndims);
  if (TF_TensorType(tensor) == TF_RESOURCE) {
    dims[0] = TF_TensorByteSize(tensor);
    *nelems = dims[0];
  } else {
    *nelems = 1;
    for (int i = 0; i < ndims; ++i) {
      dims[i] = TF_Dim(tensor, i);
      *nelems *= dims[i];
    }
  }
  return dims;
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
    Py_CLEAR(field);
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

  return Status::OK();
}
}  // namespace

// Converts the given TF_Tensor to a numpy ndarray.
// If the returned status is OK, the caller becomes the owner of *out_array.
Status TF_TensorToPyArray(Safe_TF_TensorPtr tensor, PyObject** out_ndarray) {
  // A fetched operation will correspond to a null tensor, and a None
  // in Python.
  if (tensor == nullptr) {
    Py_INCREF(Py_None);
    *out_ndarray = Py_None;
    return Status::OK();
  }
  int64 nelems = -1;
  gtl::InlinedVector<npy_intp, 4> dims =
      GetPyArrayDimensionsForTensor(tensor.get(), &nelems);

  // If the type is neither string nor resource we can reuse the Tensor memory.
  TF_Tensor* original = tensor.get();
  TF_Tensor* moved = TF_TensorMaybeMove(tensor.release());
  if (moved != nullptr) {
    if (ArrayFromMemory(dims.size(), dims.data(), TF_TensorData(moved),
                        static_cast<DataType>(TF_TensorType(moved)),
                        [moved] { TF_DeleteTensor(moved); }, out_ndarray)
            .ok()) {
      return Status::OK();
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
    memcpy(PyArray_DATA(py_array), TF_TensorData(tensor.get()),
           PyArray_NBYTES(py_array));
  }

  // PyArray_Return turns rank 0 arrays into numpy scalars
  *out_ndarray = PyArray_Return(
      reinterpret_cast<PyArrayObject*>(safe_out_array.release()));
  return Status::OK();
}

Status PyArrayToTF_Tensor(PyObject* ndarray, Safe_TF_TensorPtr* out_tensor) {
  DCHECK(out_tensor != nullptr);

  // Make sure we dereference this array object in case of error, etc.
  Safe_PyObjectPtr array_safe(make_safe(
      PyArray_FromAny(ndarray, nullptr, 0, 0, NPY_ARRAY_CARRAY, nullptr)));
  if (!array_safe) return errors::InvalidArgument("Not a ndarray.");
  PyArrayObject* array = reinterpret_cast<PyArrayObject*>(array_safe.get());

  // Convert numpy dtype to TensorFlow dtype.
  TF_DataType dtype = TF_FLOAT;
  TF_RETURN_IF_ERROR(PyArray_TYPE_to_TF_DataType(array, &dtype));

  tensorflow::int64 nelems = 1;
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
    *out_tensor = make_safe(TF_NewTensor(dtype, {}, 0, PyArray_DATA(array),
                                         size, &DelayedNumpyDecref, array));

  } else if (dtype != TF_STRING) {
    size_t size = PyArray_NBYTES(array);
    array_safe.release();
    *out_tensor = make_safe(TF_NewTensor(dtype, dims.data(), dims.size(),
                                         PyArray_DATA(array), size,
                                         &DelayedNumpyDecref, array));
  } else {
    size_t size = 0;
    void* encoded = nullptr;
    TF_RETURN_IF_ERROR(EncodePyBytesArray(array, nelems, &size, &encoded));
    *out_tensor =
        make_safe(TF_NewTensor(dtype, dims.data(), dims.size(), encoded, size,
                               [](void* data, size_t len, void* arg) {
                                 delete[] reinterpret_cast<char*>(data);
                               },
                               nullptr));
  }
  return Status::OK();
}

Status TF_TensorToTensor(const TF_Tensor* src, Tensor* dst);
TF_Tensor* TF_TensorFromTensor(const tensorflow::Tensor& src,
                               TF_Status* status);

Status NdarrayToTensor(PyObject* obj, Tensor* ret) {
  Safe_TF_TensorPtr tf_tensor = make_safe(static_cast<TF_Tensor*>(nullptr));
  Status s = PyArrayToTF_Tensor(obj, &tf_tensor);
  if (!s.ok()) {
    return s;
  }
  return TF_TensorToTensor(tf_tensor.get(), ret);
}

Status TensorToNdarray(const Tensor& t, PyObject** ret) {
  TF_Status* status = TF_NewStatus();
  Safe_TF_TensorPtr tf_tensor = make_safe(TF_TensorFromTensor(t, status));
  Status tf_status = StatusFromTF_Status(status);
  TF_DeleteStatus(status);
  if (!tf_status.ok()) {
    return tf_status;
  }
  return TF_TensorToPyArray(std::move(tf_tensor), ret);
}

}  // namespace tensorflow
