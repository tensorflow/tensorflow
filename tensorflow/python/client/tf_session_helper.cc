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

#include "tensorflow/python/client/tf_session_helper.h"

#include <cstring>

#include "tensorflow/c/c_api.h"
#include "tensorflow/c/tf_status_helper.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/log_memory.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/core/coding.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/equal_graph_def.h"
#include "tensorflow/python/lib/core/ndarray_tensor_bridge.h"

namespace tensorflow {

namespace {

static const char* kFeedDictErrorMsg =
    "feed_dict must be a dictionary mapping strings to NumPy arrays.";

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
        return errors::Internal(
            "Unable to get element from the feed as UTF-8.");
      }
      f(ptr, len);
#else
      PyObject* utemp = PyUnicode_AsUTF8String(item.get());
      if (!utemp || PyBytes_AsStringAndSize(utemp, &ptr, &len) == -1) {
        Py_XDECREF(utemp);
        return errors::Internal(
            "Unable to convert element from the feed to UTF-8.");
      }
      f(ptr, len);
      Py_DECREF(utemp);
#endif
    } else {
      int success = PyBytes_AsStringAndSize(item.get(), &ptr, &len);
      if (success != 0) {
        return errors::Internal(
            "Unable to get element from the feed as bytes.");
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

// Determine the pointer and offset of the string at offset 'i' in the string
// tensor 'src', whose total length is 'num_elements'.
static Status TF_StringTensor_GetPtrAndLen(const TF_Tensor* src,
                                           tensorflow::int64 num_elements,
                                           tensorflow::int64 i,
                                           const char** ptr,
                                           tensorflow::uint64* len) {
  const char* input = reinterpret_cast<const char*>(TF_TensorData(src));
  const size_t src_size = TF_TensorByteSize(src);
  const char* data_start = input + sizeof(tensorflow::uint64) * num_elements;
  const char* limit = input + src_size;
  tensorflow::uint64 offset =
      reinterpret_cast<const tensorflow::uint64*>(input)[i];
  const char* p =
      tensorflow::core::GetVarint64Ptr(data_start + offset, limit, len);
  if (static_cast<int64>(offset) >= (limit - data_start) || !p ||
      static_cast<int64>(*len) > (limit - p)) {
    return errors::InvalidArgument("Malformed TF_STRING tensor; element ", i,
                                   " out of range");
  }
  *ptr = p;
  return Status::OK();
}

// Copy the string at offset 'i' in the (linearized) string tensor 'tensor' into
// 'pyarray' at offset pointed by the 'i_ptr' iterator.
static Status CopyStringToPyArrayElement(PyArrayObject* pyarray, void* i_ptr,
                                         TF_Tensor* tensor,
                                         tensorflow::int64 num_elements,
                                         tensorflow::int64 i) {
  const char* ptr = nullptr;
  tensorflow::uint64 len = 0;
  TF_RETURN_IF_ERROR(
      TF_StringTensor_GetPtrAndLen(tensor, num_elements, i, &ptr, &len));
  auto py_string = tensorflow::make_safe(PyBytes_FromStringAndSize(ptr, len));
  int success = PyArray_SETITEM(
      pyarray, static_cast<char*>(PyArray_ITER_DATA(i_ptr)), py_string.get());
  if (success != 0) {
    return errors::Internal("Error setting element ", i);
  }
  return Status::OK();
}

// Determine the dimensions of a numpy ndarray to be created to represent an
// output Tensor.
gtl::InlinedVector<npy_intp, 4> GetPyArrayDimensionsForTensor(
    const TF_Tensor* tensor, tensorflow::int64* nelems) {
  if (TF_TensorType(tensor) == TF_RESOURCE) {
    gtl::InlinedVector<npy_intp, 4> dims(1);
    ResourceHandle* resource_handle =
        reinterpret_cast<ResourceHandle*>(TF_TensorData(tensor));
    dims[0] = resource_handle->SerializeAsString().size();
    *nelems = dims[0];

    return dims;
  } else {
    const int ndims = TF_NumDims(tensor);
    gtl::InlinedVector<npy_intp, 4> dims(ndims);
    *nelems = 1;
    for (int i = 0; i < ndims; ++i) {
      dims[i] = TF_Dim(tensor, i);
      *nelems *= dims[i];
    }

    return dims;
  }
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

// Converts the given TF_Tensor to a numpy ndarray.
// If the returned status is OK, the caller becomes the owner of *out_array.
Status TFTensorToPyArray(Safe_TF_TensorPtr tensor, PyObject** out_ndarray) {
  // A fetched operation will correspond to a null tensor, and a None
  // in Python.
  if (tensor == nullptr) {
    Py_INCREF(Py_None);
    *out_ndarray = Py_None;
    return Status::OK();
  }

  tensorflow::int64 nelems = -1;
  gtl::InlinedVector<npy_intp, 4> dims =
      GetPyArrayDimensionsForTensor(tensor.get(), &nelems);

  // Convert TensorFlow dtype to numpy type descriptor.
  PyArray_Descr* descr = nullptr;
  TF_RETURN_IF_ERROR(GetPyArrayDescrForTensor(tensor.get(), &descr));

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
  Safe_PyObjectPtr safe_out_array =
      tensorflow::make_safe(PyArray_Empty(dims.size(), dims.data(), descr, 0));
  if (!safe_out_array) {
    return errors::Internal("Could not allocate ndarray");
  }
  PyArrayObject* py_array =
      reinterpret_cast<PyArrayObject*>(safe_out_array.get());
  if (PyArray_NBYTES(py_array) !=
      static_cast<int64>(TF_TensorByteSize(tensor.get()))) {
    if (TF_TensorType(tensor.get()) == TF_STRING) {
      // Copy element by element.
      auto iter = tensorflow::make_safe(PyArray_IterNew(safe_out_array.get()));
      for (tensorflow::int64 i = 0; i < nelems; ++i) {
        auto s = CopyStringToPyArrayElement(py_array, iter.get(), tensor.get(),
                                            nelems, i);
        if (!s.ok()) {
          return s;
        }
        PyArray_ITER_NEXT(iter.get());
      }
    } else if (TF_TensorType(tensor.get()) == TF_RESOURCE) {
      ResourceHandle* resource_handle =
          reinterpret_cast<ResourceHandle*>(TF_TensorData(tensor.get()));
      memcpy(PyArray_DATA(py_array),
             resource_handle->SerializeAsString().c_str(),
             PyArray_NBYTES(py_array));
    } else {
      return errors::Internal("ndarray was ", PyArray_NBYTES(py_array),
                              " bytes but TF_Tensor was ",
                              TF_TensorByteSize(tensor.get()), " bytes");
    }
  } else {
    memcpy(PyArray_DATA(py_array), TF_TensorData(tensor.get()),
           PyArray_NBYTES(py_array));
  }

  // PyArray_Return turns rank 0 arrays into numpy scalars
  *out_ndarray = PyArray_Return(
      reinterpret_cast<PyArrayObject*>(safe_out_array.release()));
  return Status::OK();
}

// Converts the given numpy ndarray to a (safe) TF_Tensor. The returned
// TF_Tensor in `out_tensor` will have its own Python reference to `ndarray`s
// data. After `out_tensor` is destroyed, this reference must (eventually) be
// decremented via ClearDecrefCache().
//
// If `ndarray` contains a resource handle, `*resource_handle` will be set to
// the deserialized handle. Otherwise it is set to nullptr. Caller becomes owner
// of `*resource_handle` if it's set, and it must outlive the returned
// `out_tensor`.
//
// `resource_handle` and `out_tensor` must be non-null. Caller retains ownership
// of `ndarray`.
Status PyArrayToTFTensor(PyObject* ndarray, Safe_TF_TensorPtr* out_tensor,
                         ResourceHandle** resource_handle) {
  DCHECK(out_tensor != nullptr);
  DCHECK(resource_handle != nullptr);
  *resource_handle = nullptr;

  // Make sure we dereference this array object in case of error, etc.
  Safe_PyObjectPtr array_safe(make_safe(
      PyArray_FromAny(ndarray, nullptr, 0, 0, NPY_ARRAY_CARRAY, nullptr)));
  if (!array_safe) return errors::InvalidArgument(kFeedDictErrorMsg);
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
    const string serialized(reinterpret_cast<char*>(PyArray_DATA(array)),
                            PyArray_NBYTES(array));
    *resource_handle = new ResourceHandle();
    (*resource_handle)->ParseFromString(serialized);
    TF_Tensor* tf_tensor =
        TF_AllocateTensor(dtype, {}, 0, sizeof(ResourceHandle));
    std::memcpy(TF_TensorData(tf_tensor),
                reinterpret_cast<void*>(*resource_handle),
                sizeof(ResourceHandle));
    *out_tensor = make_safe(tf_tensor);
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

}  // namespace

Safe_PyObjectPtr make_safe(PyObject* o) {
  return Safe_PyObjectPtr(o, Py_DECREF_wrapper);
}

Safe_TF_TensorPtr make_safe(TF_Tensor* tensor) {
  return Safe_TF_TensorPtr(tensor, TF_DeleteTensor);
}

void TF_Run_wrapper_helper(TF_DeprecatedSession* session, const char* handle,
                           const TF_Buffer* run_options, PyObject* feed_dict,
                           const NameVector& output_names,
                           const NameVector& target_nodes,
                           TF_Status* out_status, PyObjectVector* out_values,
                           TF_Buffer* run_outputs) {
  // 1. Convert the feed inputs to the appropriate form for TF_Run.
  if (!PyDict_Check(feed_dict)) {
    Set_TF_Status_from_Status(out_status,
                              errors::InvalidArgument(kFeedDictErrorMsg));
    return;
  }

  NameVector input_names;
  std::vector<Safe_TF_TensorPtr> inputs_safe;  // Used to delete tensors.
  TF_TensorVector inputs_unsafe;     // Used to contain the arg to TF_Run.

  PyObject* key;
  PyObject* value;
  Py_ssize_t pos = 0;
  int index = 0;
  Status s;

  gtl::InlinedVector<std::unique_ptr<ResourceHandle>, 4> resource_handles;
  while (PyDict_Next(feed_dict, &pos, &key, &value)) {
    char* key_string = PyBytes_AsString(key);
    if (!key_string) {
      Set_TF_Status_from_Status(out_status,
                                errors::InvalidArgument(kFeedDictErrorMsg));
      return;
    }
    input_names.push_back(key_string);

    inputs_safe.emplace_back(make_safe(static_cast<TF_Tensor*>(nullptr)));
    ResourceHandle* resource_handle;
    s = PyArrayToTFTensor(value, &inputs_safe.back(), &resource_handle);
    if (!s.ok()) {
      Set_TF_Status_from_Status(out_status, s);
      return;
    }
    inputs_unsafe.push_back(inputs_safe.back().get());
    if (resource_handle != nullptr) {
      resource_handles.emplace_back(resource_handle);
    }
    ++index;
  }

  // 2. Allocate a container for the output data.
  TF_TensorVector outputs(output_names.size());

  // In case any tensors were leftover from previous runs we might as well clear
  // them here.
  ClearDecrefCache();

  // 3. Actually call TF_Run().
  Py_BEGIN_ALLOW_THREADS;
  if (handle == nullptr) {
    TF_Run(session, run_options, input_names.data(), inputs_unsafe.data(),
           input_names.size(), const_cast<const char**>(output_names.data()),
           outputs.data(), output_names.size(),
           const_cast<const char**>(target_nodes.data()), target_nodes.size(),
           run_outputs, out_status);
  } else {
    TF_PRun(session, handle, input_names.data(), inputs_unsafe.data(),
            input_names.size(), const_cast<const char**>(output_names.data()),
            outputs.data(), output_names.size(),
            const_cast<const char**>(target_nodes.data()), target_nodes.size(),
            out_status);
  }

  Py_END_ALLOW_THREADS;

  // Decref any numpy arrays we are not using anymore.
  ClearDecrefCache();

  if (TF_GetCode(out_status) != TF_OK) {
    return;
  }

  // 4. We now own the fetched tensors, so set up a safe container to
  // delete them when we exit this scope.
  std::vector<Safe_TF_TensorPtr> tf_outputs_safe;
  for (const auto& output : outputs) {
    tf_outputs_safe.emplace_back(make_safe(output));
  }

  // 5. Convert the fetched tensors into numpy ndarrays. Store them in a safe
  // container so that we do not leak
  std::vector<Safe_PyObjectPtr> py_outputs_safe;
  for (size_t i = 0; i < output_names.size(); ++i) {
    PyObject* py_array;
    s = TFTensorToPyArray(std::move(tf_outputs_safe[i]), &py_array);
    if (!s.ok()) {
      Set_TF_Status_from_Status(out_status, s);
      return;
    }
    py_outputs_safe.emplace_back(make_safe(py_array));
  }

  // 6. If we reach this point, we have successfully built a list of objects
  // so we can release them from the safe container.
  for (auto& output : py_outputs_safe) {
    out_values->push_back(output.release());
  }
}

// Wrapper for TF_Run that converts the arguments to appropriate types.
// If *out_status is OK, the caller becomes the owner of the PyObjects
// in *out_values.
void TF_Run_wrapper(TF_DeprecatedSession* session, const TF_Buffer* run_options,
                    PyObject* feed_dict, const NameVector& output_names,
                    const NameVector& target_nodes, TF_Status* out_status,
                    PyObjectVector* out_values, TF_Buffer* run_outputs) {
  TF_Run_wrapper_helper(session, nullptr, run_options, feed_dict, output_names,
                        target_nodes, out_status, out_values, run_outputs);
  ClearDecrefCache();
}

// Wrapper for TF_PRunSetup that converts the arguments to appropriate types.
// If *out_status is OK, the caller becomes the owner of *out_handle.
void TF_PRunSetup_wrapper(TF_DeprecatedSession* session,
                          const NameVector& input_names,
                          const NameVector& output_names,
                          const NameVector& target_nodes, TF_Status* out_status,
                          const char** out_handle) {
  Py_BEGIN_ALLOW_THREADS;
  TF_PRunSetup(
      session, const_cast<const char**>(input_names.data()), input_names.size(),
      const_cast<const char**>(output_names.data()), output_names.size(),
      const_cast<const char**>(target_nodes.data()), target_nodes.size(),
      out_handle, out_status);
  Py_END_ALLOW_THREADS;
}

// Wrapper for TF_PRun that converts the arguments to appropriate types.
// If *out_status is OK, the caller becomes the owner of the PyObjects
// in *out_values.
void TF_PRun_wrapper(TF_DeprecatedSession* session, const char* handle,
                     PyObject* feed_dict, const NameVector& output_names,
                     TF_Status* out_status, PyObjectVector* out_values) {
  TF_Run_wrapper_helper(session, handle, nullptr, feed_dict, output_names,
                        NameVector(), out_status, out_values, nullptr);
  ClearDecrefCache();
}

// Wrapper for TF_Reset that converts the string vectors to character arrays.
void TF_Reset_wrapper(const TF_SessionOptions* opt,
                      const NameVector& containers, TF_Status* out_status) {
  TF_Reset(opt, const_cast<const char**>(containers.data()), containers.size(),
           out_status);
}

void TF_SessionRun_wrapper_helper(TF_Session* session, const char* handle,
                                  const TF_Buffer* run_options,
                                  const std::vector<TF_Output>& inputs,
                                  const std::vector<PyObject*>& input_ndarrays,
                                  const std::vector<TF_Output>& outputs,
                                  const std::vector<TF_Operation*>& targets,
                                  TF_Buffer* run_metadata,
                                  TF_Status* out_status,
                                  std::vector<PyObject*>* py_outputs) {
  DCHECK_EQ(inputs.size(), input_ndarrays.size());
  DCHECK(py_outputs != nullptr);
  DCHECK(py_outputs->empty());
  Status s;

  // Convert input ndarray PyObjects to TF_Tensors. We maintain a continuous
  // array of TF_Tensor*s as well as scoped containers to make sure they're
  // cleaned up properly.
  //
  // Memory management:
  // PyArrayToTFTensor() creates a new ndarray PyObject from the input
  // ndarray. We manage the new ndarray's lifetime in order to keep the
  // underlying data buffer alive (the new ndarray also guarantees a contiguous
  // data buffer). The new ndarray's data buffer is used to create the
  // corresponding TF_Tensor. The TF_Tensor's deallocator will queue the new
  // ndarray to be decref'd by the next ClearDecrefCache() call (we can't call
  // Py_DECREF in the deallocator directly because the GIL must be held).
  //
  // Note that TF_Tensor may directly delegate its data and deallocator to a
  // TensorBuffer, which may outlive the TF_Tensor (e.g. if the tensor gets
  // queued or assigned to a variable).
  TF_TensorVector input_vals;
  std::vector<Safe_TF_TensorPtr> input_vals_safe;
  gtl::InlinedVector<std::unique_ptr<ResourceHandle>, 4> resource_handles;
  for (PyObject* ndarray : input_ndarrays) {
    input_vals_safe.emplace_back(make_safe(static_cast<TF_Tensor*>(nullptr)));
    ResourceHandle* resource_handle;
    s = PyArrayToTFTensor(ndarray, &input_vals_safe.back(), &resource_handle);
    if (resource_handle != nullptr) {
      resource_handles.emplace_back(resource_handle);
    }
    if (!s.ok()) {
      Set_TF_Status_from_Status(out_status, s);
      return;
    }
    input_vals.push_back(input_vals_safe.back().get());
  }

  // Allocate space for output TF_Tensor*s
  TF_TensorVector output_vals(outputs.size());

  // Clear up any unused memory leftover from previous runs
  ClearDecrefCache();

  // Call TF_SessionRun() (and release GIL during execution)
  Py_BEGIN_ALLOW_THREADS;
  if (handle == nullptr) {
    TF_SessionRun(session, run_options, inputs.data(), input_vals.data(),
                  inputs.size(), outputs.data(), output_vals.data(),
                  outputs.size(), targets.data(), targets.size(), run_metadata,
                  out_status);
  } else {
    TF_SessionPRun(session, handle, inputs.data(), input_vals.data(),
                   inputs.size(), outputs.data(), output_vals.data(),
                   outputs.size(), targets.data(), targets.size(), out_status);
  }
  Py_END_ALLOW_THREADS;

  // Create scoped containers for output tensors
  std::vector<Safe_TF_TensorPtr> output_vals_safe;
  for (TF_Tensor* output : output_vals) {
    output_vals_safe.emplace_back(make_safe(output));
  }

  // Convert outputs to ndarrays (in scoped containers)
  std::vector<Safe_PyObjectPtr> py_outputs_safe;
  for (size_t i = 0; i < outputs.size(); ++i) {
    PyObject* py_array;
    s = TFTensorToPyArray(std::move(output_vals_safe[i]), &py_array);
    if (!s.ok()) {
      Set_TF_Status_from_Status(out_status, s);
      return;
    }
    py_outputs_safe.emplace_back(make_safe(py_array));
  }

  // If we reach this point, we have successfully built a list of objects so we
  // can release them from the safe container into the return vector.
  for (size_t i = 0; i < outputs.size(); ++i) {
    py_outputs->push_back(py_outputs_safe[i].release());
  }
}

void TF_SessionRun_wrapper(TF_Session* session, const TF_Buffer* run_options,
                           const std::vector<TF_Output>& inputs,
                           const std::vector<PyObject*>& input_ndarrays,
                           const std::vector<TF_Output>& outputs,
                           const std::vector<TF_Operation*>& targets,
                           TF_Buffer* run_metadata, TF_Status* out_status,
                           std::vector<PyObject*>* py_outputs) {
  TF_SessionRun_wrapper_helper(session, nullptr, run_options, inputs,
                               input_ndarrays, outputs, targets, run_metadata,
                               out_status, py_outputs);
  // Release any unused ndarray references (see memory management comment in
  // TF_SessionRun_wrapper_helper)
  ClearDecrefCache();
}

string EqualGraphDefWrapper(const string& actual, const string& expected) {
  GraphDef actual_def;
  if (!actual_def.ParseFromString(actual)) {
    return "actual is not a valid serialized GraphDef";
  }
  GraphDef expected_def;
  if (!expected_def.ParseFromString(expected)) {
    return "expected is not a valid serialized GraphDef";
  }
  string diff;
  return EqualGraphDef(actual_def, expected_def, &diff) ? "" : diff;
}

void TF_SessionPRunSetup_wrapper(TF_Session* session,
                                 const std::vector<TF_Output>& inputs,
                                 const std::vector<TF_Output>& outputs,
                                 const std::vector<TF_Operation*>& targets,
                                 const char** out_handle,
                                 TF_Status* out_status) {
  // Call TF_SessionPRunSetup() (and release GIL during execution)
  Py_BEGIN_ALLOW_THREADS;
  TF_SessionPRunSetup(session, inputs.data(), inputs.size(), outputs.data(),
                      outputs.size(), targets.data(), targets.size(),
                      out_handle, out_status);
  Py_END_ALLOW_THREADS;
}

void TF_SessionPRun_wrapper(TF_Session* session, const char* handle,
                            const std::vector<TF_Output>& inputs,
                            const std::vector<PyObject*>& input_ndarrays,
                            const std::vector<TF_Output>& outputs,
                            TF_Status* out_status,
                            std::vector<PyObject*>* py_outputs) {
  const std::vector<TF_Operation*> targets;
  TF_SessionRun_wrapper_helper(session, handle,
                               nullptr,  // run_options
                               inputs, input_ndarrays, outputs, targets,
                               nullptr,  // run_metadata
                               out_status, py_outputs);
  // Release any unused ndarray references (see memory management comment in
  // TF_SessionRun_wrapper_helper)
  ClearDecrefCache();
}

std::vector<TF_Operation*> TF_OperationGetControlInputs_wrapper(
    TF_Operation* oper) {
  std::vector<TF_Operation*> control_inputs(TF_OperationNumControlInputs(oper));
  TF_OperationGetControlInputs(oper, control_inputs.data(),
                               control_inputs.size());
  return control_inputs;
}

}  // namespace tensorflow
