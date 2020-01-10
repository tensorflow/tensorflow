#include "tensorflow/python/client/tf_session_helper.h"

#include <cstring>

#include "tensorflow/core/lib/core/coding.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/platform/port.h"

namespace tensorflow {

namespace {

// Container types for the various temporary values used internally in
// the wrapper.

// A TF_TensorVector is a vector of borrowed pointers to TF_Tensors.
typedef gtl::InlinedVector<TF_Tensor*, 8> TF_TensorVector;

// Safe containers for (an) owned TF_Tensor(s). On destruction, the
// tensor will be deleted by TF_DeleteTensor.
typedef std::unique_ptr<TF_Tensor, decltype(&TF_DeleteTensor)>
    Safe_TF_TensorPtr;
typedef std::vector<Safe_TF_TensorPtr> Safe_TF_TensorVector;
Safe_TF_TensorPtr make_safe(TF_Tensor* tensor) {
  return Safe_TF_TensorPtr(tensor, TF_DeleteTensor);
}

// Safe container for an owned TF_Status. On destruction, the status
// will be deleted by TF_DeleteStatus.
typedef std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)>
    Safe_TF_StatusPtr;
Safe_TF_StatusPtr make_safe(TF_Status* status) {
  return Safe_TF_StatusPtr(status, TF_DeleteStatus);
}

Status PyArrayDescr_to_TF_DataType(PyArray_Descr* descr,
                                   TF_DataType* out_tf_datatype) {
  PyObject* key;
  PyObject* value;
  Py_ssize_t pos = 0;
  if (PyDict_Next(descr->fields, &pos, &key, &value)) {
    const char* key_string = PyBytes_AsString(key);
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
    } else if (key == "qint32") {
      *out_tf_datatype = TF_QINT32;
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
  PyArray_Descr* descr = array->descr;
  switch (pyarray_type) {
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
    case NPY_INT16:
      *out_tf_datatype = TF_INT16;
      break;
    case NPY_INT8:
      *out_tf_datatype = TF_INT8;
      break;
    case NPY_INT64:
      *out_tf_datatype = TF_INT64;
      break;
    case NPY_BOOL:
      *out_tf_datatype = TF_BOOL;
      break;
    case NPY_COMPLEX64:
      *out_tf_datatype = TF_COMPLEX;
      break;
    case NPY_OBJECT:
      *out_tf_datatype = TF_STRING;
      break;
    case NPY_VOID:
      // Quantized types are currently represented as custom struct types.
      // PyArray_TYPE returns NPY_VOID for structs, and we should look into
      // descr to derive the actual type.
      return PyArrayDescr_to_TF_DataType(descr, out_tf_datatype);
    default:
      // TODO(mrry): Support these.
      return errors::Internal("Unsupported feed type");
  }
  return Status::OK();
}

Status TF_DataType_to_PyArray_TYPE(TF_DataType tf_datatype,
                                   int* out_pyarray_type) {
  switch (tf_datatype) {
    case TF_FLOAT:
      *out_pyarray_type = NPY_FLOAT32;
      break;
    case TF_DOUBLE:
      *out_pyarray_type = NPY_FLOAT64;
      break;
    case TF_INT32:
      *out_pyarray_type = NPY_INT32;
      break;
    case TF_UINT8:
      *out_pyarray_type = NPY_UINT8;
      break;
    case TF_INT16:
      *out_pyarray_type = NPY_INT16;
      break;
    case TF_INT8:
      *out_pyarray_type = NPY_INT8;
      break;
    case TF_INT64:
      *out_pyarray_type = NPY_INT64;
      break;
    case TF_BOOL:
      *out_pyarray_type = NPY_BOOL;
      break;
    case TF_COMPLEX:
      *out_pyarray_type = NPY_COMPLEX64;
      break;
    case TF_STRING:
      *out_pyarray_type = NPY_OBJECT;
      break;
    // TODO(keveman): These should be changed to NPY_VOID, and the type used for
    // the resulting numpy array should be the custom struct types that we
    // expect for quantized types.
    case TF_QINT8:
      *out_pyarray_type = NPY_INT8;
      break;
    case TF_QUINT8:
      *out_pyarray_type = NPY_UINT8;
      break;
    case TF_QINT32:
      *out_pyarray_type = NPY_INT32;
      break;
    case TF_BFLOAT16:
      *out_pyarray_type = NPY_UINT16;
      break;
    default:
      return errors::Internal("Unsupported fetch type");
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
    auto item = tensorflow::make_safe(
        PyArray_GETITEM(array, PyArray_ITER_DATA(iter.get())));
    if (!item.get()) {
      return errors::Internal("Unable to get element from the feed.");
    }
    char* ptr;
    Py_ssize_t len;

#if PY_VERSION_HEX >= 0x03030000
    // Accept unicode in Python 3, by converting to UTF-8 bytes.
    if (PyUnicode_Check(item.get())) {
      ptr = PyUnicode_AsUTF8AndSize(item.get(), &len);
      if (!buf) {
        return errors::Internal("Unable to get element from the feed.");
      }
    } else {
#endif
      int success = PyBytes_AsStringAndSize(item.get(), &ptr, &len);
      if (success != 0) {
        return errors::Internal("Unable to get element from the feed.");
      }
#if PY_VERSION_HEX >= 0x03030000
    }
#endif
    f(ptr, len);
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
  if (offset >= (limit - data_start) || !p || (*len > (limit - p))) {
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
  const char* ptr;
  tensorflow::uint64 len;
  TF_RETURN_IF_ERROR(
      TF_StringTensor_GetPtrAndLen(tensor, num_elements, i, &ptr, &len));
  auto py_string = tensorflow::make_safe(PyBytes_FromStringAndSize(ptr, len));
  int success =
      PyArray_SETITEM(pyarray, PyArray_ITER_DATA(i_ptr), py_string.get());
  if (success != 0) {
    return errors::Internal("Error setting element ", i);
  }
  return Status::OK();
}

// Converts the given TF_Tensor to a Numpy array.
// If the returned status is OK, the caller becomes the owner of *out_array.
Status TF_Tensor_to_PyObject(TF_Tensor* tensor, PyObject** out_array) {
  // A fetched operation will correspond to a null tensor, and a None
  // in Python.
  if (tensor == nullptr) {
    Py_INCREF(Py_None);
    *out_array = Py_None;
    return Status::OK();
  }

  const int ndims = TF_NumDims(tensor);
  gtl::InlinedVector<npy_intp, 4> dims(ndims);
  tensorflow::int64 nelems = 1;
  for (int i = 0; i < ndims; ++i) {
    dims[i] = TF_Dim(tensor, i);
    nelems *= dims[i];
  }

  // Convert TensorFlow dtype to numpy type descriptor.
  int type_num;
  TF_RETURN_IF_ERROR(
      TF_DataType_to_PyArray_TYPE(TF_TensorType(tensor), &type_num));
  PyArray_Descr* descr = PyArray_DescrFromType(type_num);

  // Copy the TF_TensorData into a newly-created ndarray and return it.
  // TODO(mrry): Perhaps investigate zero-copy approaches. This would involve
  // creating an ndarray-like object that wraps the TF_Tensor buffer, and
  // maps its destructor to TF_DeleteTensor.
  Safe_PyObjectPtr safe_out_array =
      tensorflow::make_safe(PyArray_Empty(ndims, dims.data(), descr, 0));
  if (!safe_out_array) {
    return errors::Internal("Could not allocate ndarray");
  }
  PyArrayObject* py_array =
      reinterpret_cast<PyArrayObject*>(safe_out_array.get());
  if (PyArray_NBYTES(py_array) != TF_TensorByteSize(tensor)) {
    if (TF_TensorType(tensor) == TF_STRING) {
      // Copy element by element.
      auto iter = tensorflow::make_safe(PyArray_IterNew(safe_out_array.get()));
      for (tensorflow::int64 i = 0; i < nelems; ++i) {
        auto s =
            CopyStringToPyArrayElement(py_array, iter.get(), tensor, nelems, i);
        if (!s.ok()) {
          return s;
        }
        PyArray_ITER_NEXT(iter.get());
      }
    } else {
      return errors::Internal("ndarray was ", PyArray_NBYTES(py_array),
                              " bytes but TF_Tensor was ",
                              TF_TensorByteSize(tensor), " bytes");
    }
  } else {
    memcpy(py_array->data, TF_TensorData(tensor), PyArray_NBYTES(py_array));
  }

  // PyArray_Return turns rank 0 arrays into numpy scalars
  *out_array = PyArray_Return(
      reinterpret_cast<PyArrayObject*>(safe_out_array.release()));
  return Status::OK();
}

tensorflow::Status TF_Status_to_Status(TF_Status* tf_status) {
  TF_Code code = TF_GetCode(tf_status);
  const string message(TF_Message(tf_status));

  switch (code) {
    case TF_OK:
      return Status::OK();
    case TF_CANCELLED:
      return errors::Cancelled(message);
    case TF_UNKNOWN:
      return errors::Unknown(message);
    case TF_INVALID_ARGUMENT:
      return errors::InvalidArgument(message);
    case TF_DEADLINE_EXCEEDED:
      return errors::DeadlineExceeded(message);
    case TF_NOT_FOUND:
      return errors::NotFound(message);
    case TF_ALREADY_EXISTS:
      return errors::AlreadyExists(message);
    case TF_PERMISSION_DENIED:
      return errors::PermissionDenied(message);
    case TF_UNAUTHENTICATED:
      return errors::Unauthenticated(message);
    case TF_RESOURCE_EXHAUSTED:
      return errors::ResourceExhausted(message);
    case TF_FAILED_PRECONDITION:
      return errors::FailedPrecondition(message);
    case TF_ABORTED:
      return errors::Aborted(message);
    case TF_OUT_OF_RANGE:
      return errors::OutOfRange(message);
    case TF_UNIMPLEMENTED:
      return errors::Unimplemented(message);
    case TF_INTERNAL:
      return errors::Internal(message);
    case TF_UNAVAILABLE:
      return errors::Unavailable(message);
    case TF_DATA_LOSS:
      return errors::DataLoss(message);
    default:
      return errors::Internal("Got error with unknown code: ", code, " ",
                              message);
  }
}

static bool numpy_imported = false;

}  // namespace

Safe_PyObjectPtr make_safe(PyObject* o) {
  return Safe_PyObjectPtr(o, Py_DECREF_wrapper);
}

// Wrapper for TF_Run that converts the arguments to appropriate types.
// If *out_status is OK, the caller becomes the owner of the PyObjects
// in *out_values.
void TF_Run_wrapper(TF_Session* session, const FeedVector& inputs,
                    const NameVector& output_names,
                    const NameVector& target_nodes, Status* out_status,
                    PyObjectVector* out_values) {
  // 0. Ensure that numpy has been imported.
  if (!numpy_imported) {
    import_array();
    numpy_imported = true;
  }

  // 1. Convert the feed inputs to the appropriate form for TF_Run.
  NameVector input_names;
  Safe_PyObjectVector
      py_inputs_safe;  // Used to decref the input arrays on failure.
  Safe_TF_TensorVector inputs_safe;  // Used to delete tensors on failure.
  TF_TensorVector inputs_unsafe;     // Used to contain the arg to TF_Run.

  for (const auto& name_and_array : inputs) {
    py_inputs_safe.emplace_back(
        make_safe(reinterpret_cast<PyObject*>(name_and_array.second)));
  }

  for (int i = 0; i < inputs.size(); ++i) {
    input_names.push_back(inputs[i].first);
    PyArrayObject* array = inputs[i].second;

    // Convert numpy dtype to TensorFlow dtype.
    TF_DataType dtype;
    *out_status = PyArray_TYPE_to_TF_DataType(array, &dtype);
    if (!out_status->ok()) {
      return;
    }

    tensorflow::int64 nelems = 1;
    gtl::InlinedVector<tensorflow::int64, 4> dims;
    for (int i = 0; i < PyArray_NDIM(array); ++i) {
      dims.push_back(PyArray_SHAPE(array)[i]);
      nelems *= dims[i];
    }

    // Create a TF_Tensor based on the fed data. In the case of non-string data
    // type, this steals a reference to array, which will be relinquished when
    // the underlying buffer is deallocated. For string, a new temporary buffer
    // is allocated into which the strings are encoded.
    if (dtype != TF_STRING) {
      // NOTE(mrry): We currently copy the numpy array into a new
      // buffer to avoid possible issues on deallocation (such as
      // having to acquire the Python Global Interpreter Lock).
      // TODO(mrry): Investigate in what cases we can safely acquire
      size_t size = PyArray_NBYTES(array);
      // NOTE(mrry): 32 is the upper bound on current alignment
      // requirements for tensorflow::Tensor. We hard code this here to
      // avoid taking a dependency on Eigen in the client code.
      void* data = tensorflow::cpu_allocator()->AllocateRaw(32, size);
      std::memcpy(data, array->data, size);
      inputs_safe.emplace_back(make_safe(
          TF_NewTensor(dtype, dims.data(), dims.size(), data, size,
                       [](void* data, size_t len, void* arg) {
                         tensorflow::cpu_allocator()->DeallocateRaw(data);
                       },
                       nullptr)));
      // The destruction of the numpy array will now be handled by the
      // inputs_safe destructor.
      py_inputs_safe[i].reset();
    } else {
      size_t size;
      void* encoded;
      Status s = EncodePyBytesArray(array, nelems, &size, &encoded);
      if (!s.ok()) {
        *out_status = s;
        return;
      }
      inputs_safe.emplace_back(
          make_safe(TF_NewTensor(dtype, dims.data(), dims.size(), encoded, size,
                                 [](void* data, size_t len, void* arg) {
                                   delete[] reinterpret_cast<char*>(data);
                                 },
                                 array)));
      // The destruction of the numpy array will now be handled by the
      // inputs_safe destructor.
      py_inputs_safe[i].reset();
    }
    inputs_unsafe.push_back(inputs_safe.back().get());
  }

  // 2. Allocate a container for the output data.
  TF_TensorVector outputs(output_names.size());

  Safe_TF_StatusPtr status = make_safe(TF_NewStatus());

  // 3. Actually call TF_Run().
  Py_BEGIN_ALLOW_THREADS;
  TF_Run(session, input_names.data(), inputs_unsafe.data(), input_names.size(),
         const_cast<const char**>(output_names.data()), outputs.data(),
         output_names.size(), const_cast<const char**>(target_nodes.data()),
         target_nodes.size(), status.get());
  Py_END_ALLOW_THREADS;

  // 4. The TensorFlow runtime has taken ownership of the fed tensors,
  // so we release the safe pointers to them.
  for (auto& input : inputs_safe) {
    input.release();
  }

  if (TF_GetCode(status.get()) != TF_OK) {
    *out_status = TF_Status_to_Status(status.get());
    return;
  }

  // 5. We now own the fetched tensors, so set up a safe container to
  // delete them when we exit this scope.
  Safe_TF_TensorVector tf_outputs_safe;
  for (const auto& output : outputs) {
    tf_outputs_safe.emplace_back(make_safe(output));
  }

  // 6. Convert the fetched tensors into numpy ndarrays. Store them in a safe
  // container so that we do not leak
  Safe_PyObjectVector py_outputs_safe;
  for (int i = 0; i < output_names.size(); ++i) {
    PyObject* py_array;
    *out_status = TF_Tensor_to_PyObject(outputs[i], &py_array);
    if (!out_status->ok()) {
      return;
    }
    py_outputs_safe.emplace_back(make_safe(py_array));
  }

  // 7. If we reach this point, we have successfully built a list of objects
  // so we can release them from the safe container.
  for (auto& output : py_outputs_safe) {
    out_values->push_back(output.release());
  }
  *out_status = Status::OK();
}

}  // namespace tensorflow
