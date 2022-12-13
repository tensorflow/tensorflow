/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
// Must be included first
// clang-format off
#include "tensorflow/tsl/python/lib/core/numpy.h" //NOLINT
// clang-format on

#include "tensorflow/compiler/xla/python/py_values.h"

// NOLINTBEGIN
#include <exception>
#include <functional>
#include <memory>
#include <string>
#include <utility>
// NOLINTEND

#include "pybind11/pybind11.h"
#include "pybind11/pytypes.h"
#ifdef JAX_ENABLE_IFRT
#include "tensorflow/compiler/xla/python/ifrt/array.h"
#include "tensorflow/compiler/xla/python/ifrt/shape.h"
#include "tensorflow/compiler/xla/python/ifrt/sharding.h"
#endif
#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/python/py_array.h"
#include "tensorflow/compiler/xla/python/py_buffer.h"
#include "tensorflow/compiler/xla/python/python_ref_manager.h"
#include "tensorflow/compiler/xla/python/sharded_device_array.h"
#include "tensorflow/compiler/xla/python/sharding.h"
#include "tensorflow/compiler/xla/python/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/tsl/platform/errors.h"
#include "tensorflow/tsl/platform/statusor.h"
#include "tensorflow/tsl/profiler/lib/traceme.h"

namespace py = pybind11;

namespace xla {

namespace {

#ifdef JAX_ENABLE_IFRT
using DevicePutFunc = std::function<StatusOr<DevicePutResult>(
    py::handle, ifrt::Client*, ifrt::Device*, const DevicePutOptions& options)>;
#else
using DevicePutFunc = std::function<StatusOr<DevicePutResult>(
    py::handle, PjRtDevice*, const DevicePutOptions& options)>;
#endif

template <typename T, typename SquashedT>
StatusOr<DevicePutResult> HandlePythonScalar(py::handle obj,
#ifdef JAX_ENABLE_IFRT
                                             ifrt::Client* client,
                                             ifrt::Device* to_device,
#else
                                             PjRtDevice* to_device,
#endif
                                             const DevicePutOptions& options) {
  T data;

  try {
    data = py::cast<T>(obj);
  } catch (const std::exception& e) {
    return InvalidArgument(
        "Unable to convert Python scalar to %s. This most likely means the "
        "value (%s) overflows the range of the type.",
        PrimitiveType_Name(primitive_util::NativeToPrimitiveType<T>()),
        py::repr(obj));
  }

  void* ptr;
  SquashedT squashed_data;
  Shape shape;
  PrimitiveType type;
  if (std::is_same<T, SquashedT>() || !options.squash_64bit_types) {
    ptr = &data;
    type = primitive_util::NativeToPrimitiveType<T>();
  } else {
    // TODO(phawkins): we should check for overflow here, e.g., because of bugs
    // like https://github.com/google/jax/issues/2006
    squashed_data = static_cast<SquashedT>(data);
    ptr = &squashed_data;
    type = primitive_util::NativeToPrimitiveType<SquashedT>();
  }
  // Must release the GIL before BufferFromHostBuffer because backends may
  // decide to block/sleep for device buffer allocation.
  py::gil_scoped_release gil_release;
#ifdef JAX_ENABLE_IFRT
  TF_ASSIGN_OR_RETURN(auto ifrt_dtype, xla::ifrt::ToDType(type));
  TF_ASSIGN_OR_RETURN(
      auto ifrt_array,
      client->MakeArrayFromHostBuffer(
          ptr, ifrt_dtype, /*shape=*/ifrt::Shape({}), /*byte_strides=*/{},
          ifrt::SingleDeviceSharding::Create(to_device),
          ifrt::Client::HostBufferSemantics::kImmutableOnlyDuringCall,
          /*on_done_with_host_buffer=*/{}));
  return DevicePutResult(std::move(ifrt_array), /*weak_type=*/true);
#else
  TF_ASSIGN_OR_RETURN(
      auto buffer,
      to_device->client()->BufferFromHostBuffer(
          ptr, type, /*dims=*/{}, /*byte_strides=*/{},
          PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall,
          /*on_done_with_host_buffer=*/nullptr, to_device));
  return DevicePutResult(std::move(buffer), /*weak_type=*/true);
#endif
}

StatusOr<DevicePutResult> HandlePythonInt(py::handle obj,
#ifdef JAX_ENABLE_IFRT
                                          ifrt::Client* client,
                                          ifrt::Device* to_device,
#else
                                          PjRtDevice* to_device,
#endif
                                          const DevicePutOptions& options) {
  void* ptr;
  PrimitiveType type;
  int64_t data_int64;
  int32_t data_int32;

  if (options.squash_64bit_types) {
    try {
      data_int32 = py::cast<int32_t>(obj);
    } catch (const std::exception& e) {
      return InvalidArgument(
          "Unable to convert Python scalar to %s. This most likely means the "
          "value (%s) overflows the range of the type.",
          PrimitiveType_Name(primitive_util::NativeToPrimitiveType<int32_t>()),
          py::repr(obj));
    }
    ptr = &data_int32;
    type = S32;
  } else {
    try {
      data_int64 = py::cast<int64_t>(obj);
    } catch (const std::exception& e) {
      return InvalidArgument(
          "Unable to convert Python scalar to %s. This most likely means the "
          "value (%s) overflows the range of the type.",
          PrimitiveType_Name(primitive_util::NativeToPrimitiveType<int64_t>()),
          py::repr(obj));
    }
    ptr = &data_int64;
    type = S64;
  }
  // Must release the GIL before BufferFromHostBuffer because backends may
  // decide to block/sleep for device buffer allocation.
  py::gil_scoped_release gil_release;
#ifdef JAX_ENABLE_IFRT
  TF_ASSIGN_OR_RETURN(auto ifrt_dtype, xla::ifrt::ToDType(type));
  TF_ASSIGN_OR_RETURN(
      auto ifrt_array,
      client->MakeArrayFromHostBuffer(
          ptr, ifrt_dtype, /*shape=*/xla::ifrt::Shape({}), /*byte_strides=*/{},
          ifrt::SingleDeviceSharding::Create(to_device),
          ifrt::Client::HostBufferSemantics::kImmutableOnlyDuringCall,
          /*on_done_with_host_buffer=*/nullptr));
  return DevicePutResult(std::move(ifrt_array), /*weak_type=*/true);
#else
  TF_ASSIGN_OR_RETURN(
      auto buffer,
      to_device->client()->BufferFromHostBuffer(
          ptr, type, /*dims=*/{}, /*byte_strides=*/{},
          PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall,
          /*on_done_with_host_buffer=*/nullptr, to_device));
  return DevicePutResult(std::move(buffer), /*weak_type=*/true);
#endif
}

template <typename T, typename SquashedT = T>
StatusOr<DevicePutResult> HandleNumpyScalar(py::handle h,
#ifdef JAX_ENABLE_IFRT
                                            ifrt::Client* client,
                                            ifrt::Device* to_device,
#else
                                            PjRtDevice* to_device,
#endif
                                            const DevicePutOptions& options) {
  T data;
  SquashedT data_squashed;
  void* ptr;
  PrimitiveType type;
  if (std::is_same<T, bfloat16>()) {
    // For extension types, ScalarAsCtype returns a pointer to the data.
    PyArray_ScalarAsCtype(h.ptr(), &ptr);
    type = BF16;
  } else if (std::is_same<T, SquashedT>() || !options.squash_64bit_types) {
    PyArray_ScalarAsCtype(h.ptr(), &data);
    ptr = &data;
    type = primitive_util::NativeToPrimitiveType<T>();
  } else {
    PyArray_ScalarAsCtype(h.ptr(), &data);
    data_squashed = static_cast<SquashedT>(data);
    ptr = &data_squashed;
    type = primitive_util::NativeToPrimitiveType<SquashedT>();
  }
  // Must release the GIL before BufferFromHostBuffer because backends may
  // decide to block/sleep for device buffer allocation.
  py::gil_scoped_release gil_release;
#ifdef JAX_ENABLE_IFRT
  TF_ASSIGN_OR_RETURN(auto ifrt_dtype, xla::ifrt::ToDType(type));
  TF_ASSIGN_OR_RETURN(
      auto ifrt_array,
      client->MakeArrayFromHostBuffer(
          ptr, ifrt_dtype, /*shape=*/xla::ifrt::Shape({}), /*byte_strides=*/{},
          ifrt::SingleDeviceSharding::Create(to_device),
          ifrt::Client::HostBufferSemantics::kImmutableOnlyDuringCall,
          /*on_done_with_host_buffer=*/nullptr));
  return DevicePutResult(std::move(ifrt_array), /*weak_type=*/false);
#else
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<PjRtBuffer> buffer,
      to_device->client()->BufferFromHostBuffer(
          ptr, type, /*dims=*/{}, /*byte_strides=*/{},
          PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall,
          /*on_done_with_host_buffer=*/nullptr, to_device));
  return DevicePutResult(std::move(buffer), /*weak_type=*/false);
#endif
}

StatusOr<DevicePutResult> HandleNumpyArray(py::handle h,
#ifdef JAX_ENABLE_IFRT
                                           ifrt::Client* client,
                                           ifrt::Device* to_device,
#else
                                           PjRtDevice* to_device,
#endif
                                           const DevicePutOptions& options) {
  py::array array = py::cast<py::array>(h);
  TF_ASSIGN_OR_RETURN(PrimitiveType type, DtypeToPrimitiveType(array.dtype()));

  PrimitiveType squashed_type;
  if (options.squash_64bit_types) {
    squashed_type = Squash64BitTypes(type);
    if (squashed_type != type) {
      TF_ASSIGN_OR_RETURN(py::dtype squashed_dtype,
                          PrimitiveTypeToDtype(squashed_type));
      array = py::reinterpret_steal<py::array>(PyArray_CastToType(
          reinterpret_cast<PyArrayObject*>(array.ptr()),
          reinterpret_cast<PyArray_Descr*>(squashed_dtype.release().ptr()),
          /*fortran=*/0));
    }
  } else {
    squashed_type = type;
  }

  absl::InlinedVector<int64_t, 4> dims(array.ndim());
  absl::InlinedVector<int64_t, 4> byte_strides(array.ndim());
  for (int i = 0; i < array.ndim(); ++i) {
    dims[i] = array.shape(i);
    byte_strides[i] = array.strides(i);
  }
  const void* data = array.data();
#ifdef JAX_ENABLE_IFRT
  ifrt::Client::HostBufferSemantics host_buffer_semantics =
      ifrt::Client::HostBufferSemantics::kImmutableOnlyDuringCall;
#else
  PjRtClient::HostBufferSemantics host_buffer_semantics =
      PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall;
#endif
  std::function<void()> on_done_with_host_buffer;
  if (options.allow_zero_copy) {
    std::shared_ptr<PythonRefManager::ManagedPyObjects> py_buffer_ref =
        GlobalPyRefManager()->ManageReference(std::move(array));
    on_done_with_host_buffer =
        [py_buffer_ref{
            std::move(py_buffer_ref)}]() { /* keeps py_buffer_ref alive */ };
#ifdef JAX_ENABLE_IFRT
    host_buffer_semantics = ifrt::Client::HostBufferSemantics::kZeroCopy;
#else
    host_buffer_semantics = PjRtClient::HostBufferSemantics::kZeroCopy;
#endif
  }
  // Must release the GIL before BufferFromHostBuffer because backends may
  // decide to block/sleep for device buffer allocation.
  py::gil_scoped_release gil_release;
#ifdef JAX_ENABLE_IFRT
  TF_ASSIGN_OR_RETURN(auto ifrt_dtype, xla::ifrt::ToDType(squashed_type));
  TF_ASSIGN_OR_RETURN(
      auto ifrt_array,
      client->MakeArrayFromHostBuffer(
          data, ifrt_dtype, ifrt::Shape(dims), byte_strides,
          xla::ifrt::SingleDeviceSharding::Create(to_device),
          host_buffer_semantics, std::move(on_done_with_host_buffer)));
  return DevicePutResult(std::move(ifrt_array), /*weak_type=*/false);
#else
  TF_ASSIGN_OR_RETURN(
      auto buffer,
      to_device->client()->BufferFromHostBuffer(
          data, squashed_type, dims, byte_strides, host_buffer_semantics,
          std::move(on_done_with_host_buffer), to_device));
  return DevicePutResult(std::move(buffer), /*weak_type=*/false);
#endif
}

StatusOr<DevicePutResult> PyBufferHelper(py::handle obj, py::handle py_buffer,
                                         PyBuffer* buffer,
                                         PjRtDevice* to_device) {
  bool weak_type = buffer->weak_type()
                       ? *buffer->weak_type()
                       : py::cast<bool>(obj.attr("aval").attr("weak_type"));
#ifdef JAX_ENABLE_IFRT
  if (buffer->ifrt_array()->sharding().devices().front() == to_device) {
    return DevicePutResult(
        buffer->ifrt_array(), weak_type,
        /*owning_pybuffer=*/py::reinterpret_borrow<py::object>(py_buffer));
  } else {
    TF_ASSIGN_OR_RETURN(std::unique_ptr<ifrt::Array> copied_ifrt_array,
                        buffer->ifrt_array()->Reshard(
                            ifrt::SingleDeviceSharding::Create(to_device),
                            ifrt::ArrayCopySemantics::kReuseInput));
    return DevicePutResult(std::move(copied_ifrt_array), weak_type);
  }
#else
  if (buffer->pjrt_buffer()->device() == to_device) {
    return DevicePutResult(
        buffer->pjrt_buffer(), weak_type,
        /*owning_pybuffer=*/py::reinterpret_borrow<py::object>(py_buffer));
  } else {
    TF_ASSIGN_OR_RETURN(std::unique_ptr<PjRtBuffer> copied_buffer,
                        buffer->pjrt_buffer()->CopyToDevice(to_device));
    return DevicePutResult(std::move(copied_buffer), weak_type);
  }
#endif
}

StatusOr<DevicePutResult> HandlePyBuffer(py::handle obj,
#ifdef JAX_ENABLE_IFRT
                                         ifrt::Client* client,
                                         ifrt::Device* to_device,
#else
                                         PjRtDevice* to_device,
#endif
                                         const DevicePutOptions& options) {
  return PyBufferHelper(obj, obj, PyBuffer::AsPyBufferUnchecked(obj),
                        to_device);
}

StatusOr<DevicePutResult> HandlePyArray(py::handle obj,
#ifdef JAX_ENABLE_IFRT
                                        ifrt::Client* client,
                                        ifrt::Device* to_device,
#else
                                        PjRtDevice* to_device,
#endif
                                        const DevicePutOptions& options) {
  auto py_array = py::reinterpret_borrow<PyArray>(obj);

  // We only allow single device case for PyArray in device put.
  if (py_array.num_shards() != 1) {
    return InvalidArgument(
        "Only single-sharded Array is expected in device_put.");
  }

  if (py_array.sharding().get_type() == jax::PmapSharding::type()) {
    // We are only handling single device case for PmapSharding here. For other
    // cases, it fallbacks to python.
#ifdef JAX_ENABLE_IFRT
    return HandleNumpyArray(obj.attr("_value"), client, to_device, options);
#else
    return HandleNumpyArray(obj.attr("_value"), to_device, options);
#endif
  }

#ifdef JAX_ENABLE_IFRT
  ifrt::Array* ifrt_array = py_array.ifrt_array();
  if (ifrt_array == nullptr) {
    return InvalidArgument("Array has been deleted.");
  }
  if (ifrt_array->sharding().devices().front() == to_device) {
    return DevicePutResult(
        ifrt_array, py_array.weak_type(),
        /*owning_pybuffer=*/py::reinterpret_borrow<py::object>(obj));
  } else {
    TF_ASSIGN_OR_RETURN(
        std::unique_ptr<ifrt::Array> copied_ifrt_array,
        ifrt_array->Reshard(ifrt::SingleDeviceSharding::Create(to_device),
                            ifrt::ArrayCopySemantics::kReuseInput));
    return DevicePutResult(std::move(copied_ifrt_array), py_array.weak_type());
  }
#else
  PjRtBuffer* buffer = py_array.pjrt_buffer(0);
  if (buffer->device() == to_device) {
    return DevicePutResult(
        buffer, py_array.weak_type(),
        /*owning_pybuffer=*/py::reinterpret_borrow<py::object>(obj));
  } else {
    TF_ASSIGN_OR_RETURN(std::unique_ptr<PjRtBuffer> copied_buffer,
                        buffer->CopyToDevice(to_device));
    return DevicePutResult(std::move(copied_buffer), py_array.weak_type());
  }
#endif
}

StatusOr<DevicePutResult> HandleDeviceArray(py::handle obj,
#ifdef JAX_ENABLE_IFRT
                                            ifrt::Client* client,
                                            ifrt::Device* to_device,
#else
                                            PjRtDevice* to_device,
#endif
                                            const DevicePutOptions& options) {
  // Handle Python DeviceArray objects provided they have a .device_buffer field
  // Otherwise, fallback to handling as a NumPy array, since we do not
  // understand how to get a buffer object out. For example, ShardedDeviceArray
  // in JAX is handled by this path.
  py::object buffer = py::getattr(obj, "device_buffer", py::none());
  if (buffer.is_none()) {
#ifdef JAX_ENABLE_IFRT
    return HandleNumpyArray(obj, client, to_device, options);
#else
    return HandleNumpyArray(obj, to_device, options);
#endif
  }

  return PyBufferHelper(obj, buffer, py::cast<PyBuffer*>(buffer), to_device);
}

}  // namespace

StatusOr<DevicePutResult> DevicePut(py::handle arg,
#ifdef JAX_ENABLE_IFRT
                                    ifrt::Client* client,
                                    ifrt::Device* to_device,
#else
                                    PjRtDevice* to_device,
#endif
                                    const DevicePutOptions& options) {
  tsl::profiler::TraceMe traceme("DevicePut");
  static const absl::flat_hash_map<PyObject*, DevicePutFunc>* const handlers =
      [] {
        auto p = new absl::flat_hash_map<PyObject*, DevicePutFunc>();
        const NumpyScalarTypes& dtypes = GetNumpyScalarTypes();
        // Python scalar types.
        static_assert(sizeof(bool) == 1,
                      "Conversion code assumes bool is 1 byte");
        (*p)[reinterpret_cast<PyObject*>(&PyBool_Type)] =
            HandlePythonScalar<bool, bool>;
        (*p)[reinterpret_cast<PyObject*>(&PyLong_Type)] = HandlePythonInt;
        (*p)[reinterpret_cast<PyObject*>(&PyFloat_Type)] =
            HandlePythonScalar<double, float>;
        (*p)[reinterpret_cast<PyObject*>(&PyComplex_Type)] =
            HandlePythonScalar<complex128, complex64>;

        // Generic subclasses of DeviceArray, e.g., ShardedDeviceArray.
        (*p)[PyBuffer::base_type()] = HandleDeviceArray;

        try {
          py::object xla_module = py::module::import("jax.interpreters.xla");
          py::object device_array =
              py::getattr(xla_module, "_DeviceArray", py::none());
          if (!device_array.is_none()) {
            (*p)[device_array.ptr()] = HandleDeviceArray;
          }
        } catch (const py::error_already_set& e) {
          // Ignore; jax may not be present.
        }

        try {
          py::object pxla_module = py::module::import("jax.interpreters.pxla");
          py::object sda =
              py::getattr(pxla_module, "ShardedDeviceArray", py::none());
          if (!sda.is_none()) {
            (*p)[sda.ptr()] = HandleDeviceArray;
          }
        } catch (const py::error_already_set& e) {
          // Ignore; jax may not be present.
        }

        const auto numpy = py::module::import("numpy");
        (*p)[numpy.attr("ndarray").ptr()] = HandleNumpyArray;

        // Numpy scalar types. For some of them, we share the handler with
        // Python types (np_int64, np_float64, np_complex128).
        (*p)[dtypes.np_bool.ptr()] = HandleNumpyScalar<bool>;
        (*p)[dtypes.np_int8.ptr()] = HandleNumpyScalar<int8_t>;
        (*p)[dtypes.np_int16.ptr()] = HandleNumpyScalar<int16_t>;
        (*p)[dtypes.np_int32.ptr()] = HandleNumpyScalar<int32_t>;
        (*p)[dtypes.np_int64.ptr()] = HandleNumpyScalar<int64_t, int32_t>;
        (*p)[dtypes.np_uint8.ptr()] = HandleNumpyScalar<uint8_t>;
        (*p)[dtypes.np_uint16.ptr()] = HandleNumpyScalar<uint16_t>;
        (*p)[dtypes.np_uint32.ptr()] = HandleNumpyScalar<uint32_t>;
        (*p)[dtypes.np_uint64.ptr()] = HandleNumpyScalar<uint64_t, uint32_t>;
        (*p)[dtypes.np_bfloat16.ptr()] = HandleNumpyScalar<bfloat16>;
        (*p)[dtypes.np_float16.ptr()] = HandleNumpyScalar<half>;
        (*p)[dtypes.np_float32.ptr()] = HandleNumpyScalar<float>;
        (*p)[dtypes.np_float64.ptr()] = HandleNumpyScalar<double, float>;
        (*p)[dtypes.np_complex64.ptr()] = HandleNumpyScalar<complex64>;
        (*p)[dtypes.np_complex128.ptr()] =
            HandleNumpyScalar<complex128, complex64>;
        static_assert(sizeof(long long) == sizeof(int64_t),  // NOLINT
                      "long long must be the same size as int64_t");
        (*p)[dtypes.np_longlong.ptr()] = HandleNumpyScalar<int64_t, int32_t>;
        static_assert(sizeof(int) == sizeof(int32_t),
                      "int must be the same size as int32_t");
        (*p)[dtypes.np_intc.ptr()] = HandleNumpyScalar<int32_t>;

        return p;
      }();

  if (arg.get_type() == PyArray::type()) {
    auto array = py::reinterpret_borrow<PyArray>(arg);
    if (array.fastpath_enabled()) {
#ifdef JAX_ENABLE_IFRT
      return HandlePyArray(arg, client, to_device, options);
#else
      return HandlePyArray(arg, to_device, options);
#endif
    }
  }

  // Fast-path for the most common case of PyBuffer.
  if (arg.get_type().ptr() == PyBuffer::type()) {
#ifdef JAX_ENABLE_IFRT
    return HandlePyBuffer(arg, client, to_device, options);
#else
    return HandlePyBuffer(arg, to_device, options);
#endif
  }

  auto res = handlers->find(arg.get_type().ptr());
  if (res == handlers->end()) {
    for (auto base_class : arg.get_type().attr("__mro__")) {
      res = handlers->find(base_class.ptr());
      if (res != handlers->end()) {
#ifdef JAX_ENABLE_IFRT
        return res->second(arg, client, to_device, options);
#else
        return res->second(arg, to_device, options);
#endif
      }
    }
    return InvalidArgument(
        "%s", absl::StrCat(
                  "Not supported: The C++ jax jit execution path, only accepts "
                  "DeviceArray, Numpy arrays scalars of supported types "
                  "(see implementation), or Python scalars. Got type ",
                  py::cast<std::string>(py::str(arg.get_type()))));
  }
#ifdef JAX_ENABLE_IFRT
  return res->second(arg, client, to_device, options);
#else
  return res->second(arg, to_device, options);
#endif
}

bool IsFloat0(py::array arg) {
  static const auto* dtypes_module =
      new py::module(py::module::import("jax.dtypes"));
  static const auto* float0_dtype =
      new py::handle(dtypes_module->attr("float0"));
  return float0_dtype->is(arg.attr("dtype"));
}

std::string PyArgSignature::DebugString() const {
  std::string result = "";
  if (weak_type) {
    absl::StrAppend(&result, "weak_");
  }
  absl::StrAppend(&result, xla::PrimitiveType_Name(dtype));
  absl::StrAppend(&result, "[", absl::StrJoin(shape, ","), "]");
  return result;
}

using ToPyArgSignatureHandler =
    std::function<StatusOr<PyArgSignature>(py::handle, bool)>;

StatusOr<PyArgSignature> PyArgSignatureOfValue(py::handle arg,
                                               bool jax_enable_x64) {
  static const absl::flat_hash_map<PyObject*, ToPyArgSignatureHandler>* const
      handlers = [] {
        auto p = new absl::flat_hash_map<PyObject*, ToPyArgSignatureHandler>();

        const NumpyScalarTypes& dtypes = GetNumpyScalarTypes();

        // The 4 Python native types.
        ToPyArgSignatureHandler bool_handler =
            [](py::handle, bool) -> StatusOr<PyArgSignature> {
          return PyArgSignature(PrimitiveType::PRED, {}, true);
        };
        ToPyArgSignatureHandler int_handler =
            [](py::handle h, bool jax_enable_x64) -> StatusOr<PyArgSignature> {
          // TODO(phawkins): we should consider checking for integer overflow.
          if (jax_enable_x64) {
            return PyArgSignature(PrimitiveType::S64, {}, true);
          } else {
            return PyArgSignature(PrimitiveType::S32, {}, true);
          }
        };
        ToPyArgSignatureHandler float_handler =
            [&dtypes](py::handle h,
                      bool jax_enable_x64) -> StatusOr<PyArgSignature> {
          // Only Python native types has a True weak_type.
          bool weak_type = !py::isinstance(h, dtypes.np_float64);
          if (jax_enable_x64) {
            return PyArgSignature(PrimitiveType::F64, {}, weak_type);
          } else {
            return PyArgSignature(PrimitiveType::F32, {}, weak_type);
          }
        };
        ToPyArgSignatureHandler complex_handler =
            [&dtypes](py::handle h,
                      bool jax_enable_x64) -> StatusOr<PyArgSignature> {
          // Note that this branch is also taken  for np.complex128:
          // isinstance(np.complex128(3), complex) returns True
          // isinstance(np.complex64(3), complex) returns False
          bool weak_type = !py::isinstance(h, dtypes.np_complex128);
          if (jax_enable_x64) {
            return PyArgSignature(PrimitiveType::C128, {}, weak_type);
          } else {
            return PyArgSignature(PrimitiveType::C64, {}, weak_type);
          }
        };

        (*p)[reinterpret_cast<PyObject*>(&PyBool_Type)] = bool_handler;
        (*p)[reinterpret_cast<PyObject*>(&PyLong_Type)] = int_handler;
        (*p)[reinterpret_cast<PyObject*>(&PyFloat_Type)] = float_handler;
        (*p)[reinterpret_cast<PyObject*>(&PyComplex_Type)] = complex_handler;

        // The Buffer types except for fast-path PyBuffer.
        ToPyArgSignatureHandler device_array_handler =
            [](py::handle h, bool jax_enable_x64) -> StatusOr<PyArgSignature> {
          py::handle aval = h.attr("aval");
          TF_ASSIGN_OR_RETURN(auto dtype,
                              DtypeToPrimitiveType(aval.attr("dtype")));
          return PyArgSignature(
              dtype, py::cast<std::vector<int64_t>>(aval.attr("shape")),
              py::cast<py::bool_>(aval.attr("weak_type")));
        };
        (*p)[PyBuffer::base_type()] = device_array_handler;

        try {
          py::object xla_module = py::module::import("jax.interpreters.xla");
          py::object device_array =
              py::getattr(xla_module, "_DeviceArray", py::none());
          if (!device_array.is_none()) {
            (*p)[device_array.ptr()] = device_array_handler;
          }
        } catch (const py::error_already_set& e) {
          // Ignore; jax may not be present.
        }

        try {
          py::object pxla_module = py::module::import("jax.interpreters.pxla");
          py::object sda =
              py::getattr(pxla_module, "ShardedDeviceArray", py::none());
          if (!sda.is_none()) {
            (*p)[sda.ptr()] = device_array_handler;
          }
        } catch (const py::error_already_set& e) {
          // Ignore; jax may not be present.
        }

        ToPyArgSignatureHandler numpy_handler =
            [](py::handle h, bool jax_enable_x64) -> StatusOr<PyArgSignature> {
          py::array numpy_array = py::cast<py::array>(h);
          TF_ASSIGN_OR_RETURN(PrimitiveType dtype,
                              DtypeToPrimitiveType(numpy_array.dtype()));
          if (!jax_enable_x64) {
            dtype = Squash64BitTypes(dtype);
          }
          // We use reinterpret_cast<> to defend against environments where
          // ssize_t may not be precisely the same type as int64_t, even if it
          // is the same size (long vs long long).
          static_assert(sizeof(int64_t) == sizeof(ssize_t),
                        "Code assumes ssize_t is the same as int64_t");
          return PyArgSignature(
              dtype,
              absl::MakeConstSpan(
                  reinterpret_cast<const int64_t*>(numpy_array.shape()),
                  numpy_array.ndim()),
              /*weak_type=*/false);
        };
        const auto numpy = py::module::import("numpy");
        (*p)[numpy.attr("ndarray").ptr()] = numpy_handler;

        ToPyArgSignatureHandler np_uint64_handler =
            [](py::handle h, bool jax_enable_x64) -> StatusOr<PyArgSignature> {
          if (jax_enable_x64) {
            return PyArgSignature(PrimitiveType::U64, {}, /*weak_type=*/false);
          } else {
            return PyArgSignature(PrimitiveType::U32, {}, /*weak_type=*/false);
          }
        };
        ToPyArgSignatureHandler np_int_handler =
            [](py::handle h, bool jax_enable_x64) -> StatusOr<PyArgSignature> {
          if (jax_enable_x64) {
            return PyArgSignature(PrimitiveType::S64, {}, /*weak_type=*/false);
          } else {
            return PyArgSignature(PrimitiveType::S32, {}, /*weak_type=*/false);
          }
        };
        ToPyArgSignatureHandler numpy_array_handler =
            [](py::handle h, bool jax_enable_x64) -> StatusOr<PyArgSignature> {
          // This block deals with all numpy scalar types, except for int64_dt,
          // float64_dt and complex128_dt which are taken care of in previous if
          // blocks.
          TF_ASSIGN_OR_RETURN(auto dtype,
                              DtypeToPrimitiveType(h.attr("dtype")));
          return PyArgSignature(dtype, {}, /*weak_type=*/false);
        };

        // This block deals with all numpy scalar types, except for int64_dt,
        // float64_dt and complex128_dt which are taken care of in previous if
        // blocks.
        (*p)[dtypes.np_bool.ptr()] = numpy_array_handler;
        (*p)[dtypes.np_int8.ptr()] = numpy_array_handler;
        (*p)[dtypes.np_int16.ptr()] = numpy_array_handler;
        (*p)[dtypes.np_int32.ptr()] = numpy_array_handler;
        (*p)[dtypes.np_int64.ptr()] = np_int_handler;
        (*p)[dtypes.np_uint8.ptr()] = numpy_array_handler;
        (*p)[dtypes.np_uint16.ptr()] = numpy_array_handler;
        (*p)[dtypes.np_uint32.ptr()] = numpy_array_handler;
        (*p)[dtypes.np_uint64.ptr()] = np_uint64_handler;
        (*p)[dtypes.np_float16.ptr()] = numpy_array_handler;
        (*p)[dtypes.np_bfloat16.ptr()] = numpy_array_handler;
        (*p)[dtypes.np_float32.ptr()] = numpy_array_handler;
        (*p)[dtypes.np_float64.ptr()] = float_handler;
        (*p)[dtypes.np_complex64.ptr()] = numpy_array_handler;
        (*p)[dtypes.np_complex128.ptr()] = complex_handler;
        (*p)[dtypes.np_longlong.ptr()] = np_int_handler;
        (*p)[dtypes.np_intc.ptr()] = numpy_array_handler;

        return p;
      }();

  if (arg.get_type() == PyArray::type()) {
    auto array = py::reinterpret_borrow<PyArray>(arg);
    if (array.fastpath_enabled()) {
#ifdef JAX_ENABLE_IFRT
      ifrt::Array* ifrt_array = array.ifrt_array();
      if (ifrt_array == nullptr) {
        return xla::InvalidArgument("Array has been deleted.");
      }
      TF_ASSIGN_OR_RETURN(auto primitive_type,
                          ifrt::ToPrimitiveType(ifrt_array->dtype()));
      return PyArgSignature(primitive_type, array.shape(), array.weak_type());
#else
      if (array.IsDeleted()) {
        return xla::InvalidArgument("Array has been deleted.");
      }
      auto dtype = array.pjrt_buffer(0)->on_device_shape().element_type();
      return PyArgSignature(dtype, array.shape(), array.weak_type());
#endif
    }
  }

  // Fast-path for the most common case of PyBuffer.
  if (arg.get_type().ptr() == PyBuffer::type()) {
    TF_ASSIGN_OR_RETURN(PyBuffer * buffer, PyBuffer::AsPyBuffer(arg));
    bool weak_type = buffer->weak_type().has_value()
                         ? *buffer->weak_type()
                         : py::cast<bool>(arg.attr("aval").attr("weak_type"));
#ifdef JAX_ENABLE_IFRT
    TF_ASSIGN_OR_RETURN(auto primitive_type,
                        ifrt::ToPrimitiveType(buffer->ifrt_array()->dtype()));
    return PyArgSignature(primitive_type, buffer->ifrt_array()->shape().dims(),
                          weak_type);
#else
    return PyArgSignature(
        buffer->pjrt_buffer()->on_device_shape().element_type(),
        buffer->pjrt_buffer()->on_device_shape().dimensions(), weak_type);
#endif
  }

  // Fast-path for ShardedDeviceArray.
  if (jax::ShardedDeviceArray::IsShardedDeviceArray(arg)) {
    jax::ShardedDeviceArray* sda =
        jax::ShardedDeviceArray::AsShardedDeviceArrayUnchecked(arg);

    // TODO(jblespiau): See if we can be faster not accessing the aval attribute
    // and storing these directly.
    py::handle aval = arg.attr("aval");
    TF_ASSIGN_OR_RETURN(auto dtype, DtypeToPrimitiveType(aval.attr("dtype")));
    return PyArgSignature(dtype,
                          py::cast<std::vector<int64_t>>(aval.attr("shape")),
                          sda->weak_type());
  }

  auto res = handlers->find(arg.get_type().ptr());
  if (res == handlers->end()) {
    // We attempt to look at the MRO classes
    for (auto base_class : arg.get_type().attr("__mro__")) {
      res = handlers->find(base_class.ptr());
      if (res != handlers->end()) {
        return res->second(arg, jax_enable_x64);
      }
    }
    return InvalidArgument(
        "%s",
        absl::StrCat("Not supported: The C++ ToPyArgSignature only accepts "
                     "Buffer/DeviceArray/ShardedDeviceArray, Numpy "
                     "arrays scalars of supported types "
                     "(see implementation), or Python scalars. Got type ",
                     py::cast<std::string>(py::str(arg.get_type()))));
  }
  return res->second(arg, jax_enable_x64);
}

}  // namespace xla
