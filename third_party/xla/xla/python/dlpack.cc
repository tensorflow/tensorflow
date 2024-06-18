/* Copyright 2020 The OpenXLA Authors.

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

#include "xla/python/dlpack.h"

#include <Python.h>

#include <cstdint>
#include <functional>
#include <memory>
#include <numeric>
#include <optional>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "include/dlpack/dlpack.h"  // from @dlpack
#include "llvm/Support/Casting.h"
#include "third_party/nanobind/include/nanobind/nanobind.h"
#include "xla/layout.h"
#include "xla/pjrt/exceptions.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_common.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_layout.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/nb_class_ptr.h"
#include "xla/python/pjrt_ifrt/pjrt_array.h"
#include "xla/python/pjrt_ifrt/pjrt_client.h"
#include "xla/python/pjrt_ifrt/pjrt_device.h"
#include "xla/python/py_array.h"
#include "xla/python/py_client.h"
#include "xla/python/python_ref_manager.h"
#include "xla/python/traceback.h"
#include "xla/python/types.h"
#include "xla/python/util.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/util.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/statusor.h"

namespace nb = nanobind;

namespace xla {
namespace {

const char* const kDlTensorCapsuleName = "dltensor";

struct DLPackTensor {
  ~DLPackTensor();

  // `buffer_reference` is populated if we have shared (read-only) access.
  nb::object buffer_reference;

  // `external_reference` is always populated.
  std::unique_ptr<PjRtBuffer::ExternalReference> external_reference;

  std::vector<int64_t> shape;
  std::vector<int64_t> strides;
  DLManagedTensor tensor;
};

DLPackTensor::~DLPackTensor() {
  if (buffer_reference) {
    GlobalPyRefManager()->AddGarbage(
        absl::MakeSpan(&buffer_reference, /*size=*/1));
  }
}

void DLPackTensorDeleter(DLManagedTensor* t) {
  if (t) {
    delete static_cast<DLPackTensor*>(t->manager_ctx);
  }
}

absl::StatusOr<DLDataType> PrimitiveTypeToDLDataType(PrimitiveType type) {
  switch (type) {
    case S8:
      return DLDataType{kDLInt, 8, 1};
    case S16:
      return DLDataType{kDLInt, 16, 1};
    case S32:
      return DLDataType{kDLInt, 32, 1};
    case S64:
      return DLDataType{kDLInt, 64, 1};
    case U8:
      return DLDataType{kDLUInt, 8, 1};
    case U16:
      return DLDataType{kDLUInt, 16, 1};
    case U32:
      return DLDataType{kDLUInt, 32, 1};
    case U64:
      return DLDataType{kDLUInt, 64, 1};
    case F16:
      return DLDataType{kDLFloat, 16, 1};
    case F32:
      return DLDataType{kDLFloat, 32, 1};
    case F64:
      return DLDataType{kDLFloat, 64, 1};
    case BF16:
      return DLDataType{kDLBfloat, 16, 1};
    case PRED:
      return DLDataType{kDLBool, 8, 1};
    case C64:
      return DLDataType{kDLComplex, 64, 1};
    case C128:
      return DLDataType{kDLComplex, 128, 1};
    default:
      return Unimplemented("XLA type %s has no DLPack equivalent",
                           PrimitiveType_Name(type));
  }
}

absl::StatusOr<PrimitiveType> DLDataTypeToPrimitiveType(DLDataType type) {
  if (type.lanes != 1) {
    return Unimplemented("DLPack types with lanes != 1 not implemented, got %d",
                         type.lanes);
  }
  switch (type.code) {
    case kDLBool:
      switch (type.bits) {
        case 8:
          return PRED;
        default:
          return Unimplemented(
              "Only 8-bit DLPack booleans are supported, got %d bits",
              type.bits);
      }
    case kDLInt:
      switch (type.bits) {
        case 8:
          return S8;
        case 16:
          return S16;
        case 32:
          return S32;
        case 64:
          return S64;
        default:
          return Unimplemented(
              "Invalid or unsupported DLPack integer width: %d bits",
              type.bits);
      }
    case kDLUInt:
      switch (type.bits) {
        case 8:
          return U8;
        case 16:
          return U16;
        case 32:
          return U32;
        case 64:
          return U64;
        default:
          return Unimplemented(
              "Invalid or unsupported DLPack unsigned integer width: %d bits",
              type.bits);
      }
    case kDLFloat:
      switch (type.bits) {
        case 16:
          return F16;
        case 32:
          return F32;
        case 64:
          return F64;
        default:
          return Unimplemented(
              "Invalid or unsupported DLPack float width: %d bits", type.bits);
      }
    case kDLBfloat:
      switch (type.bits) {
        case 16:
          return BF16;
        default:
          return Unimplemented(
              "Invalid or unsupported DLPack Bfloat width: %d bits", type.bits);
      }
    case kDLComplex:
      switch (type.bits) {
        case 64:
          return C64;
        case 128:
          return C128;
        default:
          return Unimplemented(
              "Invalid or unsupported DLPack complex width: %d bits",
              type.bits);
      }
    default:
      return Unimplemented("Unknown or invalid DLPack type code %d", type.code);
  }
}

absl::StatusOr<std::vector<int64_t>> StridesToLayout(
    absl::Span<int64_t const> dims, absl::Span<int64_t const> strides) {
  CHECK_EQ(dims.size(), strides.size());
  std::vector<int64_t> minor_to_major(dims.size());
  std::iota(minor_to_major.begin(), minor_to_major.end(), 0);
  absl::c_sort(minor_to_major, [&](int a, int b) {
    if (strides[a] < strides[b]) {
      return true;
    }
    if (strides[a] > strides[b]) {
      return false;
    }
    // If two dimensions have the same stride, prefer the major-to-minor
    // interpretation of the ordering, since that's what JAX wants.
    return b < a;
  });
  int64_t stride = 1;
  for (int64_t d : minor_to_major) {
    if (dims[d] > 1 && strides[d] != stride) {
      return Unimplemented(
          "Only DLPack tensors with trivial (compact) striding are supported; "
          "i.e., tensors whose striding represents a transposition of the "
          "underlying buffer but not broadcasting. Dimensions were: [%s], "
          "strides were [%s].",
          absl::StrJoin(dims, ","), absl::StrJoin(strides, ","));
    }
    stride *= dims[d];
  }
  return minor_to_major;
}

absl::StatusOr<DLDeviceType> DLDeviceTypeForDevice(const PjRtDevice& device) {
  if (device.client()->platform_id() == CpuId()) {
    return kDLCPU;
  } else if (device.client()->platform_id() == CudaId()) {
    return kDLCUDA;
  } else if (device.client()->platform_id() == RocmId()) {
    return kDLROCM;
  }
  return InvalidArgument("Device %s cannot be used as a DLPack device.",
                         device.DebugString());
}

absl::StatusOr<DLDevice> DLDeviceForDevice(const PjRtDevice& device) {
  DLDevice context;
  TF_ASSIGN_OR_RETURN(context.device_type, DLDeviceTypeForDevice(device));
  context.device_id = device.local_hardware_id();
  return context;
}

absl::StatusOr<PjRtDevice*> DeviceForDLDevice(const PjRtClient* cpu_client,
                                              const PjRtClient* gpu_client,
                                              const DLDevice& context) {
  switch (context.device_type) {
    case kDLCPU:
      if (cpu_client == nullptr) {
        return InvalidArgument(
            "DLPack tensor is on CPU, but no CPU backend was provided.");
      }
      TF_RET_CHECK(cpu_client->platform_id() == CpuId());
      return cpu_client->LookupAddressableDevice(
          xla::PjRtLocalDeviceId(context.device_id));
    case kDLCUDA:
      if (gpu_client == nullptr) {
        return InvalidArgument(
            "DLPack tensor is on GPU, but no GPU backend was provided.");
      }
      TF_RET_CHECK(gpu_client->platform_id() == CudaId());
      return gpu_client->LookupAddressableDevice(
          xla::PjRtLocalDeviceId(context.device_id));
    case kDLROCM:
      if (gpu_client == nullptr) {
        return InvalidArgument(
            "DLPack tensor is on GPU, but no GPU backend was provided.");
      }
      TF_RET_CHECK(gpu_client->platform_id() == RocmId());
      return gpu_client->LookupAddressableDevice(
          xla::PjRtLocalDeviceId(context.device_id));
    default:
      return InvalidArgument("Unknown/unsupported DLPack device type %d",
                             context.device_type);
  }
}

}  // namespace

absl::StatusOr<nb::capsule> BufferToDLPackManagedTensor(
    nb::handle py_buffer, std::optional<std::intptr_t> stream) {
  ifrt::Array* ifrt_array = nb::cast<xla::PyArray>(py_buffer).ifrt_array();
  if (ifrt_array == nullptr) {
    return Unimplemented(
        "BufferToDLPackManagedTensor called on deleted array.");
  }
  auto* arr = llvm::dyn_cast_or_null<ifrt::PjRtCompatibleArray>(ifrt_array);
  if (arr == nullptr) {
    throw XlaRuntimeError(
        "This operation is implemented for a PjRt-compatible backend only.");
  }
  PjRtBuffer* pjrt_buffer = arr->pjrt_buffers().front().get();

  if (pjrt_buffer->IsTuple()) {
    return Unimplemented(
        "BufferToDLPackManagedTensor is not implemented for tuple "
        "buffers.");
  }
  if (pjrt_buffer->has_dynamic_dimensions()) {
    return Unimplemented("DynamicShape is not implemented in DLPack.");
  }

  auto pack = std::make_unique<DLPackTensor>();
  DLTensor& dt = pack->tensor.dl_tensor;
  {
    // AcquireExternalReference may block; there are no API guarantees.
    GlobalPyRefManager()->CollectGarbage();
    nb::gil_scoped_release gil_release;
    TF_ASSIGN_OR_RETURN(pack->external_reference,
                        pjrt_buffer->AcquireExternalReference());
    if (stream) {
      TF_RETURN_IF_ERROR(
          pack->external_reference->WaitUntilBufferReadyOnStream(*stream));
    } else {
      TF_RETURN_IF_ERROR(
          AwaitBuffersReady(absl::MakeConstSpan(&ifrt_array, 1)));
    }
  }
  pack->buffer_reference = nb::borrow<nb::object>(py_buffer);

  dt.data = pack->external_reference->OpaqueDeviceMemoryDataPointer();
  pack->tensor.manager_ctx = pack.get();
  pack->tensor.deleter = DLPackTensorDeleter;
  TF_ASSIGN_OR_RETURN(dt.device, DLDeviceForDevice(*pjrt_buffer->device()));
  dt.device.device_id = pjrt_buffer->device()->local_hardware_id();
  dt.ndim = pjrt_buffer->dimensions().size();
  TF_ASSIGN_OR_RETURN(dt.dtype,
                      PrimitiveTypeToDLDataType(pjrt_buffer->element_type()));

  pack->shape = std::vector<int64_t>(pjrt_buffer->dimensions().begin(),
                                     pjrt_buffer->dimensions().end());

  // TODO(b/327524065): use PjRtLayout directly instead of xla::Layout
  Layout xla_layout = GetXlaLayoutUnsafe(pjrt_buffer->layout());
  pack->strides = StridesForShape(pjrt_buffer->element_type(),
                                  pjrt_buffer->dimensions(), xla_layout);

  dt.shape = reinterpret_cast<std::int64_t*>(pack->shape.data());
  dt.strides = reinterpret_cast<std::int64_t*>(pack->strides.data());
  dt.byte_offset = 0;

  // We cannot use nanobind's capsule object constructor because we need to
  // detect if the capsule name has been changed in the deleter, but nanobind
  // hides the underlying Python object from the deleter.
  nb::capsule capsule = nb::steal<nb::capsule>(
      PyCapsule_New(&pack.release()->tensor, kDlTensorCapsuleName,
                    [](PyObject* obj) noexcept {
                      DLManagedTensor* dlmt = static_cast<DLManagedTensor*>(
                          PyCapsule_GetPointer(obj, kDlTensorCapsuleName));
                      if (dlmt) {
                        DLPackTensorDeleter(dlmt);
                      } else {
                        // The tensor has been deleted. Clear any error from
                        // PyCapsule_GetPointer.
                        PyErr_Clear();
                      }
                    }));
  if (!capsule.ptr()) {
    throw nb::python_error();
  }
  return capsule;
}

absl::StatusOr<nb::object> DLPackManagedTensorToBuffer(
    const nb::capsule& tensor, std::optional<nb_class_ptr<PyClient>> cpu_client,
    std::optional<nb_class_ptr<PyClient>> gpu_client) {
  // TODO(hyeontaek): This is a potential target for an IFRT client to multiplex
  // multiple PjRt clients. Devices from these PjRt clients could be expressed
  // as a unified set of IFRT devices.
  auto* cpu_pjrt_client = cpu_client ? (*cpu_client)->pjrt_client() : nullptr;
  auto* gpu_pjrt_client = gpu_client ? (*gpu_client)->pjrt_client() : nullptr;

  if (std::string_view(tensor.name()) != kDlTensorCapsuleName) {
    return InvalidArgument(
        "DLPack tensor must be a capsule with name \"dltensor\", got \"%s\". "
        "Note that a DLPack tensor may be consumed at most once.",
        std::string_view(tensor.name()));
  }
  DLManagedTensor* dlmt = static_cast<DLManagedTensor*>(tensor.data());
  if (dlmt->dl_tensor.ndim < 0) {
    return InvalidArgument(
        "Number of dimensions in DLManagedTensor must be nonnegative, got %d",
        dlmt->dl_tensor.ndim);
  }
  TF_ASSIGN_OR_RETURN(PjRtDevice * device,
                      DeviceForDLDevice(cpu_client ? cpu_pjrt_client : nullptr,
                                        gpu_client ? gpu_pjrt_client : nullptr,
                                        dlmt->dl_tensor.device));
  absl::Span<int64_t const> dimensions(
      reinterpret_cast<int64_t*>(dlmt->dl_tensor.shape), dlmt->dl_tensor.ndim);
  TF_ASSIGN_OR_RETURN(PrimitiveType element_type,
                      DLDataTypeToPrimitiveType(dlmt->dl_tensor.dtype));

  std::vector<int64_t> minor_to_major;
  if (dlmt->dl_tensor.strides &&
      absl::c_find(dimensions, 0) == dimensions.end()) {
    absl::Span<int64_t const> strides(
        reinterpret_cast<int64_t*>(dlmt->dl_tensor.strides),
        dlmt->dl_tensor.ndim);
    TF_ASSIGN_OR_RETURN(minor_to_major, StridesToLayout(dimensions, strides));
  } else {
    minor_to_major.resize(dlmt->dl_tensor.ndim);
    std::iota(minor_to_major.rbegin(), minor_to_major.rend(), 0);
  }
  Shape shape = ShapeUtil::MakeShapeWithDenseLayout(element_type, dimensions,
                                                    minor_to_major);

  // Raise an error if the resulting PjRtBuffer would have a non-default layout.
  // TODO(skyewm): we do this because JAX doesn't currently have good support
  // for non-default layouts, and will return wrong results if a non-default
  // layout is passed to a computation expecting default layouts. Remove this
  // special case when non-default layouts are better supported by JAX.
  TF_ASSIGN_OR_RETURN(Layout default_layout, device->client()->GetDefaultLayout(
                                                 element_type, dimensions));
  if (shape.layout() != default_layout) {
    return Unimplemented(
        "from_dlpack got array with non-default layout with minor-to-major "
        "dimensions (%s), expected (%s)",
        absl::StrJoin(shape.layout().minor_to_major(), ","),
        absl::StrJoin(default_layout.minor_to_major(), ","));
  }

  std::function<void()> on_delete_callback;
  if (dlmt->deleter) {
    on_delete_callback = [dlmt]() { dlmt->deleter(dlmt); };
  }
  TF_ASSIGN_OR_RETURN(auto pjrt_buffer,
                      device->client()->CreateViewOfDeviceBuffer(
                          static_cast<char*>(dlmt->dl_tensor.data) +
                              dlmt->dl_tensor.byte_offset,
                          shape, device, on_delete_callback));
  // We have taken ownership of the array inside the capsule; make sure the
  // capsule it cannot be used again.
  PyCapsule_SetName(tensor.ptr(), "used_dltensor");
  PyCapsule_SetDestructor(tensor.ptr(), nullptr);
  // TODO(phawkins): simplify the expression below once we know cpu_client is
  // always non-null.
  auto client = (cpu_client && device->client() == cpu_pjrt_client)
                    ? std::move(*cpu_client)
                    : std::move(*gpu_client);
  auto* ifrt_client =
      llvm::dyn_cast_or_null<ifrt::PjRtCompatibleClient>(client->ifrt_client());
  if (ifrt_client == nullptr) {
    throw XlaRuntimeError(
        "This operation is implemented for a PjRt-compatible backend only.");
  }
  TF_ASSIGN_OR_RETURN(auto ifrt_array,
                      ifrt_client->CreatePjRtArray(std::move(pjrt_buffer)));
  return PyArray::MakeFromSingleDeviceArray(std::move(client), Traceback::Get(),
                                            std::move(ifrt_array), false, true);
}

absl::StatusOr<nb::object> DLPackManagedTensorToBuffer(
    const nb::capsule& tensor, ifrt::Device* ifrt_device,
    nb_class_ptr<PyClient> client, std::optional<std::intptr_t> stream) {
  ifrt::PjRtDevice* device =
      llvm::dyn_cast_or_null<ifrt::PjRtDevice>(ifrt_device);
  if (device == nullptr) {
    throw XlaRuntimeError(
        "DLPack is supported for PjRt-compatible backends only.");
  }
  if (!device->IsAddressable()) {
    throw XlaRuntimeError(
        "DLPack is only supported for devices addressable by the current "
        "process.");
  }
  if (std::string_view(tensor.name()) != kDlTensorCapsuleName) {
    return InvalidArgument(
        "DLPack tensor must be a capsule with name \"dltensor\", got \"%s\". "
        "Note that a DLPack tensor may be consumed at most once.",
        std::string_view(tensor.name()));
  }
  DLManagedTensor* dlmt = static_cast<DLManagedTensor*>(tensor.data());
  if (dlmt->dl_tensor.ndim < 0) {
    return InvalidArgument(
        "Number of dimensions in DLManagedTensor must be nonnegative, got %d",
        dlmt->dl_tensor.ndim);
  }
  absl::Span<int64_t const> dimensions(
      reinterpret_cast<int64_t*>(dlmt->dl_tensor.shape), dlmt->dl_tensor.ndim);
  TF_ASSIGN_OR_RETURN(PrimitiveType element_type,
                      DLDataTypeToPrimitiveType(dlmt->dl_tensor.dtype));

  std::vector<int64_t> minor_to_major;
  if (dlmt->dl_tensor.strides &&
      absl::c_find(dimensions, 0) == dimensions.end()) {
    absl::Span<int64_t const> strides(
        reinterpret_cast<int64_t*>(dlmt->dl_tensor.strides),
        dlmt->dl_tensor.ndim);
    TF_ASSIGN_OR_RETURN(minor_to_major, StridesToLayout(dimensions, strides));
  } else {
    minor_to_major.resize(dlmt->dl_tensor.ndim);
    std::iota(minor_to_major.rbegin(), minor_to_major.rend(), 0);
  }
  Shape shape = ShapeUtil::MakeShapeWithDenseLayout(element_type, dimensions,
                                                    minor_to_major);

  std::function<void()> on_delete_callback;
  if (dlmt->deleter) {
    on_delete_callback = [dlmt]() { dlmt->deleter(dlmt); };
  }
  TF_ASSIGN_OR_RETURN(
      auto pjrt_buffer,
      device->pjrt_device()->client()->CreateViewOfDeviceBuffer(
          static_cast<char*>(dlmt->dl_tensor.data) +
              dlmt->dl_tensor.byte_offset,
          shape, device->pjrt_device(), on_delete_callback, stream));
  // We have taken ownership of the array inside the capsule; make sure the
  // capsule it cannot be used again.
  PyCapsule_SetName(tensor.ptr(), "used_dltensor");
  PyCapsule_SetDestructor(tensor.ptr(), nullptr);

  auto* ifrt_client =
      llvm::dyn_cast_or_null<ifrt::PjRtCompatibleClient>(client->ifrt_client());
  if (ifrt_client == nullptr) {
    throw XlaRuntimeError(
        "This operation is implemented for a PjRt-compatible backend only.");
  }
  TF_ASSIGN_OR_RETURN(auto ifrt_array,
                      ifrt_client->CreatePjRtArray(std::move(pjrt_buffer)));
  return PyArray::MakeFromSingleDeviceArray(std::move(client), Traceback::Get(),
                                            std::move(ifrt_array), false, true);
}

}  // namespace xla
