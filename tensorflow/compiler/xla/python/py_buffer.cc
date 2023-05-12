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

#include "tensorflow/compiler/xla/python/py_buffer.h"

#include <cstring>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/base/casts.h"
#include "pybind11/pybind11.h"  // from @pybind11
#include "pybind11/pytypes.h"  // from @pybind11
#include "tensorflow/compiler/xla/pjrt/pjrt_client.h"
#include "tensorflow/compiler/xla/python/ifrt/array.h"
#include "tensorflow/compiler/xla/python/ifrt/device.h"
#include "tensorflow/compiler/xla/python/pjrt_ifrt/pjrt_array.h"
#include "tensorflow/compiler/xla/python/py_client.h"
#include "tensorflow/compiler/xla/python/python_ref_manager.h"
#include "tensorflow/compiler/xla/python/python_utils.h"
#include "tensorflow/compiler/xla/python/status_casters.h"
#include "tensorflow/compiler/xla/python/transfer_guard_lib.h"
#include "tensorflow/compiler/xla/python/types.h"
#include "tensorflow/compiler/xla/python/util.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
namespace xla {

namespace py = pybind11;

namespace {

// Returns if shape has a major-to-minor layout.
bool HasMajorToMinorLayout(const xla::Shape& shape) {
  if (shape.has_layout()) {
    for (int i = 0; i < shape.layout().minor_to_major_size(); ++i) {
      if (shape.layout().minor_to_major(i) !=
          shape.layout().minor_to_major_size() - 1 - i) {
        return false;
      }
    }
  }
  return true;
}

// Returns byte_strides if shape has a non-major-to-minor layout.
std::optional<std::vector<int64_t>> ByteStridesOrDefaultForShapeInt64(
    const Shape& shape) {
  if (!shape.has_layout() || HasMajorToMinorLayout(shape)) {
    return std::nullopt;
  }
  return ByteStridesForShapeInt64(shape);
}

}  // namespace

/* static */ PjRtBuffer* IfrtHelpers::pjrt_buffer(ifrt::Array* ifrt_array) {
  auto* arr = llvm::dyn_cast_or_null<ifrt::PjRtCompatibleArray>(ifrt_array);
  if (arr == nullptr) {
    throw XlaRuntimeError(
        "This operation is implemented for a PjRt-compatible backend only.");
  }
  return arr->pjrt_buffers().front().get();
}

/* static */ PjRtDevice* IfrtHelpers::pjrt_device(ifrt::Array* ifrt_array) {
  return ifrt_array->sharding().devices().front();
}

/* static */ StatusOr<const Shape*> IfrtHelpers::xla_dynamic_shape(
    ifrt::Array* ifrt_array, std::optional<Shape>& scratch) {
  auto* pjrt_buffer = IfrtHelpers::pjrt_buffer(ifrt_array);

  if (pjrt_buffer->on_device_shape().is_static()) {
    return &pjrt_buffer->on_device_shape();
  }
  // Python buffer protocol references shape data by pointer, therefore we must
  // store a valid copy of the shape.
  if (!scratch) {
    Shape dynamic_shape;
    {
      py::gil_scoped_release gil_release;
      TF_ASSIGN_OR_RETURN(dynamic_shape,
                          pjrt_buffer->logical_on_device_shape());
    }
    scratch = dynamic_shape;
  }
  return &scratch.value();
}

pybind11::tuple IfrtHelpers::python_shape(ifrt::Array* ifrt_array) {
  return SpanToTuple(ifrt_array->shape().dims());
}

pybind11::dtype IfrtHelpers::python_dtype(ifrt::Array* ifrt_array) {
  // TODO(hyeontaek): Support non-XLA types such as xla::ifrt::DType::kString.
  PrimitiveType primitive = ifrt::ToPrimitiveType(ifrt_array->dtype()).value();
  return PrimitiveTypeToDtype(primitive).value();
}

/* static */ StatusOr<tsl::RCReference<ifrt::Array>> IfrtHelpers::CopyToDevice(
    ifrt::Array* ifrt_array, PjRtDevice* dst_device) {
  CHECK(dst_device != nullptr);
  auto transfer_guard_formatter = [ifrt_array, dst_device] {
    auto shape = py::cast<std::string>(py::str(python_shape(ifrt_array)));
    auto dtype = py::cast<std::string>(py::str(python_dtype(ifrt_array)));
    return absl::StrCat("shape=", shape, ", dtype=", dtype,
                        ", device=", pjrt_device(ifrt_array)->DebugString(),
                        ", dst_device=", dst_device->DebugString());
  };
  TF_RETURN_IF_ERROR(
      jax::ApplyTransferGuardToDeviceToDevice(transfer_guard_formatter));

  GlobalPyRefManager()->CollectGarbage();
  py::gil_scoped_release gil_release;
  return ifrt_array->Reshard(ifrt::SingleDeviceSharding::Create(dst_device),
                             ifrt::ArrayCopySemantics::kReuseInput);
}

/* static */ StatusOr<pybind11::object> PyHostValue::AsNumPyArray(
    std::shared_ptr<PyHostValue>& host_value,
    std::optional<Shape>& dynamic_shape_holder, ifrt::Array* ifrt_array,
    pybind11::handle this_obj) {
  if (ifrt_array->IsDeleted()) {
    return InvalidArgument("DeviceArray has been deleted.");
  }
  auto* arr = llvm::dyn_cast_or_null<ifrt::PjRtCompatibleArray>(ifrt_array);
  if (arr != nullptr) {
    auto* pjrt_buffer = arr->pjrt_buffers().front().get();
    TF_RET_CHECK(pjrt_buffer->on_device_shape().IsArray());
    // On CPU, we can return the value in a zero-copy way.
    if (pjrt_buffer->IsOnCpu()) {
      TF_ASSIGN_OR_RETURN(
          const auto* shape,
          IfrtHelpers::xla_dynamic_shape(ifrt_array, dynamic_shape_holder));
      TF_ASSIGN_OR_RETURN(py::dtype dtype,
                          PrimitiveTypeToDtype(shape->element_type()));
      // Objects that must be kept alive while the array is alive.
      struct Hold {
        tsl::RCReference<ifrt::Array> buffer;
        std::unique_ptr<PjRtBuffer::ExternalReference> external_reference_hold;
      };
      auto hold = std::make_unique<Hold>();
      TF_ASSIGN_OR_RETURN(hold->external_reference_hold,
                          pjrt_buffer->AcquireExternalReference());
      hold->buffer = tsl::FormRef(ifrt_array);
      void* data =
          hold->external_reference_hold->OpaqueDeviceMemoryDataPointer();
      py::capsule hold_capsule(hold.release(),
                               [](void* h) { delete static_cast<Hold*>(h); });
      py::array array(dtype, shape->dimensions(), ByteStridesForShape(*shape),
                      data, hold_capsule);
      array.attr("flags").attr("writeable") = Py_False;
      {
        py::gil_scoped_release gil;
        TF_RETURN_IF_ERROR(ifrt_array->GetReadyFuture().Await());
      }
      return array;
    }
  }

  TF_RETURN_IF_ERROR(
      CopyToHostAsync(host_value, dynamic_shape_holder, ifrt_array));
  if (!host_value->ready.HasBeenNotified()) {
    py::gil_scoped_release gil;
    host_value->ready.WaitForNotification();
  }
  TF_RETURN_IF_ERROR(host_value->status);
  TF_ASSIGN_OR_RETURN(py::object array, LiteralToPython(host_value->value));
  array.attr("flags").attr("writeable") = Py_False;
  return array;
}

/* static */ Status PyHostValue::CopyToHostAsync(
    std::shared_ptr<PyHostValue>& host_value,
    std::optional<Shape>& dynamic_shape_holder, ifrt::Array* ifrt_array) {
  if (host_value) {
    return OkStatus();
  }
  auto* arr = llvm::dyn_cast_or_null<ifrt::PjRtCompatibleArray>(ifrt_array);
  if (arr != nullptr) {
    auto* pjrt_buffer = arr->pjrt_buffers().front().get();
    if (pjrt_buffer->IsOnCpu()) {
      return OkStatus();
    }
  }
  auto transfer_guard_formatter = [ifrt_array] {
    auto shape =
        py::cast<std::string>(py::str(IfrtHelpers::python_shape(ifrt_array)));
    auto dtype =
        py::cast<std::string>(py::str(IfrtHelpers::python_dtype(ifrt_array)));
    return absl::StrCat("shape=", shape, ", dtype=", dtype, ", device=",
                        IfrtHelpers::pjrt_device(ifrt_array)->DebugString());
  };
  TF_RETURN_IF_ERROR(
      jax::ApplyTransferGuardToDeviceToHost(transfer_guard_formatter));

  auto host_value_copy = std::make_shared<PyHostValue>();
  host_value = host_value_copy;
  // TODO(b/182461453): This is a blocking call. If we further implemented
  // populating dynamic shape metadata while fetching the literal, we wouldn't
  // need this static approach.
  const xla::Shape* dynamic_shape;
  std::optional<xla::Shape> shape_holder;
  if (llvm::isa<ifrt::PjRtCompatibleArray>(ifrt_array)) {
    TF_ASSIGN_OR_RETURN(dynamic_shape, IfrtHelpers::xla_dynamic_shape(
                                           ifrt_array, dynamic_shape_holder));
  } else {
    // Skip querying the dynamic shape for a non-PjRt Array.
    TF_ASSIGN_OR_RETURN(xla::PrimitiveType type,
                        ifrt::ToPrimitiveType(ifrt_array->dtype()));
    shape_holder = ShapeUtil::MakeShapeWithDescendingLayout(
        type, ifrt_array->shape().dims());
    dynamic_shape = &*shape_holder;
  }

  py::gil_scoped_release gil;
  xla::Shape host_shape = ShapeUtil::DeviceShapeToHostShape(*dynamic_shape);
  // TODO(hyeontaek): Several PjRt runtimes assume that the host buffer uses
  // the same transposition as the device buffer. This is different from
  // PjRtBuffer::ToLiteral()'s semantics that the runtime respects the layout
  // of the host buffer literal. On the other hand, the runtime often knows
  // better about an efficient layout for the host buffer. It will be useful
  // to revisit the semantics of PjRtBuffer::ToLiteral() to see if it is
  // desirable for the runtime to choose the layout.
  host_value_copy->value = std::make_shared<Literal>(host_shape);
  ifrt::Future<Status> copy_future = ifrt_array->CopyToHostBuffer(
      host_value_copy->value->untyped_data(),
      ByteStridesOrDefaultForShapeInt64(host_shape),
      ifrt::ArrayCopySemantics::kReuseInput);
  copy_future.OnReady([host_value{std::move(host_value_copy)}](Status status) {
    host_value->status = std::move(status);
    host_value->ready.Notify();
  });
  return OkStatus();
}

StatusOr<pybind11::dict> IfrtHelpers::CudaArrayInterface(
    ifrt::Array* ifrt_array, std::optional<Shape>& scratch) {
  auto* pjrt_buffer = IfrtHelpers::pjrt_buffer(ifrt_array);
  // TODO(zhangqiaorjc): Differentiate between NVidia and other GPUs.
  if (pjrt_buffer->client()->platform_id() != GpuId()) {
    return InvalidArgument(
        "__cuda_array_interface__ is only defined for NVidia GPU buffers.");
  }
  if (!pjrt_buffer->on_device_shape().IsArray()) {
    return InvalidArgument(
        "__cuda_array_interface__ is only defined for array buffers.");
  }
  if (pjrt_buffer->on_device_shape().element_type() == BF16) {
    return InvalidArgument(
        "__cuda_array_interface__ is not supported for bfloat16 buffers.");
  }
  if (pjrt_buffer->on_device_shape().element_type() == F8E4M3FN) {
    return InvalidArgument(
        "__cuda_array_interface__ is not supported for F8E4M3FN buffers.");
  }
  if (pjrt_buffer->on_device_shape().element_type() == F8E4M3B11FNUZ) {
    return InvalidArgument(
        "__cuda_array_interface__ is not supported for F8E4M3B11FNUZ buffers.");
  }
  if (pjrt_buffer->on_device_shape().element_type() == F8E5M2) {
    return InvalidArgument(
        "__cuda_array_interface__ is not supported for F8E5M2 buffers.");
  }
  TF_RET_CHECK(LayoutUtil::IsMonotonicWithDim0Major(
      pjrt_buffer->on_device_shape().layout()));

  py::dict result;
  TF_ASSIGN_OR_RETURN(const auto* dynamic_shape,
                      IfrtHelpers::xla_dynamic_shape(ifrt_array, scratch));
  result["shape"] = SpanToTuple(dynamic_shape->dimensions());
  TF_ASSIGN_OR_RETURN(py::str typestr,
                      TypeDescriptorForPrimitiveType(
                          pjrt_buffer->on_device_shape().element_type()));
  result["typestr"] = std::move(typestr);
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<PjRtBuffer::ExternalReference> external_reference_hold,
      pjrt_buffer->AcquireExternalReference());
  const void* root_ptr =
      external_reference_hold->OpaqueDeviceMemoryDataPointer();
  py::tuple data(2);
  data[0] = py::int_(absl::bit_cast<std::uintptr_t>(root_ptr));
  data[1] = py::bool_(true);  // read-only
  result["data"] = std::move(data);
  result["version"] = py::int_(2);
  return result;
}

StatusOr<ifrt::DType> ToIfRtDType(py::dtype dtype) {
  TF_ASSIGN_OR_RETURN(auto primitive_type, DtypeToPrimitiveType(dtype));
  return ifrt::ToDType(primitive_type);
}

StatusOr<py::dtype> ToPybind11DType(ifrt::DType dtype) {
  TF_ASSIGN_OR_RETURN(auto primitive_type, ifrt::ToPrimitiveType(dtype));
  return PrimitiveTypeToDtype(primitive_type);
}

}  // namespace xla
