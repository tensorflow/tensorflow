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

#include "xla/python/py_buffer.h"

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "pybind11/pybind11.h"  // from @pybind11
#include "pybind11/pytypes.h"  // from @pybind11
#include "xla/pjrt/pjrt_client.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/pjrt_ifrt/pjrt_array.h"
#include "xla/python/python_ref_manager.h"
#include "xla/python/transfer_guard_lib.h"
#include "xla/python/types.h"
#include "xla/xla_data.pb.h"
namespace xla {

namespace py = pybind11;

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

  if (!scratch) {
    absl::Span<const int64_t> dims;
    std::optional<std::vector<int64_t>> logical_dims_storage;
    if (pjrt_buffer->has_dynamic_dimensions()) {
      {
        py::gil_scoped_release gil_release;
        TF_ASSIGN_OR_RETURN(std::vector<int64_t> logical_dims,
                            pjrt_buffer->logical_dimensions());
        logical_dims_storage.emplace(std::move(logical_dims));
      }
      dims = *logical_dims_storage;
    } else {
      dims = pjrt_buffer->dimensions();
    }
    Shape shape = ShapeUtil::MakeShape(pjrt_buffer->element_type(), dims);
    *shape.mutable_layout() = pjrt_buffer->layout();
    scratch = std::move(shape);
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
  // TODO(yashkatariya): Plumb sharding or memory_kind here.
  return ifrt_array->Reshard(
      ifrt::SingleDeviceSharding::Create(dst_device, ifrt::MemoryKind()),
      ifrt::ArrayCopySemantics::kReuseInput);
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
