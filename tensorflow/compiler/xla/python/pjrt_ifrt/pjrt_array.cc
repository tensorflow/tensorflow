/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/python/pjrt_ifrt/pjrt_array.h"

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "llvm/Support/Casting.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_client.h"
#include "tensorflow/compiler/xla/python/ifrt/array.h"
#include "tensorflow/compiler/xla/python/ifrt/sharding.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace xla {
namespace ifrt {

char PjRtArray::ID = 0;

StatusOr<xla::PrimitiveType> ToPrimitiveType(DType dtype) {
  switch (dtype.kind()) {
    case DType::kInvalid:
    case DType::kPred:
    case DType::kS8:
    case DType::kS16:
    case DType::kS32:
    case DType::kS64:
    case DType::kU8:
    case DType::kU16:
    case DType::kU32:
    case DType::kU64:
    case DType::kF16:
    case DType::kF32:
    case DType::kBF16:
    case DType::kF64:
    case DType::kC64:
    case DType::kC128:
    case DType::kToken:
      return static_cast<xla::PrimitiveType>(static_cast<int>(dtype.kind()));
    case DType::kString:
      return InvalidArgument("Not supported as XLA PrimitiveType: %d",
                             static_cast<int>(dtype.kind()));
  }
  return InvalidArgument("Invalid DType: %d", static_cast<int>(dtype.kind()));
}

StatusOr<DType> ToDType(xla::PrimitiveType primitive_type) {
  switch (primitive_type) {
    case xla::PrimitiveType::PRIMITIVE_TYPE_INVALID:
    case xla::PrimitiveType::PRED:
    case xla::PrimitiveType::S8:
    case xla::PrimitiveType::S16:
    case xla::PrimitiveType::S32:
    case xla::PrimitiveType::S64:
    case xla::PrimitiveType::U8:
    case xla::PrimitiveType::U16:
    case xla::PrimitiveType::U32:
    case xla::PrimitiveType::U64:
    case xla::PrimitiveType::F16:
    case xla::PrimitiveType::F32:
    case xla::PrimitiveType::BF16:
    case xla::PrimitiveType::F64:
    case xla::PrimitiveType::C64:
    case xla::PrimitiveType::C128:
    case xla::PrimitiveType::TOKEN:
      return DType(static_cast<DType::Kind>(static_cast<int>(primitive_type)));
    default:
      return InvalidArgument("Invalid XLA PrimitiveType: %d",
                             static_cast<int>(primitive_type));
  }
}

StatusOr<std::unique_ptr<Array>> PjRtArray::Create(
    Client* client, DType dtype, Shape shape,
    std::shared_ptr<const Sharding> sharding, PjRtBuffers pjrt_buffers) {
  if (!llvm::isa_and_nonnull<PjRtClient>(client)) {
    return InvalidArgument("PjRtClient expected");
  }
  if (pjrt_buffers.empty()) {
    return InvalidArgument("pjrt_buffers must be non-empty");
  }
  if (sharding->devices().size() != pjrt_buffers.size()) {
    return InvalidArgument("device and buffer counts mismatch: %d vs. %d",
                           sharding->devices().size(), pjrt_buffers.size());
  }
  return std::unique_ptr<Array>(
      new PjRtArray(static_cast<PjRtClient*>(client), dtype, std::move(shape),
                    std::move(sharding), std::move(pjrt_buffers)));
}

StatusOr<std::unique_ptr<Array>> PjRtArray::Create(
    Client* client, std::shared_ptr<PjRtBuffer> pjrt_buffer) {
  if (!llvm::isa_and_nonnull<PjRtClient>(client)) {
    return InvalidArgument("PjRtClient expected");
  }
  TF_ASSIGN_OR_RETURN(auto dtype,
                      ToDType(pjrt_buffer->on_device_shape().element_type()));
  Shape shape(pjrt_buffer->on_device_shape().dimensions());
  auto sharding = SingleDeviceSharding::Create(pjrt_buffer->device());
  return std::unique_ptr<Array>(new PjRtArray(
      static_cast<PjRtClient*>(client), dtype, std::move(shape),
      std::move(sharding), PjRtBuffers({std::move(pjrt_buffer)})));
}

StatusOr<std::unique_ptr<Array>> PjRtArray::Create(
    Client* client, std::unique_ptr<PjRtBuffer> pjrt_buffer) {
  return PjRtArray::Create(client,
                           std::shared_ptr<PjRtBuffer>(pjrt_buffer.release()));
}

StatusOr<std::unique_ptr<Array>> PjRtArray::Create(Client* client, Shape shape,
                                                   PjRtBuffers pjrt_buffers) {
  TF_ASSIGN_OR_RETURN(
      auto dtype, xla::ifrt::ToDType(
                      pjrt_buffers.front()->on_device_shape().element_type()));
  DeviceList::Devices devices;
  devices.reserve(pjrt_buffers.size());
  std::vector<Shape> shapes;
  shapes.reserve(pjrt_buffers.size());

  for (const auto& pjrt_buffer : pjrt_buffers) {
    devices.push_back(pjrt_buffer->device());
    shapes.push_back(Shape(pjrt_buffer->on_device_shape().dimensions()));
  }
  return PjRtArray::Create(
      client, dtype, std::move(shape),
      ifrt::OpaqueSharding::Create(
          xla::ifrt::DeviceList(std::move(devices)),
          xla::ifrt::OpaqueSharding::MakeDisassembleFuncFromShapes(
              std::move(shapes))),
      std::move(pjrt_buffers));
}

PjRtArray::PjRtArray(PjRtClient* client, DType dtype, Shape shape,
                     std::shared_ptr<const Sharding> sharding,
                     PjRtBuffers pjrt_buffers)
    : client_(client),
      dtype_(dtype),
      shape_(std::move(shape)),
      sharding_(std::move(sharding)),
      pjrt_buffers_(std::move(pjrt_buffers)) {}

StatusOr<std::vector<std::unique_ptr<Array>>>
PjRtArray::DisassembleIntoSingleDeviceArrays(ArrayCopySemantics semantics) {
  DCHECK(this);
  std::vector<std::unique_ptr<Array>> result;
  result.reserve(sharding_->devices().size());
  TF_ASSIGN_OR_RETURN(auto shape_and_shardings, sharding_->Disassemble(shape_));
  for (int i = 0; i < sharding_->devices().size(); ++i) {
    PjRtBuffers buffers;
    buffers.reserve(1);
    switch (semantics) {
      case ArrayCopySemantics::kAlwaysCopy:
        // TODO(hyeontaek): kAlwaysCopy should clone the buffer, but the PjRt
        // API does not have efficient buffer cloning on the same device.
        buffers.push_back(pjrt_buffers_[i]);
        break;
      case ArrayCopySemantics::kReuseInput:
        buffers.push_back(pjrt_buffers_[i]);
        break;
      case ArrayCopySemantics::kDonateInput:
        // TODO(hyeontaek): We may try std::move(pjrt_buffers_[i]), but this
        // would be unsafe if there is a subsequent access to the buffer.
        buffers.push_back(pjrt_buffers_[i]);
        break;
    }
    TF_ASSIGN_OR_RETURN(
        auto array, PjRtArray::Create(client_, dtype_,
                                      std::move(shape_and_shardings[i].first),
                                      std::move(shape_and_shardings[i].second),
                                      std::move(buffers)));
    result.push_back(std::move(array));
  }
  return result;
}

Future<Status> PjRtArray::CopyToHostBuffer(
    void* data, std::optional<absl::Span<const int64_t>> byte_strides,
    ArrayCopySemantics semantics) {
  DCHECK(this);
  if (sharding_->devices().size() != 1) {
    return Future<Status>(
        InvalidArgument("Only single-shard is implemented, but got %d",
                        sharding_->devices().size()));
  }
  if (byte_strides.has_value()) {
    return Future<Status>(
        InvalidArgument("Non-default byte_strides is not yet supported"));
  }

  auto dtype = ToPrimitiveType(dtype_);
  if (!dtype.ok()) {
    return Future<Status>(std::move(dtype).status());
  }

  xla::Shape xla_shape(*dtype, shape_.dims(), /*dynamic_dimensions=*/
                       absl::InlinedVector<bool, Shape::kInlineDimensionSize>(
                           /*n=*/shape_.dims().size(), /*v=*/false),
                       /*tuple_shapes=*/{});
  xla::LayoutUtil::SetToDefaultLayout(&xla_shape);
  auto literal = std::make_unique<xla::MutableBorrowingLiteral>(
      static_cast<char*>(data), xla_shape);
  auto* literal_ptr = literal.get();
  auto promise = Future<Status>::CreatePromise();
  Future<Status> future(promise);
  PjRtBuffer* pjrt_buffer = pjrt_buffers_.front().get();
  // TODO(hyeontaek): Handle semantics == kDonateInput.
  pjrt_buffer->ToLiteral(literal_ptr)
      .OnReady([literal = std::move(literal),
                promise = std::move(promise)](Status s) mutable {
        promise.Set(std::move(s));
        literal = nullptr;
      });
  return future;
}

StatusOr<std::unique_ptr<Array>> PjRtArray::Reshard(
    std::shared_ptr<const Sharding> new_sharding,
    ArrayCopySemantics semantics) {
  DCHECK(this);
  if (new_sharding->devices().size() != sharding_->devices().size()) {
    return InvalidArgument(
        "Resharding to a different number of devices: %d; expected %d",
        new_sharding->devices().size(), sharding_->devices().size());
  }
  // TODO(hyeontaek): We should have an equivalence test for sharding that
  // permits device changes and nothing else.
  PjRtBuffers buffers;
  buffers.reserve(pjrt_buffers_.size());
  for (int i = 0; i < pjrt_buffers_.size(); ++i) {
    if (pjrt_buffers_[i]->device() == new_sharding->devices()[i]) {
      switch (semantics) {
        case ArrayCopySemantics::kAlwaysCopy:
          // TODO(hyeontaek): kAlwaysCopy should clone the buffer, but the PjRt
          // API does not have efficient buffer cloning on the same device.
          buffers.push_back(pjrt_buffers_[i]);
          break;
        case ArrayCopySemantics::kReuseInput:
          buffers.push_back(pjrt_buffers_[i]);
          break;
        case ArrayCopySemantics::kDonateInput:
          // TODO(hyeontaek): We may try std::move(pjrt_buffers_[i]), but this
          // would be unsafe if there is a subsequent access to the buffer.
          buffers.push_back(pjrt_buffers_[i]);
          break;
      }
    } else {
      TF_ASSIGN_OR_RETURN(
          std::unique_ptr<xla::PjRtBuffer> copied_buffer,
          pjrt_buffers_[i]->CopyToDevice(new_sharding->devices()[i]));
      if (semantics == ArrayCopySemantics::kDonateInput) {
        pjrt_buffers_[i] = nullptr;
      }
      buffers.push_back(std::shared_ptr<PjRtBuffer>(copied_buffer.release()));
    }
  }
  return PjRtArray::Create(client_, dtype_, shape_, std::move(new_sharding),
                           std::move(buffers));
}

Future<Status> PjRtArray::GetReadyFuture() const {
  DCHECK(this);
  if (pjrt_buffers_.size() == 1) {
    return pjrt_buffers_.front()->GetReadyFuture();
  }
  std::vector<Future<Status>> futures;
  futures.reserve(pjrt_buffers_.size());
  for (auto& buf : pjrt_buffers_) {
    futures.push_back(buf->GetReadyFuture());
  }
  return JoinFutures(absl::MakeSpan(futures));
}

Future<Status> PjRtArray::Delete() {
  DCHECK(this);
  for (auto& buffer : pjrt_buffers_) {
    buffer->Delete();
  }
  // TODO(hyeontaek): Return a correct future.
  return Future<Status>(OkStatus());
}

bool PjRtArray::IsDeleted() const {
  DCHECK(this);
  // TODO(hyeontaek): This may be incorrect if PjRtBuffers are shared and a
  // portion of pjrt_buffers_ is deleted or not deleted.
  return pjrt_buffers_.front()->IsDeleted();
}

std::string PjRtArray::DebugString() const {
  DCHECK(this);
  return absl::StrFormat("PjRtArray(dtype=%s; shape=%s; sharding=%s)",
                         dtype_.DebugString(), shape_.DebugString(),
                         sharding_->DebugString());
}

}  // namespace ifrt
}  // namespace xla
