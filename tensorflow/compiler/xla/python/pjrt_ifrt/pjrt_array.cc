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
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_client.h"
#include "tensorflow/compiler/xla/pjrt/utils.h"
#include "tensorflow/compiler/xla/python/ifrt/array.h"
#include "tensorflow/compiler/xla/python/ifrt/device.h"
#include "tensorflow/compiler/xla/python/ifrt/memory.h"
#include "tensorflow/compiler/xla/python/ifrt/sharding.h"
#include "tensorflow/compiler/xla/python/pjrt_ifrt/pjrt_client.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace xla {
namespace ifrt {

char PjRtCompatibleArray::ID = 0;
char PjRtArray::ID = 0;

StatusOr<xla::PrimitiveType> ToPrimitiveType(DType dtype) {
  switch (dtype.kind()) {
    case DType::kInvalid:
    case DType::kPred:
    case DType::kS4:
    case DType::kS8:
    case DType::kS16:
    case DType::kS32:
    case DType::kS64:
    case DType::kU4:
    case DType::kU8:
    case DType::kU16:
    case DType::kU32:
    case DType::kU64:
    case DType::kF8E4M3FN:
    case DType::kF8E4M3B11FNUZ:
    case DType::kF8E4M3FNUZ:
    case DType::kF8E5M2:
    case DType::kF8E5M2FNUZ:
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
    case xla::PrimitiveType::S4:
    case xla::PrimitiveType::S8:
    case xla::PrimitiveType::S16:
    case xla::PrimitiveType::S32:
    case xla::PrimitiveType::S64:
    case xla::PrimitiveType::U4:
    case xla::PrimitiveType::U8:
    case xla::PrimitiveType::U16:
    case xla::PrimitiveType::U32:
    case xla::PrimitiveType::U64:
    case xla::PrimitiveType::F8E4M3FN:
    case xla::PrimitiveType::F8E4M3B11FNUZ:
    case xla::PrimitiveType::F8E4M3FNUZ:
    case xla::PrimitiveType::F8E5M2:
    case xla::PrimitiveType::F8E5M2FNUZ:
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

MemoryKind MakeMemoryKindFromPjRtBuffer(PjRtBuffer* pjrt_buffer) {
  if (pjrt_buffer->memory_space() == nullptr) {
    return MemoryKind();
  }
  return MemoryKind(pjrt_buffer->memory_space()->memory_space_kind());
}

StatusOr<tsl::RCReference<PjRtArray>> PjRtArray::Create(
    PjRtCompatibleClient* client, DType dtype, Shape shape,
    std::shared_ptr<const Sharding> sharding, PjRtBuffers pjrt_buffers) {
  if (pjrt_buffers.empty()) {
    return InvalidArgument("pjrt_buffers must be non-empty");
  }
  if (sharding->devices().size() != pjrt_buffers.size()) {
    return InvalidArgument("device and buffer counts mismatch: %d vs. %d",
                           sharding->devices().size(), pjrt_buffers.size());
  }

  for (int i = 0; i < sharding->devices().size(); ++i) {
    if (pjrt_buffers[i]->device() != sharding->devices()[i]) {
      return InvalidArgument(
          "PjRtBuffer's memory space is addressed by device %s vs sharding is "
          "on device %s",
          pjrt_buffers[i]->device()->DebugString(),
          sharding->devices()[i]->DebugString());
    }
    // TODO(yashkatariya): Check for memory kind after PJRT C API supports
    // memories on PJRT_Buffer.
  }
  return tsl::MakeRef<PjRtArray>(client, dtype, std::move(shape),
                                 std::move(sharding), std::move(pjrt_buffers));
}

StatusOr<tsl::RCReference<PjRtArray>> PjRtArray::Create(
    PjRtCompatibleClient* client, std::shared_ptr<PjRtBuffer> pjrt_buffer) {
  TF_ASSIGN_OR_RETURN(auto dtype, ToDType(pjrt_buffer->element_type()));
  Shape shape(pjrt_buffer->dimensions());
  auto sharding = SingleDeviceSharding::Create(
      pjrt_buffer->device(), MakeMemoryKindFromPjRtBuffer(pjrt_buffer.get()));
  return tsl::MakeRef<PjRtArray>(client, dtype, std::move(shape),
                                 std::move(sharding),
                                 PjRtBuffers({std::move(pjrt_buffer)}));
}

StatusOr<tsl::RCReference<Array>> PjRtArray::FullyReplicatedShard(
    ArrayCopySemantics semantics) {
  return PjRtArray::Create(client(), GetPjRtBuffer(semantics, 0));
}

std::shared_ptr<PjRtBuffer> PjRtArray::GetPjRtBuffer(
    ArrayCopySemantics semantics, int index) const {
  switch (semantics) {
    case ArrayCopySemantics::kAlwaysCopy:
      // TODO(hyeontaek): kAlwaysCopy should clone the buffer, but the PjRt
      // API does not have efficient buffer cloning on the same device.
      return pjrt_buffers_[index];
    case ArrayCopySemantics::kReuseInput:
      return pjrt_buffers_[index];
    case ArrayCopySemantics::kDonateInput:
      // TODO(hyeontaek): We may try std::move(pjrt_buffers_[i]), but this
      // would be unsafe if there is a subsequent access to the buffer.
      return pjrt_buffers_[index];
  }
}

StatusOr<tsl::RCReference<PjRtArray>> PjRtArray::Create(
    PjRtCompatibleClient* client, Shape shape, PjRtBuffers pjrt_buffers) {
  TF_ASSIGN_OR_RETURN(auto dtype,
                      xla::ifrt::ToDType(pjrt_buffers.front()->element_type()));
  DeviceList::Devices devices;
  devices.reserve(pjrt_buffers.size());
  std::vector<Shape> shapes;
  shapes.reserve(pjrt_buffers.size());

  const auto first_memory_kind =
      MakeMemoryKindFromPjRtBuffer(pjrt_buffers.front().get());
  const MemoryKind canonical_first_memory_kind =
      CanonicalizeMemoryKind(first_memory_kind, pjrt_buffers.front()->device());
  for (const auto& pjrt_buffer : pjrt_buffers) {
    devices.push_back(pjrt_buffer->device());
    shapes.push_back(Shape(pjrt_buffer->dimensions()));
    if (auto memory_kind = MakeMemoryKindFromPjRtBuffer(pjrt_buffer.get());
        canonical_first_memory_kind !=
        CanonicalizeMemoryKind(memory_kind, devices.back())) {
      return InvalidArgument(
          "Memory kind mismatch between PjRtBuffers. Got one buffer with "
          "memory kind: %s and another with memory_kind: %s",
          first_memory_kind.DebugString(), memory_kind.DebugString());
    }
  }
  auto sharding = ifrt::ConcreteSharding::Create(DeviceList(std::move(devices)),
                                                 first_memory_kind,
                                                 /*shape=*/shape,
                                                 /*shard_shapes=*/shapes);
  return PjRtArray::Create(client, dtype, std::move(shape), std::move(sharding),
                           std::move(pjrt_buffers));
}

PjRtArray::PjRtArray(PjRtCompatibleClient* client, DType dtype, Shape shape,
                     std::shared_ptr<const Sharding> sharding,
                     PjRtBuffers pjrt_buffers)
    : client_(client),
      dtype_(dtype),
      shape_(std::move(shape)),
      sharding_(std::move(sharding)),
      pjrt_buffers_(std::move(pjrt_buffers)) {}

StatusOr<std::vector<tsl::RCReference<Array>>>
PjRtArray::DisassembleIntoSingleDeviceArrays(ArrayCopySemantics semantics) {
  DCHECK(this);
  std::vector<tsl::RCReference<Array>> result;
  result.reserve(sharding_->devices().size());
  TF_ASSIGN_OR_RETURN(auto shape_and_shardings, sharding_->Disassemble(shape_));
  for (int i = 0; i < sharding_->devices().size(); ++i) {
    PjRtBuffers buffers;
    buffers.reserve(1);
    buffers.push_back(GetPjRtBuffer(semantics, i));
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

  auto dtype = ToPrimitiveType(dtype_);
  if (!dtype.ok()) {
    return Future<Status>(std::move(dtype).status());
  }

  PjRtBuffer* pjrt_buffer = pjrt_buffers_.front().get();
  absl::Span<const int64_t> dims;
  StatusOr<std::vector<int64_t>> logical_dims;
  if (!pjrt_buffer->has_dynamic_dimensions()) {
    dims = shape_.dims();
  } else {
    // TODO(b/182461453): This is a blocking call. If we further implemented
    // populating dynamic shape metadata while fetching the literal, we wouldn't
    // need this static approach.
    // TODO(hyeontaek): Clean up this dynamic shape access once we formalize
    // dynamic shape support in IFRT.
    logical_dims = pjrt_buffer->logical_dimensions();
    if (!logical_dims.ok()) {
      return Future<Status>(std::move(logical_dims).status());
    }
    dims = *logical_dims;
  }

  std::unique_ptr<xla::MutableBorrowingLiteral> literal;
  if (byte_strides.has_value()) {
    auto xla_shape =
        MakeShapeWithTrivialByteStrides(*dtype, dims, *byte_strides);
    if (!xla_shape.ok()) {
      return Future<Status>(std::move(xla_shape).status());
    }
    literal = std::make_unique<xla::MutableBorrowingLiteral>(
        static_cast<char*>(data), *xla_shape);
  } else {
    auto xla_shape = ShapeUtil::MakeShapeWithDescendingLayout(*dtype, dims);
    literal = std::make_unique<xla::MutableBorrowingLiteral>(
        static_cast<char*>(data), xla_shape);
  }
  auto* literal_ptr = literal.get();
  auto promise = Future<Status>::CreatePromise();
  Future<Status> future(promise);
  // TODO(hyeontaek): Handle semantics == kDonateInput.
  pjrt_buffer->ToLiteral(literal_ptr)
      .OnReady([literal = std::move(literal),
                promise = std::move(promise)](Status s) mutable {
        promise.Set(std::move(s));
        literal = nullptr;
      });
  return future;
}

StatusOr<std::unique_ptr<PjRtBuffer>> TransferPjRtBufferBetweenMemories(
    std::shared_ptr<PjRtBuffer> pjrt_buffer, ifrt::Device* new_device,
    ifrt::MemoryKind new_memory_kind) {
  // Fast path for transferring asynchronously from host to device.
  // TODO(yashkatariya, hyeontaek): Make this work for all memory kinds as the
  // default path and remove the fallback code below.
  if ((pjrt_buffer->memory_space() != nullptr &&
       pjrt_buffer->memory_space()->memory_space_kind() == "unpinned_host") &&
      (new_memory_kind.memory_kind().has_value() &&
       new_memory_kind.memory_kind() == "tpu_hbm") &&
      !absl::StrContains(new_device->client()->platform_version(),
                         "PJRT C API")) {
    // This is on_device_shape because pjrt_buffer is on the host.
    std::shared_ptr<xla::MutableLiteralBase> literal =
        std::make_shared<xla::Literal>(pjrt_buffer->on_device_shape());
    TF_ASSIGN_OR_RETURN(
        auto transfer_manager,
        new_device->client()->CreateBuffersForAsyncHostToDevice(
            absl::MakeConstSpan({pjrt_buffer->on_device_shape()}), new_device));
    std::unique_ptr<PjRtBuffer> output_pjrt_buffer =
        transfer_manager->RetrieveBuffer(0);

    PjRtFuture<Status> future = pjrt_buffer->ToLiteral(literal.get());
    future.OnReady([literal = std::move(literal),
                    transfer_manager =
                        std::move(transfer_manager)](Status status) mutable {
      if (!status.ok()) {
        transfer_manager->SetBufferError(0, std::move(status));
        return;
      }
      LiteralSlice literal_slice = *literal;
      // transfer_manager destruction could be blocking depending on the
      // backend, extend its lifetime to after the transfer is done to avoid
      // blocking a thread.
      auto transfer_manager_ptr = transfer_manager.get();
      Status transfer_status = transfer_manager_ptr->TransferLiteralToBuffer(
          0, literal_slice,
          [literal = std::move(literal),
           transfer_manager = std::move(transfer_manager)]() {});
      if (!transfer_status.ok()) {
        transfer_manager_ptr->SetBufferError(0, std::move(transfer_status));
      }
    });
    return output_pjrt_buffer;
  }
  TF_ASSIGN_OR_RETURN(std::shared_ptr<Literal> literal,
                      pjrt_buffer->ToLiteralSync());
  // Avoid use-after-free on `literal` due to unsequenced move and use.
  Literal* literal_pointer = literal.get();
  absl::InlinedVector<int64_t, 4> byte_strides(
      literal->shape().dimensions_size());
  TF_RETURN_IF_ERROR(
      ShapeUtil::ByteStrides(literal->shape(), absl::MakeSpan(byte_strides)));
  ifrt::Client::HostBufferSemantics host_buffer_semantics =
      ifrt::Client::HostBufferSemantics::kImmutableUntilTransferCompletes;

  PjRtMemorySpace* memory_space = nullptr;
  for (PjRtMemorySpace* ms : pjrt_buffer->device()->memory_spaces()) {
    if (ms->memory_space_kind() == new_memory_kind.memory_kind()) {
      memory_space = ms;
      break;
    }
  }
  if (memory_space == nullptr) {
    return InvalidArgument(
        "Invalid memory kind: %s; available memory kinds: %s",
        new_memory_kind.DebugString(),
        absl::StrJoin(pjrt_buffer->device()->memory_spaces(), ", ",
                      [](std::string* out, PjRtMemorySpace* ms) {
                        absl::StrAppend(out, ms->memory_space_kind());
                      }));
  }
  return new_device->client()->BufferFromHostBuffer(
      literal_pointer->untyped_data(), literal_pointer->shape().element_type(),
      literal_pointer->shape().dimensions(), byte_strides,
      host_buffer_semantics,
      [literal{std::move(literal)}]() { /* free literal */ }, memory_space,
      /*device_layout=*/nullptr);
}

StatusOr<tsl::RCReference<Array>> PjRtArray::Reshard(
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
    // TODO(yashkatariya): Remove the
    // `pjrt_buffers_[i]->memory_space() != nullptr` check after PJRT C API
    // populates memory space on PJRT_Buffer.
    bool memory_kind_equal =
        !new_sharding->memory_kind().memory_kind().has_value() ||
        (pjrt_buffers_[i]->memory_space() != nullptr &&
         pjrt_buffers_[i]->memory_space()->memory_space_kind() ==
             new_sharding->memory_kind().memory_kind());
    bool devices_equal =
        pjrt_buffers_[i]->device() == new_sharding->devices()[i];

    if (devices_equal && memory_kind_equal) {
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
      if (new_sharding->devices()[i]->client() == nullptr) {
        return InvalidArgument(
            "The destination device is owned by a non-PjRt-compatible client. "
            "To use this Array on the destination device, the Array must be "
            "first fetched to the host and then sent to the destination "
            "device.");
      }
      // If memory kinds match but devices are not the same.
      if (!devices_equal && memory_kind_equal) {
        TF_ASSIGN_OR_RETURN(
            std::unique_ptr<xla::PjRtBuffer> copied_buffer,
            pjrt_buffers_[i]->CopyToDevice(new_sharding->devices()[i]));
        if (semantics == ArrayCopySemantics::kDonateInput) {
          pjrt_buffers_[i] = nullptr;
        }
        buffers.push_back(std::shared_ptr<PjRtBuffer>(copied_buffer.release()));
      } else if (devices_equal && !memory_kind_equal) {
        TF_ASSIGN_OR_RETURN(std::unique_ptr<PjRtBuffer> copied_buffer,
                            TransferPjRtBufferBetweenMemories(
                                pjrt_buffers_[i], new_sharding->devices()[i],
                                new_sharding->memory_kind()));
        if (semantics == ArrayCopySemantics::kDonateInput) {
          return Unimplemented(
              "Donation across different memory kinds is not implemented.");
        }
        buffers.push_back(std::shared_ptr<PjRtBuffer>(copied_buffer.release()));
      } else {
        CHECK(!devices_equal && !memory_kind_equal);
        TF_ASSIGN_OR_RETURN(
            std::shared_ptr<xla::PjRtBuffer> copied_buffer,
            pjrt_buffers_[i]->CopyToDevice(new_sharding->devices()[i]));
        TF_ASSIGN_OR_RETURN(
            std::unique_ptr<PjRtBuffer> transferred_buffer,
            TransferPjRtBufferBetweenMemories(std::move(copied_buffer),
                                              new_sharding->devices()[i],
                                              new_sharding->memory_kind()));
        if (semantics == ArrayCopySemantics::kDonateInput) {
          return Unimplemented(
              "Donation across different memory kinds is not implemented.");
        }
        buffers.push_back(
            std::shared_ptr<PjRtBuffer>(transferred_buffer.release()));
      }
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
