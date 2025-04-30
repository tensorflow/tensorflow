/* Copyright 2022 The OpenXLA Authors.

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

#include "xla/python/pjrt_ifrt/pjrt_array.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "llvm/Support/Casting.h"
#include "xla/layout.h"
#include "xla/literal.h"
#include "xla/pjrt/host_memory_spaces.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_layout.h"
#include "xla/pjrt/utils.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/basic_device_list.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/device_list.h"
#include "xla/python/ifrt/dtype.h"
#include "xla/python/ifrt/future.h"
#include "xla/python/ifrt/memory.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/ifrt/sharding.h"
#include "xla/python/pjrt_ifrt/pjrt_client.h"
#include "xla/python/pjrt_ifrt/pjrt_device.h"
#include "xla/python/pjrt_ifrt/pjrt_dtype.h"
#include "xla/python/pjrt_ifrt/pjrt_memory.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace ifrt {

namespace {

static const xla::ifrt::MemoryKind kPinnedHostMemoryKind(
    xla::PinnedHostMemorySpace::kKind);

// Validates the sharding and PjRtBuffers have consistent device and memory
// kind.
absl::Status ValidateArrayCreationInput(
    PjRtCompatibleClient* client, std::shared_ptr<const Sharding> sharding,
    const PjRtArray::PjRtBuffers& pjrt_buffers) {
  absl::Span<Device* const> sharding_devices =
      sharding->devices()->AddressableDeviceList()->devices();
  if (sharding_devices.size() != pjrt_buffers.size()) {
    return InvalidArgument("device and buffer counts mismatch: %d vs. %d",
                           sharding_devices.size(), pjrt_buffers.size());
  }
  if (pjrt_buffers.empty()) {
    return absl::OkStatus();
  }

  // Canonicalize memory kind in case it hasn't been done before.
  MemoryKind canonicalized_sharding_memory_kind =
      CanonicalizeMemoryKind(sharding->memory_kind(), sharding_devices.front());
  for (int i = 0; i < sharding_devices.size(); ++i) {
    PjRtCompatibleDevice* device =
        llvm::dyn_cast<PjRtCompatibleDevice>(sharding_devices[i]);
    if (!device) {
      return InvalidArgument("Sharding device %d is not a PjRtDevice", i);
    }
    if (device->client() != client) {
      return InvalidArgument(
          "sharding client mismatches array client: %s vs %s",
          sharding_devices[i]->DebugString(), client->platform_version());
    }
    if (pjrt_buffers[i]->device() != device->pjrt_device()) {
      return InvalidArgument(
          "PjRtBuffer's memory space is addressed by device %s vs sharding is "
          "on device %s",
          pjrt_buffers[i]->device()->DebugString(),
          sharding_devices[i]->DebugString());
    }
    MemoryKind buffer_memory_kind =
        MakeMemoryKindFromPjRtBuffer(pjrt_buffers[i].get());
    if (canonicalized_sharding_memory_kind != buffer_memory_kind) {
      return InvalidArgument(
          "PjRtBuffer's memory kind does not match sharding's memory kind. Got "
          "PjRtBuffer's memory kind: %v vs shardings's memory kind: %v",
          buffer_memory_kind, canonicalized_sharding_memory_kind);
    }
  }
  return absl::OkStatus();
}

// Validates the PjRtBuffers have consistent memory kind and returns the memory
// kind.
absl::StatusOr<MemoryKind> GetMemoryKindFromPjRtBuffers(
    const PjRtArray::PjRtBuffers& pjrt_buffers) {
  const auto first_memory_kind =
      MakeMemoryKindFromPjRtBuffer(pjrt_buffers.front().get());
  const MemoryKind canonical_first_memory_kind =
      CanonicalizeMemoryKindWithPjRtDevice(first_memory_kind,
                                           pjrt_buffers.front()->device());
  for (const auto& pjrt_buffer : pjrt_buffers) {
    if (auto memory_kind = MakeMemoryKindFromPjRtBuffer(pjrt_buffer.get());
        canonical_first_memory_kind !=
        CanonicalizeMemoryKindWithPjRtDevice(memory_kind,
                                             pjrt_buffer->device())) {
      return InvalidArgument(
          "Memory kind mismatch between PjRtBuffers. Got one buffer with "
          "memory kind: %v and another with memory_kind: %v",
          first_memory_kind, memory_kind);
    }
  }
  return first_memory_kind;
}

}  // namespace

char PjRtCompatibleArray::ID = 0;
char PjRtArray::ID = 0;

MemoryKind MakeMemoryKindFromPjRtBuffer(PjRtBuffer* pjrt_buffer) {
  if (pjrt_buffer->memory_space() == nullptr) {
    return MemoryKind();
  }
  return MemoryKind(pjrt_buffer->memory_space()->kind());
}

absl::StatusOr<tsl::RCReference<PjRtArray>> PjRtArray::Create(
    PjRtCompatibleClient* client, DType dtype, Shape shape,
    std::shared_ptr<const Sharding> sharding, PjRtBuffers pjrt_buffers,
    std::shared_ptr<const xla::PjRtLayout> layout) {
  TF_RETURN_IF_ERROR(
      ValidateArrayCreationInput(client, sharding, pjrt_buffers));
  return tsl::MakeRef<PjRtArray>(client, dtype, std::move(shape),
                                 std::move(sharding), std::move(pjrt_buffers),
                                 std::move(layout));
}

absl::StatusOr<tsl::RCReference<PjRtArray>> PjRtArray::Create(
    PjRtCompatibleClient* client, DType dtype, DynamicShape dynamic_shape,
    std::shared_ptr<const Sharding> sharding, PjRtBuffers pjrt_buffers,
    std::shared_ptr<const xla::PjRtLayout> layout) {
  TF_RETURN_IF_ERROR(
      ValidateArrayCreationInput(client, sharding, pjrt_buffers));
  return tsl::MakeRef<PjRtArray>(client, dtype, std::move(dynamic_shape),
                                 std::move(sharding), std::move(pjrt_buffers),
                                 std::move(layout));
}

absl::StatusOr<tsl::RCReference<PjRtArray>> PjRtArray::Create(
    PjRtCompatibleClient* client, std::shared_ptr<PjRtBuffer> pjrt_buffer) {
  TF_ASSIGN_OR_RETURN(auto dtype, ToDType(pjrt_buffer->element_type()));
  Shape shape(pjrt_buffer->dimensions());
  TF_ASSIGN_OR_RETURN(auto device,
                      client->LookupPjRtDevice(pjrt_buffer->device()));
  auto sharding = SingleDeviceSharding::Create(
      device, MakeMemoryKindFromPjRtBuffer(pjrt_buffer.get()));
  auto layout = (dtype.kind() == DType::kToken)
                    ? std::make_shared<xla::PjRtLayout>(xla::Layout())
                    : pjrt_buffer->layout();
  return tsl::MakeRef<PjRtArray>(
      client, dtype, std::move(shape), std::move(sharding),
      PjRtBuffers({std::move(pjrt_buffer)}), std::move(layout));
}

absl::StatusOr<tsl::RCReference<Array>> PjRtArray::FullyReplicatedShard(
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

absl::StatusOr<tsl::RCReference<PjRtArray>> PjRtArray::Create(
    PjRtCompatibleClient* client, Shape shape, PjRtBuffers pjrt_buffers) {
  if (pjrt_buffers.empty()) {
    return InvalidArgument("PjRtBuffers must be non-empty.");
  }
  TF_ASSIGN_OR_RETURN(auto dtype,
                      xla::ifrt::ToDType(pjrt_buffers.front()->element_type()));
  TF_ASSIGN_OR_RETURN(MemoryKind memory_kind,
                      GetMemoryKindFromPjRtBuffers(pjrt_buffers));

  BasicDeviceList::Devices devices;
  devices.reserve(pjrt_buffers.size());
  std::vector<Shape> shapes;
  shapes.reserve(pjrt_buffers.size());

  for (const auto& pjrt_buffer : pjrt_buffers) {
    TF_ASSIGN_OR_RETURN(auto device,
                        client->LookupPjRtDevice(pjrt_buffer->device()));
    devices.push_back(device);
    shapes.push_back(Shape(pjrt_buffer->dimensions()));
  }
  auto sharding = ifrt::ConcreteSharding::Create(
      BasicDeviceList::Create(std::move(devices)), memory_kind,
      /*shape=*/shape,
      /*shard_shapes=*/shapes);
  auto layout = pjrt_buffers.front()->layout();
  return PjRtArray::Create(client, dtype, std::move(shape), std::move(sharding),
                           std::move(pjrt_buffers), std::move(layout));
}

absl::StatusOr<tsl::RCReference<PjRtArray>> PjRtArray::Create(
    PjRtCompatibleClient* client, DynamicShape dynamic_shape,
    PjRtBuffers pjrt_buffers) {
  if (pjrt_buffers.empty()) {
    return InvalidArgument("PjRtBuffers must be non-empty.");
  }
  TF_ASSIGN_OR_RETURN(auto dtype,
                      xla::ifrt::ToDType(pjrt_buffers.front()->element_type()));
  TF_ASSIGN_OR_RETURN(auto memory_kind,
                      GetMemoryKindFromPjRtBuffers(pjrt_buffers));

  BasicDeviceList::Devices devices;
  devices.reserve(pjrt_buffers.size());
  std::vector<DynamicShape> dynamic_shapes;
  dynamic_shapes.reserve(pjrt_buffers.size());

  for (const auto& pjrt_buffer : pjrt_buffers) {
    TF_ASSIGN_OR_RETURN(auto device,
                        client->LookupPjRtDevice(pjrt_buffer->device()));
    devices.push_back(device);
    TF_ASSIGN_OR_RETURN(
        DynamicShape dynamic_shape,
        // Extracts dynamic shape info from the buffers.
        DynamicShape::Create(
            Shape(pjrt_buffer->dimensions()),
            BoundedDynamicShapeTag(pjrt_buffer->is_dynamic_dimension())));
    dynamic_shapes.push_back(std::move(dynamic_shape));
  }
  auto sharding = ifrt::ConcreteSharding::Create(
      BasicDeviceList::Create(std::move(devices)), memory_kind,
      /*dynamic_shape=*/dynamic_shape,
      /*shard_dynamic_shapes=*/dynamic_shapes);
  auto layout = pjrt_buffers.front()->layout();
  return PjRtArray::Create(client, dtype, std::move(dynamic_shape),
                           std::move(sharding), std::move(pjrt_buffers),
                           std::move(layout));
}

PjRtArray::PjRtArray(PjRtCompatibleClient* client, DType dtype, Shape shape,
                     std::shared_ptr<const Sharding> sharding,
                     PjRtBuffers pjrt_buffers,
                     std::shared_ptr<const xla::PjRtLayout> layout)
    : client_(client),
      dtype_(dtype),
      shape_(std::move(shape)),
      sharding_(std::move(sharding)),
      pjrt_buffers_(std::move(pjrt_buffers)),
      layout_(std::move(layout)) {}

PjRtArray::PjRtArray(PjRtCompatibleClient* client, DType dtype,
                     DynamicShape dynamic_shape,
                     std::shared_ptr<const Sharding> sharding,
                     PjRtBuffers pjrt_buffers,
                     std::shared_ptr<const xla::PjRtLayout> layout)
    : client_(client),
      dtype_(dtype),
      shape_(std::move(dynamic_shape)),
      sharding_(std::move(sharding)),
      pjrt_buffers_(std::move(pjrt_buffers)),
      layout_(std::move(layout)) {}

absl::StatusOr<std::vector<tsl::RCReference<Array>>>
PjRtArray::DisassembleIntoSingleDeviceArrays(
    ArrayCopySemantics semantics,
    SingleDeviceShardSemantics single_device_shard_semantics) {
  DCHECK(this);
  if (single_device_shard_semantics == SingleDeviceShardSemantics::kAllShards &&
      !sharding_->devices()->IsFullyAddressable()) {
    return InvalidArgument(
        "All shards are requested but the sharding has non-addressable "
        "devices: %v",
        *sharding_->devices());
  }
  std::vector<tsl::RCReference<Array>> result;
  result.reserve(sharding_->devices()->AddressableDeviceList()->size());
  TF_RETURN_IF_ERROR(std::visit(
      [&](const auto& this_shape) {
        TF_ASSIGN_OR_RETURN(
            auto shape_and_shardings,
            sharding_->Disassemble(
                this_shape, SingleDeviceShardSemantics::kAddressableShards));
        for (int i = 0; i < shape_and_shardings.size(); ++i) {
          PjRtBuffers buffers;
          buffers.reserve(1);
          buffers.push_back(GetPjRtBuffer(semantics, i));
          TF_ASSIGN_OR_RETURN(
              auto array,
              PjRtArray::Create(client_, dtype_,
                                std::move(shape_and_shardings[i].first),
                                std::move(shape_and_shardings[i].second),
                                std::move(buffers), layout_));
          result.push_back(std::move(array));
        }
        return absl::OkStatus();
      },
      shape_));

  return result;
}

Future<> PjRtArray::CopyToHostBuffer(
    void* data, std::optional<absl::Span<const int64_t>> byte_strides,
    ArrayCopySemantics semantics) {
  DCHECK(this);
  if (sharding_->devices()->size() != 1) {
    return Future<>(
        InvalidArgument("Only single-shard is implemented, but got %d",
                        sharding_->devices()->size()));
  }

  auto dtype = ToPrimitiveType(dtype_);
  if (!dtype.ok()) {
    return Future<>(std::move(dtype).status());
  }

  PjRtBuffer* pjrt_buffer = pjrt_buffers_.front().get();
  absl::Span<const int64_t> dims;
  absl::StatusOr<std::vector<int64_t>> logical_dims;
  if (!pjrt_buffer->has_dynamic_dimensions()) {
    dims = std::get<Shape>(shape_).dims();
  } else {
    // TODO(b/182461453): This is a blocking call. If we further implemented
    // populating dynamic shape metadata while fetching the literal, we wouldn't
    // need this static approach.
    // TODO(hyeontaek): Clean up this dynamic shape access once we formalize
    // dynamic shape support in IFRT.
    // TODO(b/314805296): Use the new dynamic shape here.
    logical_dims = pjrt_buffer->logical_dimensions();
    if (!logical_dims.ok()) {
      return Future<>(std::move(logical_dims).status());
    }
    dims = *logical_dims;
  }

  std::unique_ptr<xla::MutableBorrowingLiteral> literal;
  if (byte_strides.has_value()) {
    auto xla_shape =
        MakeShapeWithTrivialByteStrides(*dtype, dims, *byte_strides);
    if (!xla_shape.ok()) {
      return Future<>(std::move(xla_shape).status());
    }
    literal = std::make_unique<xla::MutableBorrowingLiteral>(
        static_cast<char*>(data), *xla_shape);
  } else {
    auto xla_shape = ShapeUtil::MakeShapeWithDescendingLayout(*dtype, dims);
    literal = std::make_unique<xla::MutableBorrowingLiteral>(
        static_cast<char*>(data), xla_shape);
  }
  auto* literal_ptr = literal.get();
  auto promise = Future<>::CreatePromise();
  Future<> future(promise);
  // TODO(hyeontaek): Handle semantics == kDonateInput.
  pjrt_buffer->ToLiteral(literal_ptr)
      .OnReady([literal = std::move(literal),
                promise = std::move(promise)](absl::Status s) mutable {
        promise.Set(std::move(s));
        literal = nullptr;
      });
  return future;
}

// TODO(yashkatariya): Maybe move this to ifrt::Device?
absl::StatusOr<Memory*> GetMemorySpaceFromMemoryKind(
    ifrt::Device* device, ifrt::MemoryKind memory_kind) {
  Memory* memory = nullptr;
  for (Memory* ms : device->Memories()) {
    if (ms->Kind() == memory_kind) {
      memory = ms;
      break;
    }
  }
  if (memory == nullptr) {
    return InvalidArgument(
        "Invalid memory kind: %v; available memory kinds: %s", memory_kind,
        absl::StrJoin(device->Memories(), ", ",
                      [](std::string* out, Memory* m) {
                        absl::StrAppend(out, m->Kind());
                      }));
  }
  return memory;
}

absl::StatusOr<tsl::RCReference<Array>> PjRtArray::Copy(
    std::optional<xla::ifrt::DeviceListRef> devices,
    std::optional<xla::ifrt::MemoryKind> memory_kind,
    ArrayCopySemantics semantics) {
  DCHECK(this);
  TF_ASSIGN_OR_RETURN(auto new_sharding,
                      sharding().WithDeviceAssignment(devices, memory_kind));
  if (new_sharding->devices()->size() != sharding_->devices()->size()) {
    return InvalidArgument(
        "Resharding to a different number of devices: %d; expected %d",
        new_sharding->devices()->size(), sharding_->devices()->size());
  }
  // TODO(hyeontaek): We should have an equivalence test for sharding that
  // permits device changes and nothing else.
  PjRtBuffers buffers;
  buffers.reserve(pjrt_buffers_.size());
  CHECK_GT(new_sharding->devices()->size(), 0);
  // Canonicalize memory kind in case it hasn't been done before.
  MemoryKind canonicalized_sharding_memory_kind = CanonicalizeMemoryKind(
      new_sharding->memory_kind(), new_sharding->devices()->devices().front());
  bool new_sharding_has_memory_kind =
      canonicalized_sharding_memory_kind.memory_kind().has_value();
  const absl::Span<Device* const> new_sharding_devices =
      new_sharding->devices()->devices();
  PjRtCompatibleClient* new_client = nullptr;
  for (int i = 0; i < pjrt_buffers_.size(); ++i) {
    TF_ASSIGN_OR_RETURN(Device * buffer_device,
                        client_->LookupPjRtDevice(pjrt_buffers_[i]->device()));
    bool devices_equal = buffer_device == new_sharding_devices[i];
    bool memory_kind_equal =
        new_sharding_has_memory_kind &&
        pjrt_buffers_[i]->memory_space()->kind() ==
            canonicalized_sharding_memory_kind.memory_kind();

    // No need for data transfer.
    if (devices_equal && (!new_sharding_has_memory_kind || memory_kind_equal)) {
      switch (semantics) {
        case ArrayCopySemantics::kAlwaysCopy: {
          TF_ASSIGN_OR_RETURN(
              auto memory,
              GetMemorySpaceFromMemoryKind(new_sharding_devices[i],
                                           canonicalized_sharding_memory_kind));
          PjRtMemory* pjrt_memory = llvm::dyn_cast<PjRtMemory>(memory);
          TF_ASSIGN_OR_RETURN(
              auto copied_buffer,
              pjrt_buffers_[i]->CopyToMemorySpace(pjrt_memory->pjrt_memory()));
          buffers.push_back(std::move(copied_buffer));
          break;
        }
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
      PjRtCompatibleDevice* pjrt_device =
          llvm::dyn_cast<PjRtCompatibleDevice>(new_sharding_devices[i]);
      new_client = llvm::dyn_cast<PjRtCompatibleClient>(pjrt_device->client());
      if (!pjrt_device) {
        return InvalidArgument(
            "The destination device is owned by a non-PjRt-compatible client. "
            "To use this Array on the destination device, the Array must be "
            "first fetched to the host and then sent to the destination "
            "device.");
      }
      if (!pjrt_device->IsAddressable()) {
        return InvalidArgument("Cannot copy array to non-addressable device %s",
                               pjrt_device->DebugString());
      }
      PjRtMemorySpace* pjrt_memory_space = nullptr;
      if (new_sharding_has_memory_kind) {
        TF_ASSIGN_OR_RETURN(
            auto memory,
            GetMemorySpaceFromMemoryKind(new_sharding_devices[i],
                                         canonicalized_sharding_memory_kind));
        PjRtMemory* pjrt_memory = llvm::dyn_cast<PjRtMemory>(memory);
        TF_RET_CHECK(pjrt_memory != nullptr);
        pjrt_memory_space = pjrt_memory->pjrt_memory();
      } else {
        TF_ASSIGN_OR_RETURN(pjrt_memory_space,
                            pjrt_device->pjrt_device()->default_memory_space());
      }
      TF_ASSIGN_OR_RETURN(
          std::unique_ptr<PjRtBuffer> copied_buffer,
          pjrt_buffers_[i]->CopyToMemorySpace(pjrt_memory_space));
      if (semantics == ArrayCopySemantics::kDonateInput) {
        if (!memory_kind_equal) {
          return Unimplemented(
              "Donation across different memory kinds is not implemented.");
        }
        pjrt_buffers_[i] = nullptr;
      }
      buffers.push_back(std::move(copied_buffer));
    }
  }
  if (new_client == nullptr) {
    new_client = client_;
  }
  return std::visit(
      [this, new_client, &new_sharding, &buffers](const auto& shape) {
        return PjRtArray::Create(new_client, dtype_, shape,
                                 std::move(new_sharding), std::move(buffers),
                                 layout_);
      },
      shape_);
}

Future<> PjRtArray::GetReadyFuture() const {
  DCHECK(this);
  if (pjrt_buffers_.size() == 1) {
    return pjrt_buffers_.front()->GetReadyFuture();
  }
  std::vector<Future<>> futures;
  futures.reserve(pjrt_buffers_.size());
  for (auto& buf : pjrt_buffers_) {
    futures.push_back(buf->GetReadyFuture());
  }
  return JoinFutures(absl::MakeSpan(futures));
}

Future<> PjRtArray::Delete() {
  DCHECK(this);
  for (auto& buffer : pjrt_buffers_) {
    buffer->Delete();
  }
  is_deleted_ = true;
  // TODO(hyeontaek): Return a correct future.
  return Future<>(absl::OkStatus());
}

bool PjRtArray::IsDeleted() const {
  DCHECK(this);
  // TODO(hyeontaek): This may be incorrect if PjRtBuffers are shared and a
  // portion of pjrt_buffers_ is deleted or not deleted.
  return is_deleted_ ||
         (!pjrt_buffers_.empty() && pjrt_buffers_.front()->IsDeleted());
}

std::string PjRtArray::DebugString() const {
  DCHECK(this);
  absl::StatusOr<std::shared_ptr<const xla::PjRtLayout>> layout_ptr = layout();
  std::string layout_str =
      layout_ptr.ok() ? (*layout_ptr)->ToString() : "<unknown>";

  return absl::StrFormat(
      "PjRtArray(dtype=%s; shape=%s; sharding=%s; layout=%s)",
      dtype_.DebugString(),
      std::visit([](const auto& shape) { return shape.DebugString(); }, shape_),
      sharding_->DebugString(), layout_str);
}

absl::StatusOr<std::shared_ptr<const xla::PjRtLayout>> PjRtArray::layout()
    const {
#ifndef NDEBUG
  for (int i = 1; i < pjrt_buffers_.size(); ++i) {
    std::shared_ptr<const xla::PjRtLayout> layout_i =
        pjrt_buffers_[i]->layout();
    DCHECK(*layout_ == *layout_i)
        << "PjRtArray has mismatched layouts across shards! "
        << "shard 0: " << layout_->ToString() << ", shard " << i << ": "
        << layout_i->ToString();
  }
#endif
  return layout_;
}

}  // namespace ifrt
}  // namespace xla
