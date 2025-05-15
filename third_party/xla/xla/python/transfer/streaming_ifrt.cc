/* Copyright 2025 The OpenXLA Authors.

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
#include "xla/python/transfer/streaming_ifrt.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <utility>
#include <vector>

#include "absl/functional/any_invocable.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/pjrt_ifrt/pjrt_client.h"
#include "xla/python/transfer/streaming.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/tsl/platform/statusor.h"

namespace aux {

class PremappedPjRtBuffer {
 public:
  PremappedPjRtBuffer(absl::Span<uint8_t> data, xla::PjRtClient* client,
                      std::shared_ptr<void> owner)
      : data_(data), client_(client), owner_(std::move(owner)) {}
  ~PremappedPjRtBuffer();

  absl::Span<uint8_t>* data() { return &data_; }

 private:
  absl::Span<uint8_t> data_;
  xla::PjRtClient* client_;
  std::shared_ptr<void> owner_;
};

PremappedPjRtBuffer::~PremappedPjRtBuffer() {
  if (client_) {
    CHECK_OK(client_->DmaUnmap(data_.data()));
  }
}

absl::StatusOr<std::shared_ptr<absl::Span<uint8_t>>> MapPjrtMemory(
    xla::ifrt::Client* client, void* data, size_t buffer_size,
    std::shared_ptr<void> owner) {
  auto* ifrt_client = llvm::dyn_cast_or_null<xla::ifrt::PjRtClient>(client);
  if (ifrt_client == nullptr) {
    return absl::InternalError("MapPjrtMemory only supports PjRtClient");
  }
  auto* pjrt_client = ifrt_client->pjrt_client();
  auto s = pjrt_client->DmaMap(data, buffer_size);
  if (s.code() == ::absl::StatusCode::kUnimplemented) {
    pjrt_client = nullptr;
  } else if (!s.ok()) {
    return s;
  }
  auto result = std::make_shared<PremappedPjRtBuffer>(
      absl::MakeSpan(reinterpret_cast<uint8_t*>(data), buffer_size),
      pjrt_client, std::move(owner));
  return std::shared_ptr<absl::Span<uint8_t>>(result, result->data());
}

absl::StatusOr<std::shared_ptr<absl::Span<uint8_t>>> AllocateAndMapPjrtMemory(
    xla::ifrt::Client* client, size_t buffer_size) {
  void* data = nullptr;
  if (posix_memalign(&data, kCpuPageSize, buffer_size) != 0) {
    return absl::InternalError("error in posix_memalign.");
  }
  std::shared_ptr<void> owner =
      std::shared_ptr<void>(data, [](void* data) { free(data); });
  return MapPjrtMemory(client, data, buffer_size, std::move(owner));
}

absl::StatusOr<std::vector<DmaCopyChunk>>
DmaCopyChunk::DivideBufferCopiesEvenly(xla::ifrt::ArrayRef arr,
                                       size_t xfer_size, size_t buffer_id) {
  auto* pjrt_arr =
      llvm::dyn_cast_or_null<xla::ifrt::PjRtCompatibleArray>(arr.get());
  auto* buffer = pjrt_arr->pjrt_buffers()[0].get();
  TF_ASSIGN_OR_RETURN(size_t copy_size, buffer->GetOnDeviceSizeInBytes());
  size_t total_num_copies = (copy_size + xfer_size - 1) / xfer_size;
  std::vector<DmaCopyChunk> work_units;
  work_units.reserve(total_num_copies);
  for (size_t i = 0; i < total_num_copies; ++i) {
    work_units.push_back(
        DmaCopyChunk{arr, buffer, buffer_id, i * xfer_size,
                     std::min(copy_size - i * xfer_size, xfer_size)});
  }
  return work_units;
}

PremappedCopierState::PremappedCopierState(
    std::shared_ptr<absl::Span<uint8_t>> scratch,
    size_t max_num_parallel_copies, size_t xfer_size)
    : scratch_(scratch),
      max_num_parallel_copies_(max_num_parallel_copies),
      xfer_size_(xfer_size) {
  max_copies_ = scratch->size() / xfer_size_;
  max_copies_ = std::min(max_copies_, size_t(8));
  available_copy_offsets_.reserve(max_copies_);
  for (size_t i = 0; i < max_copies_; ++i) {
    available_copy_offsets_.push_back(reinterpret_cast<char*>(scratch->data()) +
                                      i * xfer_size_);
  }
}

void PremappedCopierState::ScheduleCopy(
    const DmaCopyChunk& blob,
    absl::AnyInvocable<void(PremappedCopierState* state, void* buf,
                            const DmaCopyChunk& chunk) &&>
        on_done) {
  WorkList work_list;
  {
    absl::MutexLock l(&mu_);
    work_queue_.push_back(WorkQueueItem{blob, nullptr,
                                        base_seq_id_ + work_queue_.size(),
                                        false, std::move(on_done)});
    work_list = FindWorkLocked();
  }
  StartWorkUnlocked(work_list);
}

void PremappedCopierState::ReturnBuffer(void* buffer) {
  WorkList work_list;
  {
    absl::MutexLock l(&mu_);
    available_copy_offsets_.push_back(buffer);
    work_list = FindWorkLocked();
  }
  StartWorkUnlocked(work_list);
}

PremappedCopierState::WorkList PremappedCopierState::FindWorkLocked() {
  WorkList out;
  while (read_seq_id_ < base_seq_id_ + work_queue_.size() &&
         num_parallel_copies_ < max_num_parallel_copies_ &&
         !available_copy_offsets_.empty()) {
    auto* temp = &work_queue_[read_seq_id_ - base_seq_id_];
    temp->dest_buffer = available_copy_offsets_.back();
    ++num_parallel_copies_;
    ++read_seq_id_;
    available_copy_offsets_.pop_back();
    out.push_back(temp);
  }
  return out;
}

void PremappedCopierState::StartWorkUnlocked(const WorkList& work_list) {
  for (WorkQueueItem* work_item : work_list) {
    const auto& wu = work_item->work;
    wu.buffer->CopyRawToHost(work_item->dest_buffer, wu.offset, wu.size)
        .OnReady([this, work_item](absl::Status s) {
          CHECK_OK(s);
          WorkList work_list2;
          {
            absl::MutexLock l(&mu_);
            --num_parallel_copies_;
            work_item->is_ready = true;
            FlushReadyWorkItemsInOrder();
            work_list2 = FindWorkLocked();
          }
          StartWorkUnlocked(work_list2);
        });
  }
}

void PremappedCopierState::FlushReadyWorkItemsInOrder() {
  while (!work_queue_.empty()) {
    auto* work_item = &work_queue_.front();
    if (!work_item->is_ready) {
      return;
    }
    std::move(work_item->on_done)(this, work_item->dest_buffer,
                                  work_item->work);
    work_queue_.pop_front();
    ++base_seq_id_;
  }
}

class DmaDestination : public ChunkDestination {
 public:
  DmaDestination(
      std::shared_ptr<xla::PjRtClient::AsyncHostToDeviceTransferManager> atm,
      int buffer_index, size_t transfer_size)
      : atm_(std::move(atm)),
        buffer_index_(buffer_index),
        semaphore_(transfer_size) {}
  ~DmaDestination() override = default;

  absl::Status Put(const void* data, int64_t offset, size_t size,
                   absl::AnyInvocable<void() &&> on_done) override {
    return semaphore_.DoWork(size, [&](bool is_last_transfer) {
      return atm_->TransferRawDataToSubBuffer(buffer_index_, data, offset, size,
                                              is_last_transfer,
                                              std::move(on_done));
    });
  }

  std::shared_ptr<xla::PjRtClient::AsyncHostToDeviceTransferManager> atm_;
  int buffer_index_;
  // Some small modification of AsyncHostToDeviceTransferManager (eg: optional
  // expected send amount) would make this unnecessary.
  internal::IsLastSemaphore semaphore_;
};

tsl::RCReference<ChunkDestination> MakeDmaDestination(
    std::shared_ptr<xla::PjRtClient::AsyncHostToDeviceTransferManager> atm,
    int buffer_index, size_t transfer_size) {
  return tsl::MakeRef<DmaDestination>(atm, buffer_index, transfer_size);
}

}  // namespace aux
