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
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_future.h"
#include "xla/python/transfer/streaming.h"
#include "xla/python/transfer/transfer_socket.pb.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/tsl/platform/statusor.h"

namespace aux {

class PremappedPjRtBuffer {
 public:
  PremappedPjRtBuffer(absl::Span<uint8_t> data,
                      std::shared_ptr<xla::PjRtClient> client,
                      std::shared_ptr<void> owner)
      : data_(data), client_(client), owner_(std::move(owner)) {}
  ~PremappedPjRtBuffer();

  absl::Span<uint8_t>* data() { return &data_; }

 private:
  absl::Span<uint8_t> data_;
  std::shared_ptr<xla::PjRtClient> client_;
  std::shared_ptr<void> owner_;
};

PremappedPjRtBuffer::~PremappedPjRtBuffer() {
  if (client_) {
    CHECK_OK(client_->DmaUnmap(data_.data()));
  }
}

absl::StatusOr<std::shared_ptr<absl::Span<uint8_t>>> MapPjrtMemory(
    std::shared_ptr<xla::PjRtClient> pjrt_client, void* data,
    size_t buffer_size, std::shared_ptr<void> owner) {
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
    std::shared_ptr<xla::PjRtClient> client, size_t buffer_size) {
  void* data = nullptr;
  if (posix_memalign(&data, kCpuPageSize, buffer_size) != 0) {
    return absl::InternalError("error in posix_memalign.");
  }
  std::shared_ptr<void> owner =
      std::shared_ptr<void>(data, [](void* data) { free(data); });
  return MapPjrtMemory(client, data, buffer_size, std::move(owner));
}

absl::StatusOr<std::vector<DmaCopyChunk>>
DmaCopyChunk::DivideBufferCopiesEvenly(std::shared_ptr<xla::PjRtBuffer> buffer,
                                       size_t xfer_size, size_t buffer_id) {
  TF_ASSIGN_OR_RETURN(size_t copy_size, buffer->GetOnDeviceSizeInBytes());
  size_t total_num_copies = (copy_size + xfer_size - 1) / xfer_size;
  std::vector<DmaCopyChunk> work_units;
  work_units.reserve(total_num_copies);
  for (size_t i = 0; i < total_num_copies; ++i) {
    work_units.push_back(
        DmaCopyChunk{[buffer](void* dst, int64_t offset,
                              int64_t transfer_size) -> xla::PjRtFuture<> {
                       return buffer->CopyRawToHost(dst, offset, transfer_size);
                     },
                     buffer_id, i* xfer_size,
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
    DmaCopyChunk blob, absl::AnyInvocable<void(PremappedCopierState* state,
                                               absl::StatusOr<void*> buf,
                                               const DmaCopyChunk& chunk) &&>
                           on_done) {
  WorkList work_list;
  {
    absl::MutexLock l(&mu_);
    work_queue_.push_back(WorkQueueItem{std::move(blob),
                                        nullptr,
                                        base_seq_id_ + work_queue_.size(),
                                        false,
                                        {},
                                        std::move(on_done)});
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
    auto& wu = work_item->work;
    wu.copy_fn(work_item->dest_buffer, wu.offset, wu.size)
        .OnReady([this, this_shared = shared_from_this(),
                  work_item](absl::Status s) {
          WorkList work_list2;
          {
            absl::MutexLock l(&mu_);
            --num_parallel_copies_;
            work_item->is_ready = true;
            work_item->result_status = s;
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
    if (work_item->result_status.ok()) {
      std::move(work_item->on_done)(this, work_item->dest_buffer,
                                    work_item->work);
    } else {
      std::move(work_item->on_done)(this, work_item->result_status,
                                    work_item->work);
      available_copy_offsets_.push_back(work_item->dest_buffer);
    }
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
        buffer_size_(atm_->buffer_size(buffer_index)),
        semaphore_(transfer_size) {}
  ~DmaDestination() override = default;

  absl::Status Put(const void* data, int64_t offset, size_t size,
                   absl::AnyInvocable<void() &&> on_done) override {
    if (offset < 0 || offset > buffer_size_ || buffer_size_ - offset < size) {
      semaphore_.Poison();
      atm_->SetBufferError(buffer_index_,
                           absl::InvalidArgumentError(absl::StrFormat(
                               "Invalid slicing of buffer size %lld with "
                               "invalid offset %lld, slice size %lld",
                               buffer_size_, offset, size)));
      return absl::OkStatus();
    }
    return semaphore_.DoWork(size, [&](bool is_last_transfer) {
      return atm_->TransferRawDataToSubBuffer(buffer_index_, data, offset, size,
                                              is_last_transfer,
                                              std::move(on_done));
    });
  }

  void Poison(absl::Status s) override {
    semaphore_.Poison();
    atm_->SetBufferError(buffer_index_, std::move(s));
  }

 private:
  std::shared_ptr<xla::PjRtClient::AsyncHostToDeviceTransferManager> atm_;
  int buffer_index_;
  size_t buffer_size_;
  // Some small modification of AsyncHostToDeviceTransferManager (eg: optional
  // expected send amount) would make this unnecessary.
  internal::IsLastSemaphore semaphore_;
};

tsl::RCReference<ChunkDestination> MakeDmaDestination(
    std::shared_ptr<xla::PjRtClient::AsyncHostToDeviceTransferManager> atm,
    int buffer_index, size_t transfer_size) {
  return tsl::MakeRef<DmaDestination>(atm, buffer_index, transfer_size);
}

RawBufferEntry::RawBufferEntry(std::vector<BufferRef> arrs,
                               std::shared_ptr<PremappedCopierState> state,
                               size_t xfer_size)
    : arrs_(std::move(arrs)), state_(state), xfer_size_(xfer_size) {}
bool RawBufferEntry::Handle(tsl::RCReference<ConnectionState> state,
                            const SocketTransferPullRequest& req,
                            size_t base_req_id) {
  for (uint64_t bid : req.buffer_ids()) {
    auto req_id = base_req_id;
    ++base_req_id;
    if (bid >= arrs_.size()) {
      state->SendError(
          req_id, 0, 0, true,
          absl::InternalError(absl::StrFormat("Buffer id: %d out of range %d",
                                              bid, arrs_.size())));
      continue;
    }
    arrs_[bid].ready_future.OnReady(
        [state, copier_state = state_, xfer_size = xfer_size_,
         buf_size = arrs_[bid].buf_size, req_id, bid,
         buffer = std::move(arrs_[bid].buffer)](absl::Status s) {
          if (!s.ok()) {
            state->SendError(req_id, 0, buf_size, true, s);
            return;
          }
          if (!buffer) {
            state->SendError(
                req_id, 0, buf_size, true,
                absl::InternalError(absl::StrFormat(
                    "Buffer id: %d has already been fetched", bid)));

            return;
          }
          for (size_t i = 0; i * xfer_size < buf_size; ++i) {
            DmaCopyChunk blob;
            blob.copy_fn = [buffer](
                               void* dst, int64_t offset,
                               int64_t transfer_size) -> xla::PjRtFuture<> {
              return buffer->CopyRawDeviceToHost(dst, offset, transfer_size);
            };
            blob.buffer_id = bid;
            blob.offset = i * xfer_size;
            blob.size = std::min(xfer_size, buf_size - blob.offset);
            bool is_largest = blob.size + blob.offset == buf_size;
            copier_state->ScheduleCopy(
                std::move(blob),
                [req_id, state, copier_state, is_largest](
                    PremappedCopierState* copier_state_ptr,
                    absl::StatusOr<void*> buf, const DmaCopyChunk& chunk) {
                  if (!buf.ok()) {
                    state->SendError(req_id, chunk.offset, chunk.size,
                                     is_largest, buf.status());
                    return;
                  }
                  CHECK_OK(buf.status());
                  state->Send(req_id, buf.value(), chunk.offset, chunk.size,
                              is_largest, [copier_state, buf = buf.value()]() {
                                copier_state->ReturnBuffer(buf);
                              });
                });
          }
        });
  }

  num_consumed_bufs_ += req.buffer_ids().size();
  return num_consumed_bufs_ == arrs_.size();
}

PjRtBufferEntry::PjRtBufferEntry(std::vector<BufferRef> arrs,
                                 std::shared_ptr<PremappedCopierState> state,
                                 size_t xfer_size)
    : arrs_(std::move(arrs)), state_(state), xfer_size_(xfer_size) {
  for (auto& arr : arrs_) {
    if (!arr.ready_future.IsValid()) {
      arr.ready_future = arr.buffer->GetReadyFuture();
    }
  }
}
bool PjRtBufferEntry::Handle(tsl::RCReference<ConnectionState> state,
                             const SocketTransferPullRequest& req,
                             size_t base_req_id) {
  for (uint64_t bid : req.buffer_ids()) {
    auto req_id = base_req_id;
    ++base_req_id;
    if (bid >= arrs_.size()) {
      state->SendError(
          req_id, 0, 0, true,
          absl::InternalError(absl::StrFormat("Buffer id: %d out of range %d",
                                              bid, arrs_.size())));
      continue;
    }
    arrs_[bid].ready_future.OnReady(
        [state, copier_state = state_, xfer_size = xfer_size_,
         buf_size = arrs_[bid].buf_size, req_id, bid,
         buffer = std::move(arrs_[bid].buffer)](absl::Status s) {
          if (!s.ok()) {
            state->SendError(req_id, 0, buf_size, true, s);
            return;
          }
          if (!buffer) {
            state->SendError(
                req_id, 0, buf_size, true,
                absl::InternalError(absl::StrFormat(
                    "Buffer id: %d has already been fetched", bid)));

            return;
          }
          for (size_t i = 0; i * xfer_size < buf_size; ++i) {
            DmaCopyChunk blob;
            blob.copy_fn = [buffer = std::move(buffer)](
                               void* dst, int64_t offset,
                               int64_t transfer_size) -> xla::PjRtFuture<> {
              return buffer->CopyRawToHost(dst, offset, transfer_size);
            };
            blob.buffer_id = bid;
            blob.offset = i * xfer_size;
            blob.size = std::min(xfer_size, buf_size - blob.offset);
            bool is_largest = blob.size + blob.offset == buf_size;
            copier_state->ScheduleCopy(
                std::move(blob),
                [req_id, state, copier_state, is_largest](
                    PremappedCopierState* copier_state_ptr,
                    absl::StatusOr<void*> buf, const DmaCopyChunk& chunk) {
                  if (!buf.ok()) {
                    state->SendError(req_id, chunk.offset, chunk.size,
                                     is_largest, buf.status());
                    return;
                  }
                  state->Send(req_id, buf.value(), chunk.offset, chunk.size,
                              is_largest, [copier_state, buf = buf.value()]() {
                                copier_state->ReturnBuffer(buf);
                              });
                });
          }
        });
  }

  num_consumed_bufs_ += req.buffer_ids().size();
  return num_consumed_bufs_ == arrs_.size();
}

}  // namespace aux
