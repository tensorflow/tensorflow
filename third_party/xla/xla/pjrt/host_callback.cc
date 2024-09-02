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

#include "xla/pjrt/host_callback.h"

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/shape_util.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/status.h"

namespace xla {

static thread_local int on_send_guard = 0;

void EnterHostCallback() { ++on_send_guard; }
void LeaveHostCallback() { --on_send_guard; }

bool ThisThreadIsInsideHostCallback() { return on_send_guard > 0; }

class HostCallbackContext {
 public:
  HostCallbackContext(
      HostCallback host_callback,
      bool use_major_to_minor_data_layout_for_callbacks,
      PjRtHostMemoryForDeviceManager* host_memory_for_device_manager)
      : host_callback_(std::move(host_callback)),
        use_major_to_minor_data_layout_for_callbacks_(
            use_major_to_minor_data_layout_for_callbacks),
        host_memory_for_device_manager_(host_memory_for_device_manager),
        args_(host_callback_.operands.size()),
        result_channels_(host_callback_.results.size()),
        ready_count_(args_.size()) {
    if (!use_major_to_minor_data_layout_for_callbacks_) {
      CHECK(host_memory_for_device_manager_);
    }
    for (auto& channel : result_channels_) {
      channel = std::make_unique<ThreadSafePjRtChunkQueue>();
    }
  }

  const HostCallback& HostCallback() const { return host_callback_; }

  SendCallback OnSend(int channel_id, int arg_num) {
    return SendCallback{
        channel_id,
        // Capture by reference is OK since this object is alive when the
        // callback is called.
        [&](const PjRtTransferMetadata& metadata, PjRtChunk data,
            size_t total_size_in_bytes, bool done) {
          if (!use_major_to_minor_data_layout_for_callbacks_) {
            const auto& arg_info = host_callback_.operands.at(arg_num);
            const auto& host_shape = arg_info.shape;
            const auto& device_shape = metadata.device_shape;

            size_t host_size = ShapeUtil::ByteSizeOf(host_shape);
            DCHECK_GE(data.size(), host_size);

            auto delinearized = PjRtChunk::AllocateDefault(host_size);
            TF_CHECK_OK(host_memory_for_device_manager_->ToHostLayout(
                data.data(), data.size(), device_shape, delinearized.data(),
                delinearized.size(), host_shape));

            data = std::move(delinearized);
          }
          return absl::OkStatus();
        }};
  }

  RecvCallback OnRecv(int channel_id, int res_num) {
    return RecvCallback{
        channel_id, [&](const PjRtTransferMetadata& metadata,
                        std::unique_ptr<CopyToDeviceStream> stream) {
          auto& result_channel = result_channels_.at(res_num);
          result_channel->Pop().OnReady(
              [this, res_num, metadata, stream = std::move(stream)](
                  absl::StatusOr<PjRtChunk> chunk) mutable {
                TF_CHECK_OK(chunk.status());

                if (!use_major_to_minor_data_layout_for_callbacks_) {
                  const auto& host_shape =
                      host_callback_.results.at(res_num).shape;
                  const auto& device_shape = metadata.device_shape;
                  auto statusor_linearized =
                      host_memory_for_device_manager_->ToDeviceLayout(
                          chunk->data(), chunk->size(), host_shape,
                          device_shape);
                  chunk = std::move(statusor_linearized.value());
                }

                stream->AddChunk(*std::move(chunk)).OnReady([](absl::Status s) {
                  TF_CHECK_OK(s);
                });
              });
          return absl::OkStatus();
        }};
  }

 private:
  struct HostCallback host_callback_;
  bool use_major_to_minor_data_layout_for_callbacks_;
  PjRtHostMemoryForDeviceManager* host_memory_for_device_manager_ = nullptr;
  std::vector<PjRtChunk> args_;
  std::vector<std::unique_ptr<ThreadSafePjRtChunkQueue>> result_channels_;
  std::atomic<int> ready_count_;
};

void HostCallbackStates::AddHostCallback(
    HostCallback host_callback,
    bool use_major_to_minor_data_layout_for_callbacks,
    PjRtHostMemoryForDeviceManager* host_memory_for_device_manager) {
  states_.push_back(std::make_unique<HostCallbackContext>(
      std::move(host_callback), use_major_to_minor_data_layout_for_callbacks,
      host_memory_for_device_manager));

  const HostCallback& cb = states_.back()->HostCallback();

  send_callbacks_.emplace_back();
  for (int arg_num = 0; arg_num < cb.operands.size(); ++arg_num) {
    const auto& operand_info = cb.operands[arg_num];
    send_callbacks_.back().push_back(
        states_.back()->OnSend(operand_info.channel_id, arg_num));
  }

  for (int res_num = 0; res_num < cb.results.size(); ++res_num) {
    const auto& result_info = cb.results[res_num];
    recv_callbacks_.back().push_back(
        states_.back()->OnRecv(result_info.channel_id, res_num));
  }
}

}  // namespace xla
