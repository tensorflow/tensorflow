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

absl::Status HostCallbackContext::OnSend(int arg_num,
                                         const PjRtTransferMetadata& metadata,
                                         PjRtChunk data) {
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

  // This assignment to update `args_` will not race with the assignments in
  // future send ops for this `arg_num` because send callbacks are supposed to
  // be invoked sequentially.
  args_.at(arg_num) = std::move(data);

  DCHECK_GE(ready_count_.load(), 1);
  if (ready_count_.fetch_sub(1) != 1) {
    return absl::OkStatus();
  }

  // This atomic store won't race against the next invocation of OnSend()
  // (e.g. by the next iteration of while loop) because send callbacks are
  // supposed to be invoked sequentially.
  ready_count_.store(args_.size());

  std::vector<void*> arg_ptrs;
  arg_ptrs.reserve(args_.size());
  for (auto& arg : args_) {
    arg_ptrs.push_back(arg.data());
  }

  std::vector<PjRtChunk> results;
  std::vector<void*> result_ptrs;
  results.reserve(result_channels_.size());
  result_ptrs.reserve(result_channels_.size());
  for (int i = 0; i < result_channels_.size(); ++i) {
    const auto& host_shape = host_callback_.results.at(i).shape;
    size_t host_size = ShapeUtil::ByteSizeOf(host_shape);
    results.push_back(PjRtChunk::AllocateDefault(host_size));
    result_ptrs.push_back(results.back().data());
  }

  EnterHostCallback();
  auto status = host_callback_.callback(result_ptrs.data(), arg_ptrs.data());
  LeaveHostCallback();

  // TODO(chky): Consider populating garbage data in results upon errors.

  // Clear the arguments for this invocation. This won't race with next
  // invocation as send callbacks are supposed to be invoked sequentially.
  for (auto& arg : args_) {
    arg = PjRtChunk{};
  }

  // Sending the results to recv callbacks if there is any. Note that after
  // this point, this callback can be invoked again (e.g. in a loop) anytime.
  for (int i = 0; i < result_channels_.size(); ++i) {
    auto& result_channel = result_channels_[i];
    result_channel->Push(std::move(results[i]));
  }

  return status;
}

void HostCallbackContext::Receive(int res_num,
                                  const PjRtTransferMetadata& metadata,
                                  std::unique_ptr<CopyToDeviceStream> stream) {
  auto& result_channel = result_channels_.at(res_num);
  result_channel->Pop().OnReady(
      [this, res_num, metadata,
       stream = std::move(stream)](absl::StatusOr<PjRtChunk> chunk) mutable {
        TF_CHECK_OK(chunk.status());

        if (!use_major_to_minor_data_layout_for_callbacks_) {
          const auto& host_shape = host_callback_.results.at(res_num).shape;
          const auto& device_shape = metadata.device_shape;
          auto statusor_linearized =
              host_memory_for_device_manager_->ToDeviceLayout(
                  chunk->data(), chunk->size(), host_shape, device_shape);
          chunk = std::move(statusor_linearized.value());
        }

        stream->AddChunk(*std::move(chunk)).OnReady([](absl::Status s) {
          TF_CHECK_OK(s);
        });
      });
}

std::unique_ptr<HostCallbackContext>
CreateHostCallbackStateAndAppendSendRecvCallbacks(
    HostCallback host_callback,
    PjRtHostMemoryForDeviceManager* host_memory_for_device_manager,
    std::vector<SendCallback>& send_callbacks,
    std::vector<RecvCallback>& recv_callbacks,
    bool use_major_to_minor_data_layout_for_callbacks) {
  auto context = std::make_unique<HostCallbackContext>(
      std::move(host_callback), use_major_to_minor_data_layout_for_callbacks,
      host_memory_for_device_manager);

  const auto& hb = context->host_callback();
  for (int arg_num = 0; arg_num < hb.operands.size(); ++arg_num) {
    const auto& operand_info = hb.operands[arg_num];
    send_callbacks.push_back(SendCallback{
        /*channel_id=*/operand_info.channel_id,
        /*callback=*/[arg_num, context = context.get()](
                         const PjRtTransferMetadata& metadata, PjRtChunk input,
                         size_t total_size_in_bytes, bool done) {
          return context->OnSend(arg_num, metadata, std::move(input));
        }});
  }

  for (int res_num = 0; res_num < hb.results.size(); ++res_num) {
    const auto& result_info = hb.results[res_num];
    recv_callbacks.push_back(RecvCallback{
        /*channel_id=*/result_info.channel_id,
        /*callback=*/[res_num, context = context.get()](
                         const PjRtTransferMetadata& metadata,
                         std::unique_ptr<CopyToDeviceStream> stream) {
          context->Receive(res_num, metadata, std::move(stream));
        }});
  }

  return context;
}

}  // namespace xla
