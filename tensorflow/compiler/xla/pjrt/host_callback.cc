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

#include "tensorflow/compiler/xla/pjrt/host_callback.h"

#include <utility>

namespace xla {

void HostCallbackContext::OnSend(int arg_num,
                                 const PjRtTransferMetadata& metadata,
                                 PjRtChunk data) {
  auto* host_memory_for_device_manager =
      client_->GetPjRtHostMemoryForDeviceManager();
  const auto& arg_info = host_callback_->operands.at(arg_num);
  const auto& host_shape = arg_info.shape;
  const auto& device_shape = metadata.device_shape;

  size_t host_size = ShapeUtil::ByteSizeOf(host_shape);
  DCHECK_GE(data.size(), host_size);

  auto delinearized = PjRtChunk::AllocateDefault(host_size);
  TF_CHECK_OK(host_memory_for_device_manager->ToHostLayout(
      data.data(), data.size(), device_shape, delinearized.data(),
      delinearized.size(), host_shape));

  // This assignment to update `args_` will not race with the assignments in
  // future send ops for this `arg_num` because we always insert recv after
  // send ops.
  args_.at(arg_num) = std::move(delinearized);

  DCHECK_GE(ready_count_.load(), 1);
  if (ready_count_.fetch_sub(1) != 1) return;

  // This atomic store won't race against the next invocation of OnSend()
  // (e.g. by the next iteration of while loop) because we always insert a
  // Recv op after the Send ops.
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
    const auto& host_shape = host_callback_->results.at(i).shape;
    size_t host_size = ShapeUtil::ByteSizeOf(host_shape);
    results.push_back(PjRtChunk::AllocateDefault(host_size));
    result_ptrs.push_back(results.back().data());
  }

  host_callback_->callback(result_ptrs.data(), arg_ptrs.data());

  // Clear the arguments for this invocation. This won't race with next
  // invocation as we always insert recvs after these sends.
  for (auto& arg : args_) {
    arg = PjRtChunk{};
  }

  // Sending the results to recv callbacks if there is any. Note that after
  // this point, this callback can be invoked again (e.g. in a loop) anytime.
  for (int i = 0; i < result_channels_.size(); ++i) {
    auto& result_channel = result_channels_[i];
    result_channel->Push(std::move(results[i]));
  }
}

void HostCallbackContext::Receive(int res_num,
                                  const PjRtTransferMetadata& metadata,
                                  CopyToDeviceStream& stream) {
  auto& result_channel = result_channels_.at(res_num);
  PjRtChunk chunk = result_channel->Pop();

  auto* host_memory_for_device_manager =
      client_->GetPjRtHostMemoryForDeviceManager();
  const auto& host_shape = host_callback_->results.at(res_num).shape;
  const auto& device_shape = metadata.device_shape;

  auto statusor_linearized = host_memory_for_device_manager->ToDeviceLayout(
      chunk.data(), chunk.size(), host_shape, device_shape);
  TF_CHECK_OK(stream.AddChunk(std::move(statusor_linearized).value()));
}

std::unique_ptr<HostCallbackContext>
CreateHostCallbackStateAndAppendSendRecvCallbacks(
    const HostCallback* host_callback, PjRtClient* client,
    std::vector<SendCallback>& send_callbacks,
    std::vector<RecvCallback>& recv_callbacks) {
  auto context = std::make_unique<HostCallbackContext>(host_callback, client);

  for (int arg_num = 0; arg_num < host_callback->operands.size(); ++arg_num) {
    const auto& operand_info = host_callback->operands[arg_num];
    send_callbacks.push_back(SendCallback{
        /*channel_id=*/operand_info.channel_id,
        /*callback=*/[arg_num, context = context.get()](
                         const PjRtTransferMetadata& metadata, PjRtChunk input,
                         size_t total_size_in_bytes, bool done) {
          context->OnSend(arg_num, metadata, std::move(input));
        }});
  }

  for (int res_num = 0; res_num < host_callback->results.size(); ++res_num) {
    const auto& result_info = host_callback->results[res_num];
    recv_callbacks.push_back(
        RecvCallback{/*channel_id=*/result_info.channel_id,
                     /*callback=*/[res_num, context = context.get()](
                                      const PjRtTransferMetadata& metadata,
                                      CopyToDeviceStream& stream) {
                       context->Receive(res_num, metadata, stream);
                     }});
  }

  return context;
}

}  // namespace xla
