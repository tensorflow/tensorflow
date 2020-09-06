/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/distributed_runtime/eager/remote_execute_node.h"

#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"

namespace tensorflow {
namespace eager {

void RemoteExecuteNode::RunAsync(StatusCallback done) {
  auto response = std::make_shared<EnqueueResponse>();

  const gtl::InlinedVector<TensorHandle*, 4>& inputs = inputs_;
  const gtl::InlinedVector<TensorHandle*, 2>& retvals = retvals_;
  Device* device = device_;

  // Filled and used only when VLOG(3) is on.
  string rpc_description;
  if (VLOG_IS_ON(3)) {
    std::vector<string> ops;
    ops.reserve(request_->queue_size());
    for (const QueueItem& item : request_->queue()) {
      if (item.has_operation()) {
        ops.push_back(item.operation().name());
      } else {
        ops.push_back(absl::StrCat("DeleteHandle(",
                                   item.handle_to_decref().op_id(), ":",
                                   item.handle_to_decref().output_num(), ")"));
      }
    }
    rpc_description =
        absl::StrCat("RemoteOperation(", absl::StrJoin(ops, ", "), ")");
  }
  VLOG(3) << "Issuing: " << rpc_description;

  CancellationManager* cm = cancellation_manager_;
  CancellationToken token = 0;
  auto call_opts = std::make_shared<CallOptions>();
  if (cm != nullptr) {
    token = cm->get_cancellation_token();
    const bool already_cancelled = !cm->RegisterCallback(
        token, [call_opts, response, done]() { call_opts->StartCancel(); });
    if (already_cancelled) {
      Status s = errors::Cancelled("RemoteExecuteNode::RunAsync");
      for (size_t i = 0; i < retvals.size(); ++i) {
        retvals[i]->PoisonRemote(s, device, context_view_id_);
      }
      done(s);
      return;
    }
  }

  for (auto handle : inputs_) {
    handle->Ref();
  }
  for (auto handle : retvals) {
    handle->Ref();
  }

  eager_client_->StreamingEnqueueAsync(
      call_opts.get(), request_.get(), response.get(),
      [inputs, retvals, call_opts, response, device,
       context_view_id = context_view_id_, rpc_description, cm, token,
       done](const Status& status) {
        if (cm != nullptr) {
          cm->TryDeregisterCallback(token);
        }
        for (auto handle : inputs) {
          handle->Unref();
        }
        if (status.ok()) {
          VLOG(3) << "Completed successfully: " << rpc_description;
        } else {
          VLOG(3) << "Failed: " << rpc_description << " with status "
                  << status.ToString();
        }
        for (size_t i = 0; i < retvals.size(); ++i) {
          if (status.ok()) {
            const string output_device =
                response->queue_response(0).device().empty()
                    ? ""
                    : response->queue_response(0).device(i);
            Status s = retvals[i]->SetRemoteShapeAndDevice(
                response->queue_response(0).shape(i), device, context_view_id,
                output_device);

            if (!s.ok()) {
              LOG(ERROR) << "Ignoring an error encountered when setting "
                            "remote shape of tensor handle: "
                         << retvals[i]
                         << " with execute status: " << status.ToString()
                         << " and SetRemoteShape status: " << s.ToString()
                         << "\nThis should never happen. "
                            "Please file an issue with the TensorFlow Team.";
            }
          } else {
            retvals[i]->PoisonRemote(status, device, context_view_id);
          }
          retvals[i]->Unref();
        }
        done(status);
      });
}

}  // namespace eager
}  // namespace tensorflow
