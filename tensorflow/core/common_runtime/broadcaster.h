/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_BROADCASTER_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_BROADCASTER_H_

#include <vector>
#include "tensorflow/core/common_runtime/base_collective_executor.h"
#include "tensorflow/core/framework/collective.h"
#include "tensorflow/core/framework/device_attributes.pb.h"

namespace tensorflow {

// Tree-algorithm implementation of collective broadcast.
class Broadcaster {
 public:
  Broadcaster(CollectiveExecutor* col_exec, const DeviceMgr* dev_mgr,
              OpKernelContext* ctx, OpKernelContext::Params* params,
              const CollectiveParams& col_params, const string& exec_key,
              int64 step_id, Tensor* output);

  void Run(StatusCallback done);

  // Returns the rank of the device from which this device should receive
  // its value, -1 if no value should be received.
  static int TreeRecvFrom(const CollectiveParams& cp);

  // Populates targets with the ranks of the devices to which this device
  // should forward the value.
  static void TreeSendTo(const CollectiveParams& cp, std::vector<int>* targets);

 private:
  void DispatchSend(int dst_rank, const Tensor* src_tensor,
                    const StatusCallback& done);
  void DispatchRecv(int src_rank, Tensor* dst_tensor,
                    const StatusCallback& done);
  void RunTree();

  Status status_;
  CollectiveExecutor* col_exec_;  // Not owned
  const DeviceMgr* dev_mgr_;      // Not owned
  OpKernelContext* ctx_;          // Not owned
  const CollectiveParams& col_params_;
  const string exec_key_;
  const int rank_;
  const bool is_source_;
  Tensor* output_;  // Not owned
  std::unique_ptr<CollectiveAdapter> ca_;
  StatusCallback done_;
  Device* device_;  // The device for which this instance labors
  DeviceLocality device_locality_;
};

}  // namespace tensorflow
#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_BROADCASTER_H_
