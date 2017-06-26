/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CONTRIB_RDMA_UTIL_H_
#define TENSORFLOW_CONTRIB_RDMA_UTIL_H_

#include <string>

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {

class TensorProto;

class VerbsUtil {
 public:
  // synchronous wrapper of SetProtoFromGPU
  static Status SetProtoFromGPUSync(const Tensor& tensor, Device* dev,
                                    const DeviceContext* device_context,
                                    TensorProto* proto, bool is_dead);
  static string AppendStepidToKey(const string& key, int64 step_id);
  static void GetKeyAndStepId(const string& key_with_step_id, string& key,
                              int64& step_id);
};

}  // namespace tensorflow
#endif  // TENSORFLOW_CONTRIB_RDMA_UTIL_H_
