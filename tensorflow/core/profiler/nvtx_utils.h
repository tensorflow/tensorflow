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

#ifndef TENSORFLOW_CORE_PLATFORM_NVTX_H_
#define TENSORFLOW_CORE_PLATFORM_NVTX_H_

#include "third_party/nvtx3/nvToolsExt.h"

// #include "tensorflow/core/framework/attr_value.pb.h"
// #include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
// #include "tensorflow/core/framework/types.pb.h"
// #include "tensorflow/core/util/env_var.h"

namespace tensorflow {

namespace nvtx {

class NvtxDomain {
 public:
  explicit NvtxDomain(const char* name) : handle_(nvtxDomainCreateA(name)) {}
  ~NvtxDomain() { nvtxDomainDestroy(handle_); }
  operator nvtxDomainHandle_t() const { return handle_; }

 private:
  nvtxDomainHandle_t handle_;
  TF_DISALLOW_COPY_AND_ASSIGN(NvtxDomain);
};

// A helper function to decide whether to enable CUDA NVTX profiling ranges.
bool NvtxRangesEnabled();

// A helper function to decide whether to enable CUDA NVTX profiling ranges
// with detailed node information.
bool NvtxRangesDetailedEnabled();

string DataTypeToNumpyString(DataType dtype);

// TODO(benbarsdell): This is a bit crude and hacky (and inefficient).
string AttrValueToJson(const AttrValue& attr_value);

string MaybeGetNvtxDomainRangeMessage(const OpKernel* op_kernel,
                                      const int num_inputs,
                                      std::vector<const TensorShape*> input_shape_array);

nvtxRangeId_t MaybeNvtxDomainRangeStart(string node_op, string node_name);

nvtxRangeId_t MaybeNvtxDomainRangeStartMsg(string msg, string node_op);

void MaybeNvtxDomainRangeEnd(nvtxRangeId_t nvtx_range);

}  // namespace nvtx
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PLATFORM_NVTX_H_
