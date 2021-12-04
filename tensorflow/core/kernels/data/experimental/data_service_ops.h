/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_KERNELS_DATA_EXPERIMENTAL_DATA_SERVICE_OPS_H_
#define TENSORFLOW_CORE_KERNELS_DATA_EXPERIMENTAL_DATA_SERVICE_OPS_H_

#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/kernels/data/iterator_ops.h"

namespace tensorflow {
namespace data {

// Registers a dataset with the tf.data service.
//
// The address and protocol inputs are used to connect to the dispatcher.
// The external state policy attribute determines whether to ignore, warn, or
// error out when the dataset contains external state.
// The op produces a dataset id for identifying the registered dataset.
class RegisterDatasetOp : public OpKernel {
 public:
  static constexpr const char* const kAddress = "address";
  static constexpr const char* const kProtocol = "protocol";
  static constexpr const char* const kExternalStatePolicy =
      "external_state_policy";
  static constexpr const char* const kElementSpec = "element_spec";
  static constexpr const char* const kMetadata = "metadata";
  static constexpr const char* const kTimeoutMs = "timeout_ms";

  explicit RegisterDatasetOp(OpKernelConstruction* ctx);

  void Compute(OpKernelContext* ctx) override;

 private:
  SerializationContext::ExternalStatePolicy external_state_policy_;
  std::string element_spec_;
  std::string serialized_metadata_;
};

}  // namespace data
}  // namespace tensorflow
#endif  // TENSORFLOW_CORE_KERNELS_DATA_EXPERIMENTAL_DATA_SERVICE_OPS_H_
