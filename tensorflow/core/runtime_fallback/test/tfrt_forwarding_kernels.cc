/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

// TFRT kernels for testing tfrt_forwarding delegate.

#include "tensorflow/core/framework/tensor.h"
#include "tfrt/host_context/kernel_registry.h"  // from @tf_runtime
#include "tfrt/host_context/kernel_utils.h"  // from @tf_runtime

namespace tensorflow {
// Returns an initialized 5D tensor with all dimensions
// with size 1 and all values equal to "value".
static void TFDConstantTensor5D(tfrt::Argument<int32_t> value,
                                tfrt::Result<Tensor> tensor) {
  Tensor out(DT_INT32, TensorShape({1, 1, 1, 1, 1}));
  out.flat<int32>()(0) = value.get();
  tensor.Emplace(out);
}

void RegisterTFDForwardingTestKernels(tfrt::KernelRegistry* registry) {
  registry->AddKernel("tfd.constant_tensor5D",
                      TFRT_KERNEL(TFDConstantTensor5D));
}
}  // namespace tensorflow
