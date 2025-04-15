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

// This file uses a static constructor to automatically register kernels
// defined in tfd_forwarding_test_kernels.cc.

#include "tfrt/host_context/kernel_registry.h"  // from @tf_runtime

namespace tensorflow {

void RegisterTFDForwardingTestKernels(::tfrt::KernelRegistry* registry);

TFRT_STATIC_KERNEL_REGISTRATION(RegisterTFDForwardingTestKernels);

}  // namespace tensorflow
