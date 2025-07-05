/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_TPU_KERNELS_TPU_CONFIGURATION_OPS_FALLBACK_H_
#define TENSORFLOW_CORE_TPU_KERNELS_TPU_CONFIGURATION_OPS_FALLBACK_H_

// Fallback implementation for public TPU configuration ops when the graph
// rewrite pass doesn't work properly, such as in Google Colab V2 environment.

namespace tensorflow {

// Helper functions for TPU fallback operations
bool CheckTpuLibraryAvailability();
void LogTpuEnvironmentInfo();

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_TPU_KERNELS_TPU_CONFIGURATION_OPS_FALLBACK_H_
