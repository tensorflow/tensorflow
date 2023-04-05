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

#if !defined(PLATFORM_GOOGLE)
#include "tensorflow/compiler/xla/stream_executor/tpu/tpu_initializer_helper.h"
#include "tensorflow/core/platform/status.h"
#endif


namespace tensorflow {
namespace tpu {
namespace {
Status InitializeTpuLibrary() {
  Status status = FindAndLoadTpuLibrary();
  if (!status.ok()) {
    LOG(INFO) << "FindAndLoadTpuLibrary failed with " << status.ToString()
              << ". This is expected if TPU is not used.";
  }
  return status;
}

#if !defined(PLATFORM_GOOGLE)
static Status tpu_library_finder = InitializeTpuLibrary();
#endif
}  // namespace
}  // namespace tpu
}  // namespace tensorflow
