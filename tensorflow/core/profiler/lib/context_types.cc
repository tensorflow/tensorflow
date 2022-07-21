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
#include "tensorflow/core/profiler/lib/context_types.h"

namespace tensorflow {
namespace profiler {

const char* GetContextTypeString(ContextType context_type) {
  switch (context_type) {
    case ContextType::kGeneric:
    case ContextType::kLegacy:
      return "";
    case ContextType::kTfExecutor:
      return "tf_exec";
    case ContextType::kTfrtExecutor:
      return "tfrt_exec";
    case ContextType::kSharedBatchScheduler:
      return "batch_sched";
    case ContextType::kPjRt:
      return "PjRt";
    case ContextType::kAdaptiveSharedBatchScheduler:
      return "as_batch_sched";
    case ContextType::kTfrtTpuRuntime:
      return "tfrt_rt";
    case ContextType::kTpuEmbeddingEngine:
      return "tpu_embed";
    case ContextType::kGpuLaunch:
      return "gpu_launch";
    case ContextType::kBatcher:
      return "batcher";
    case ContextType::kTpuStream:
      return "tpu_stream";
    case ContextType::kTpuLaunch:
      return "tpu_launch";
  }
}

}  // namespace profiler
}  // namespace tensorflow
