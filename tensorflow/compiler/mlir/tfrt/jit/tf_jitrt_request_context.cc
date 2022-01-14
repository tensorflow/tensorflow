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

#include "tensorflow/compiler/mlir/tfrt/jit/tf_jitrt_request_context.h"

namespace tensorflow {

using ::tfrt::jitrt::JitExecutableCache;

Status SetUpTfJitRtRequestContext(tfrt::RequestContextBuilder* builder) {
  // TODO(ezhulenev): Instead of keeping JitExecutableCache in the
  // ResourceContext it can be stored in the SavedModel itself to avoid
  // quite exensive GetOrCreateResource operation.
  JitExecutableCache* jit_executable_cache =
      builder->resource_context()->GetOrCreateResource<JitExecutableCache>(
          "tf_jitrt.jit_executable_cache");

  builder->context_data().emplace<TfJitRtRequestState>(jit_executable_cache);

  return Status::OK();
}

}  // namespace tensorflow
