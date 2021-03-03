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
#include "tensorflow/core/tpu/kernels/tpu_op_consts.h"

namespace tensorflow {
namespace tpu {
const char kCompilationCacheResourceName[] = "tpu_compilation_cache";
const char kCompiledProtoCacheResourceName[] = "tpu_proto_cache";
const char kCompilationCacheUnloaderResourceName[] =
    "tpu_compilation_cache_unloader";
const char kFingerprintLookupResourceName[] = "tpu_fingerprint_lookup";
}  // namespace tpu
}  // namespace tensorflow
