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

#include "tensorflow/core/tpu/tpu_defs.h"

namespace tensorflow {

const char* const DEVICE_TPU_NODE = "TPU";
const char* const TPU_FAST_MEM_ATTR = "_TPU_FAST_MEM";
const char* const DEVICE_TPU_REPLICATED_CORE = "TPU_REPLICATED_CORE";
const char* const DEVICE_TPU_SYSTEM = "TPU_SYSTEM";
const char* const DEVICE_TPU_XLA_JIT = "XLA_TPU_JIT";
const char* const TPUREPLICATE_MIRRORED_VAR_INDICES_ATTR =
    "_mirrored_variable_indices";

}  // namespace tensorflow
