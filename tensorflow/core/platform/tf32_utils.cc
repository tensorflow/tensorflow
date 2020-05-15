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

#include "tensorflow/core/platform/tf32_utils.h"

namespace tensorflow {

// TODO(nluehr): enable tf32 execution by default after TF32 Ampere testing.
static bool tf32_enabled = false;

void allow_tf32_execution(bool allow) { tf32_enabled = allow; }

bool tf32_execution_allowed() { return tf32_enabled; }

}  // namespace tensorflow
