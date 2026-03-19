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

#ifndef TENSORFLOW_CORE_TPU_TPU_INIT_MODE_H_
#define TENSORFLOW_CORE_TPU_TPU_INIT_MODE_H_

#include "absl/status/status.h"
#include "tensorflow/core/platform/status.h"

namespace tensorflow {

enum class TPUInitMode : int { kNone, kGlobal, kRegular };

// Sets the TPU initialization mode appropriately.
//
// Requires that mode is not kNone, and mode doesn't transition kGlobal
// <-> kRegular.
//
// IMPLEMENTATION DETAILS:
// Used internally to record the current mode and type of API used for TPU
// initialization in a global static variable.
absl::Status SetTPUInitMode(TPUInitMode mode);

// Returns the current TPUInitMode.
TPUInitMode GetTPUInitMode();

namespace test {

// Forces the tpu init mode to be changed.
void ForceSetTPUInitMode(TPUInitMode mode);

}  // namespace test

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_TPU_TPU_INIT_MODE_H_
