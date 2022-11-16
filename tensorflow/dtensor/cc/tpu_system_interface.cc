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

#include "tensorflow/dtensor/cc/tpu_system_interface.h"

namespace tensorflow {
namespace dtensor {

namespace {

static TpuSystemInterface* preferred_tpu_system = nullptr;

}  // namespace

void SetPreferredTpuSystem(TpuSystemInterface* tpu_system) {
  if (preferred_tpu_system != nullptr) {
    delete preferred_tpu_system;
  }
  preferred_tpu_system = tpu_system;
}

TpuSystemInterface* GetPreferredTpuSystem() { return preferred_tpu_system; }

}  // namespace dtensor
}  // namespace tensorflow
