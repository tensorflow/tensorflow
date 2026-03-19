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

#include "xla/stream_executor/cuda/cudnn_frontend_helpers.h"

namespace stream_executor {
namespace gpu {

int CuDnnTensorUID(int offset) {
  constexpr int kFirstUid = 1;
  return kFirstUid + offset;
}

}  // namespace gpu
}  // namespace stream_executor
