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

#include "tensorflow/stream_executor/tpu/proto_helper.h"

extern "C" {

void StreamExecutor_Tpu_FreeSerializedProto(const TpuSerializedProto* proto) {
  CHECK_NE(proto, nullptr);
  CHECK_NE(proto->bytes, nullptr);
  CHECK_GT(proto->size, 0);
  delete[] proto->bytes;
}

}  // extern "C"
