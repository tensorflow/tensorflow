// Copyright 2024 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tensorflow/compiler/mlir/lite/experimental/lrt/c/lite_rt_common.h"

struct LrtStatusT {
  LrtStatusCode code;
  // TODO: b/365295276 - Implement error message payloads for lrt status.
};

LrtStatusCode GetStatusCode(LrtStatus status) { return status->code; }

void StatusDestroy(LrtStatus status) { delete status; }

LrtStatus StatusCreate(LrtStatusCode code) {
  auto* res = new LrtStatusT;
  res->code = code;
  return res;
}

LrtStatus StatusOk() { return StatusCreate(kLrtStatusOk); }
