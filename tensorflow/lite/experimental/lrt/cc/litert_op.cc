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

#include "tensorflow/lite/experimental/lrt/cc/litert_op.h"

#include <memory>

#include "tensorflow/lite/experimental/lrt/c/litert_common.h"
#include "tensorflow/lite/experimental/lrt/c/litert_model.h"
#include "tensorflow/lite/experimental/lrt/c/litert_op_code.h"
#include "tensorflow/lite/experimental/lrt/c/litert_support.h"

namespace litert {

LiteRtStatus LiteRtOpManager::MakeFromOp(LiteRtOp op, Unique& result) {
  result = std::make_unique<LiteRtOpManager>();
  LITERT_RETURN_STATUS_IF_NOT_OK(GetOpCode(op, &result->code_));
  result->op_ = op;
  return kLiteRtStatusOk;
}

LiteRtOpCode LiteRtOpManager::Code() const { return code_; }

LiteRtOp LiteRtOpManager::Op() { return op_; }

}  // namespace litert
