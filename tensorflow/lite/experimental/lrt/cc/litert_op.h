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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LRT_CC_LITERT_OP_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LRT_CC_LITERT_OP_H_

#include <memory>

#include "tensorflow/lite/experimental/lrt/c/litert_common.h"
#include "tensorflow/lite/experimental/lrt/c/litert_model.h"
#include "tensorflow/lite/experimental/lrt/c/litert_op_code.h"

namespace litert {

// [WIP] Simple C++ wrapper over the C op api. Provided for convenience.
//
// NOTE ON USAGE: This "unpacks" upfront some of the data behind the LiteRtOp
// for efficiency and a cleaner interface (no status checks needed on getters).
// Because of this, it is required that `op : LiteRtOp` is stable and
// unmutated throughout the lifetime. This is guaranteed within (but not
// between) calls to an LiteRtCompilerPlugin. Plugins should close all
// LiteRtOpManagers before exiting a call and initialize fresh ones in later
// calls.
//
// This is an evolution of "graph_tools" and logic will be consolidated in
// the future.
//
// TODO: Expand this abstraction to handle options and edges (as
// LiteRtTensorManagers).
class LiteRtOpManager {
 public:
  using Unique = std::unique_ptr<LiteRtOpManager>;

  static LiteRtStatus MakeFromOp(LiteRtOp op, Unique& result);

  LiteRtOpCode Code() const;

  LiteRtOp Op();

 private:
  LiteRtOp op_;

  LiteRtOpCode code_;
};

}  // namespace litert

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LRT_CC_LITERT_OP_H_
