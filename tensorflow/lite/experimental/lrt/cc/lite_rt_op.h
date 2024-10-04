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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LRT_CC_LITE_RT_OP_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LRT_CC_LITE_RT_OP_H_

#include <memory>

#include "tensorflow/lite/experimental/lrt/c/lite_rt_model.h"
#include "tensorflow/lite/experimental/lrt/c/lite_rt_op_code.h"

namespace lrt {

// [WIP] Simple C++ wrapper over the C op api. Provided for convenience.
//
// NOTE ON USAGE: This "unpacks" upfront some of the data behind the LrtOp
// for efficiency and a cleaner interface (no status checks needed on getters).
// Becasuse of this, it is required that `op : LrtOp` is stable and
// unmutated throughout the lifetime. This is guaranteed within (but not
// between) calls to an LrtCompilerPlugin. Plugins should close all
// LrtOpManagers before exiting a call and initialize fresh ones in later calls.
//
// This is an evolution of "graph_tools" and logic will be consolidated in
// the future.
//
// TODO: Expand this abstraction to handle options and edges (as
// LrtTensorManagers).
class LrtOpManager {
 public:
  using Unique = std::unique_ptr<LrtOpManager>;

  static LrtStatus MakeFromOp(LrtOp op, Unique& result);

  LrtOpCode Code() const;

  LrtOp Op();

 private:
  LrtOp op_;

  LrtOpCode code_;
};

}  // namespace lrt

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LRT_CC_LITE_RT_OP_H_
