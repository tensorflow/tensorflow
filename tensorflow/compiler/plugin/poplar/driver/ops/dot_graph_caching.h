/* Copyright 2019 Graphcore Ltd
 */

/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_OPS_DOT_GRAPH_CACHING_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_OPS_DOT_GRAPH_CACHING_H_

#include "tensorflow/compiler/plugin/poplar/driver/ops/graph_caching_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops/ops.h"

#include <poplar/Tensor.hpp>
#include <poputil/GraphFunction.hpp>

namespace xla {
namespace poplarplugin {
struct CompilerResources;

namespace dot_graph_caching {

// Fwd and bwd convolution caches

enum class MatMulPass {
  TRAINING_FWD,
  TRAINING_BWD,
  TRAINING_WU,
  INFERENCE_FWD
};

// The dot cache key is:
// * Shape of the rhs tensor
// * Shape of the lhs tensor
// * MatMulPass type
// * sharding device ID
using DotCacheKey = std::tuple<PoplarTensorSignature, PoplarTensorSignature,
                               MatMulPass, uint64>;
using DotGraphCache = std::map<DotCacheKey, poputil::graphfn::TensorFunction>;

poplar::Tensor DoCachedDot(poplar::Graph& graph, CompilerResources& res,
                           const poplar::Tensor& A, const poplar::Tensor& B,
                           poplar::program::Sequence& prog,
                           const MatMulPass pass, const uint64 device_id,
                           const std::string& debugPrefix = "");

}  // namespace dot_graph_caching
}  // namespace poplarplugin
}  // namespace xla

#endif
