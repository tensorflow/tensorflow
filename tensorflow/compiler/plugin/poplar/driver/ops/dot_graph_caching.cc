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

#include "tensorflow/compiler/plugin/poplar/driver/ops/dot_graph_caching.h"

#include "tensorflow/compiler/plugin/poplar/driver/compiler_resources.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops/graph_caching_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops/ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"

#include <poplar/Tensor.hpp>
#include <poplin/MatMul.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Reduce.hpp>
#include <popops/ScaledAdd.hpp>
#include <poputil/GraphFunction.hpp>

namespace pe = popops::expr;

namespace xla {
namespace poplarplugin {

namespace dot_graph_caching {

namespace {
std::string GetMatMulPassName(MatMulPass pass) {
  switch (pass) {
    case MatMulPass::TRAINING_FWD:
      return "TRAINING_FWD";
    case MatMulPass::TRAINING_BWD:
      return "TRAINING_BWD";
    case MatMulPass::TRAINING_WU:
      return "TRAINING_WU";
    case MatMulPass::INFERENCE_FWD:
      return "INFERENCE_FWD";
  }
}
}  // namespace

poplar::Tensor DoCachedDot(poplar::Graph& graph, CompilerResources& res,
                           const poplar::Tensor& lhs, const poplar::Tensor& rhs,
                           poplar::program::Sequence& prog,
                           const MatMulPass pass, const uint64 device_id,
                           const std::string& debugPrefix) {
  const auto lhs_sig = graph_caching_util::GetPoplarTensorSignature(lhs);
  const auto rhs_sig = graph_caching_util::GetPoplarTensorSignature(rhs);

  const DotCacheKey key = {lhs_sig, rhs_sig, pass, device_id};

  auto itr = res.dot_graph_cache.find(key);

  if (itr != res.dot_graph_cache.end()) {
    std::vector<poplar::Tensor> args = {lhs, rhs};
    return itr->second(args, prog);
  } else {
    auto lamda = [&graph, &res, pass, debugPrefix](
                     std::vector<poplar::Tensor>& args,
                     poplar::program::Sequence& p) {
      poplar::OptionFlags opts;
      opts.set("fullyConnectedPass", GetMatMulPassName(pass));
      return poplin::matMulGrouped(graph, args[0], args[1], p, debugPrefix,
                                   opts, &res.dot_cache);
    };

    using namespace poputil::graphfn;
    auto tf =
        TensorFunction(graph, {input(lhs, "lhs"), input(rhs, "rhs")}, lamda);

    res.dot_graph_cache.insert({key, tf});

    std::vector<poplar::Tensor> args = {lhs, rhs};
    return tf(args, prog);
  }
}

}  // namespace dot_graph_caching
}  // namespace poplarplugin
}  // namespace xla
