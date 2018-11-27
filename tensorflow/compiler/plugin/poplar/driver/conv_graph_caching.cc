/* Copyright 2017 Graphcore Ltd
 */

/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/plugin/poplar/driver/conv_graph_caching.h"

#include "tensorflow/compiler/plugin/poplar/driver/compiler_resources.h"
#include "tensorflow/compiler/plugin/poplar/driver/convolution_classifier.h"
#include "tensorflow/compiler/plugin/poplar/driver/graph_caching_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/util.h"

#include <poplar/Tensor.hpp>
#include <poplin/ConvUtil.hpp>
#include <poplin/Convolution.hpp>
#include <popops/ScaledAdd.hpp>
#include <poputil/GraphFunction.hpp>

namespace pe = popops::expr;

namespace xla {
namespace poplarplugin {

namespace conv_graph_caching {
namespace {

BwdWeightCacheKey GetBwdWeightCacheKey(const poplar::Tensor& weights,
                                       const poplar::Tensor& bwd_weights) {
  return {graph_caching_util::GetPoplarTensorSignature(weights),
          graph_caching_util::GetPoplarTensorSignature(bwd_weights)};
}

void CreateCachedBwdWeights(poplar::Graph& graph, CompilerResources& res,
                            const poplar::Tensor& weights,
                            const poplar::Tensor& bwd_weights,
                            poplar::program::Sequence& prog,
                            const std::string& debug_prefix) {
  auto key = GetBwdWeightCacheKey(weights, bwd_weights);
  std::vector<poplar::Tensor> args = {weights, bwd_weights};
  auto it = res.bwd_weight_graph_cache.find(key);
  if (it != res.bwd_weight_graph_cache.end()) {
    auto& f = it->second;
    f(args, prog);
    return;
  }
  using namespace poputil::graphfn;
  auto f = VoidFunction(
      graph, {input(weights, "weights"), output(bwd_weights, "bwd_weights")},
      [&](std::vector<poplar::Tensor>& args, poplar::program::Sequence& prog) {
        poplin::weightsTransposeChansFlipXY(graph, args[0], args[1], prog,
                                            debug_prefix);
        return prog;
      });
  res.bwd_weight_graph_cache.emplace(key, f);
  f(args, prog);
}

ConvolutionCacheKey GetConvolutionCacheKey(
    const poplin::ConvParams& params, const ConvClassificationType& conv_type,
    bool transpose_and_flip_weights) {
  // Create signature for the convolution input
  std::vector<std::size_t> in_shape = {params.getBatchSize(),
                                       params.getNumInputChans()};
  in_shape.insert(in_shape.end(), params.inputFieldShape.begin(),
                  params.inputFieldShape.end());
  PoplarTensorSignature in_sig(params.dType, std::move(in_shape));

  // Create signature for the weights
  std::vector<std::size_t> weights_shape = {
      params.getNumConvGroups(), params.getNumOutputChansPerConvGroup(),
      params.getNumInputChansPerConvGroup()};
  weights_shape.insert(weights_shape.end(), params.kernelShape.begin(),
                       params.kernelShape.end());
  PoplarTensorSignature weights_sig(params.dType, std::move(weights_shape));
  return std::make_tuple(in_sig, weights_sig,
                         poplin::canonicalizeParams(params), conv_type,
                         transpose_and_flip_weights);
}

WeightUpdateConvolutionCacheKey GetWeightUpdateConvolutionCacheKey(
    const poplin::ConvParams& params, const ConvClassificationType& conv_type,
    double learning_rate) {
  // Create signature for the convolution input
  std::vector<std::size_t> in_shape = {params.getBatchSize(),
                                       params.getNumInputChans()};
  in_shape.insert(in_shape.end(), params.inputFieldShape.begin(),
                  params.inputFieldShape.end());
  PoplarTensorSignature in_sig(params.dType, std::move(in_shape));

  // Create signature for the gradients
  std::vector<std::size_t> grad_shape = {params.getNumConvGroups(),
                                         params.getNumOutputChansPerConvGroup(),
                                         params.getNumInputChansPerConvGroup()};
  grad_shape.insert(grad_shape.end(), params.kernelShape.begin(),
                    params.kernelShape.end());
  PoplarTensorSignature grad_sig(params.dType, std::move(grad_shape));
  return std::make_tuple(in_sig, grad_sig, poplin::canonicalizeParams(params),
                         conv_type, learning_rate);
}
}  // namespace

poplar::Tensor DoCachedConvolution(
    poplar::Graph& graph, CompilerResources& res, const poplar::Tensor& in,
    const poplar::Tensor& weights, const poplin::ConvParams& params,
    const ConvClassificationType& conv_type, bool transpose_and_flip_weights,
    poplar::program::Sequence& prog, const std::string& debug_prefix) {
  // If this is a pass bwd convolution, try and see if we can turn it into a
  // weightsTransposeChansFlipXY and a fwd pass convolution - this allows us to
  // reause the graph for the convolution and save code space
  auto fwd_type = ConvClassificationType::FORWARD;
  auto fwd_key = conv_graph_caching::GetConvolutionCacheKey(
      params, ConvClassificationType::FORWARD, false);
  auto it = res.conv_graph_cache.find(fwd_key);

  poplar::OptionFlags opts = res.default_conv_options;
  if (conv_type == ConvClassificationType::BACKPROP_INPUT &&
      it != res.conv_graph_cache.end() &&
      !res.disable_graph_convolution_caching) {
    // We found a matching convolution in the forward pass. Transform the
    // weights prior to the convolution so we can reuse the existing
    // graph.
    opts.set("pass", ConvClassificationTypeToString(fwd_type));
    auto bwd_weights = poplin::createWeights(graph, params, "bwd_weights", opts,
                                             &res.convolution_cache);
    CreateCachedBwdWeights(graph, res, weights, bwd_weights, prog,
                           debug_prefix);
    std::vector<poplar::Tensor> args = {in, bwd_weights};
    // Execute the convolution.
    auto& f = it->second;
    return f(args, prog);
  } else {
    // Otherwise try and get a convolution, if one doesn't exist, then create it
    // and execute it
    std::vector<poplar::Tensor> args = {in, weights};
    auto key = GetConvolutionCacheKey(poplin::canonicalizeParams(params),
                                      conv_type, transpose_and_flip_weights);
    auto it = res.conv_graph_cache.find(key);
    if (it != res.conv_graph_cache.end() &&
        !res.disable_graph_convolution_caching) {
      auto& f = it->second;
      return f(args, prog);
    }
    opts.set("pass", ConvClassificationTypeToString(conv_type));
    using namespace poputil::graphfn;
    auto f = TensorFunction(graph, {input(in, "in"), input(weights, "weights")},
                            [&](std::vector<poplar::Tensor>& args,
                                poplar::program::Sequence& prog) {
                              return convolution(
                                  graph, args[0], args[1], params,
                                  transpose_and_flip_weights, prog,
                                  debug_prefix, opts, &res.convolution_cache);
                            });
    res.conv_graph_cache.emplace(key, f);
    return f(args, prog);
  }
}

Status DoCachedConvolutionWithScaledAdd(
    poplar::Graph& graph, CompilerResources& res, const poplar::Tensor& w,
    const poplar::Tensor& in, const poplar::Tensor& deltas,
    const poplin::ConvParams& params, poplar::program::Sequence& prog,
    const HloInstruction* root, const HloInstruction* conv) {
  auto conv_type = GetConvClassificationType(conv, res.annotations);

  // Get the scalar multiplier
  const auto* const_inst = root->operand(1)->operand(1)->operand(0);
  CHECK_EQ(const_inst->opcode(), HloOpcode::kConstant);

  double lr;
  TF_ASSIGN_OR_RETURN(lr, LiteralScalarDoubleToDouble(const_inst->literal()));

  switch (root->opcode()) {
    case HloOpcode::kAdd: {
      break;
    }
    case HloOpcode::kSubtract: {
      lr = -lr;
      break;
    }
    default: {
      return xla::FailedPrecondition("Unsupported scaled inplace op: %s",
                                     root->name().c_str());
    }
  }

  std::vector<poplar::Tensor> args = {in, deltas, w};

  auto key = GetWeightUpdateConvolutionCacheKey(params, conv_type, lr);
  auto it = res.wu_graph_cache.find(key);
  if (it != res.wu_graph_cache.end() &&
      !res.disable_graph_convolution_caching) {
    auto& f = it->second;
    f(args, prog);
    return Status::OK();
  }

  using namespace poputil::graphfn;
  auto f = VoidFunction(
      graph, {input(in, "in"), input(deltas, "deltas"), inout(w, "w")},
      [&](std::vector<poplar::Tensor>& args, poplar::program::Sequence& prog) {
        poplar::Tensor in_shuffled =
            ShuffleConvolutionInputToPoplar(conv, args[0]);

        poplar::Tensor deltas_shuffled =
            ShuffleConvolutionWeightsToPoplar(conv, args[1], false);
        deltas_shuffled =
            AddGroupsDimensionToWeights(params, deltas_shuffled, false);

        poplar::OptionFlags opts = res.default_conv_options;
        opts.set("pass", ConvClassificationTypeToString(conv_type));

        auto c_out = poplin::convolution(
            graph, in_shuffled, deltas_shuffled, params, false, prog,
            GetDebugName(conv), opts, &res.convolution_cache);

        c_out = ShuffleConvolutionOutputToTensorflow(conv, c_out);

        // Call the inplace op
        popops::scaledAddTo(graph, args[2], c_out, lr, prog,
                            GetDebugName(root));
      });
  res.wu_graph_cache.emplace(key, f);
  f(args, prog);
  return Status::OK();
}
}  // namespace conv_graph_caching
}  // namespace poplarplugin
}  // namespace xla
