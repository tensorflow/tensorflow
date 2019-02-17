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

#include "tensorflow/compiler/plugin/poplar/driver/ops/conv_graph_caching.h"

#include "tensorflow/compiler/plugin/poplar/driver/compiler_resources.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops/graph_caching_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops/ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/convolution_classifier.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"

#include <poplar/Tensor.hpp>
#include <poplin/ConvUtil.hpp>
#include <poplin/Convolution.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Reduce.hpp>
#include <popops/ScaledAdd.hpp>
#include <poputil/GraphFunction.hpp>

namespace pe = popops::expr;

namespace xla {
namespace poplarplugin {

namespace conv_graph_caching {
namespace {

BwdWeightCacheKey GetBwdWeightCacheKey(const poplar::Tensor& weights,
                                       const poplar::Tensor& bwd_weights,
                                       const uint64 device_id) {
  return std::make_tuple(
      graph_caching_util::GetPoplarTensorSignature(weights),
      graph_caching_util::GetPoplarTensorSignature(bwd_weights), device_id);
}

void CreateCachedBwdWeights(poplar::Graph& graph, CompilerResources& res,
                            const poplar::Tensor& weights,
                            const poplar::Tensor& bwd_weights,
                            const uint64 device_id,
                            poplar::program::Sequence& prog,
                            const std::string& debug_prefix) {
  auto key = GetBwdWeightCacheKey(weights, bwd_weights, device_id);
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
    bool transpose_and_flip_weights, const uint64 device_id) {
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
                         transpose_and_flip_weights, device_id);
}

ConvolutionScaledInplaceCacheKey GetConvolutionScaledInplaceCacheKey(
    const poplin::ConvParams& params, const ConvClassificationType& conv_type,
    const bool learning_rate_is_constant, const double learning_rate,
    const HloOpcode op_type, const uint64 device_id) {
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
                         conv_type, learning_rate_is_constant, learning_rate,
                         op_type, device_id);
}

BiasApplyCacheKey GetBiasApplyCacheKey(
    const poplar::Tensor& input, const poplar::Tensor& deltas,
    const std::vector<std::size_t>& reduction_dims,
    const bool learning_rate_is_constant, const double learning_rate,
    const uint64 device_id) {
  return std::make_tuple(graph_caching_util::GetPoplarTensorSignature(input),
                         graph_caching_util::GetPoplarTensorSignature(deltas),
                         reduction_dims, learning_rate_is_constant,
                         learning_rate, device_id);
}
}  // namespace

poplar::Tensor DoCachedConvolution(
    poplar::Graph& graph, CompilerResources& res, const poplar::Tensor& in,
    const poplar::Tensor& weights, const poplin::ConvParams& params,
    const ConvClassificationType& conv_type, bool transpose_and_flip_weights,
    const uint64 device_id, poplar::program::Sequence& prog,
    const std::string& debug_prefix) {
  // If this is a pass bwd convolution, try and see if we can turn it into a
  // weightsTransposeChansFlipXY and a fwd pass convolution - this allows us to
  // reause the graph for the convolution and save code space
  auto fwd_type = ConvClassificationType::FORWARD;
  auto fwd_key = conv_graph_caching::GetConvolutionCacheKey(
      params, ConvClassificationType::FORWARD, false, device_id);
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
    CreateCachedBwdWeights(graph, res, weights, bwd_weights, device_id, prog,
                           debug_prefix);
    std::vector<poplar::Tensor> args = {in, bwd_weights};
    // Execute the convolution.
    auto& f = it->second;
    return f(args, prog);
  } else {
    // Otherwise try and get a convolution, if one doesn't exist, then create it
    // and execute it
    std::vector<poplar::Tensor> args = {in, weights};
    auto key =
        GetConvolutionCacheKey(poplin::canonicalizeParams(params), conv_type,
                               transpose_and_flip_weights, device_id);
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

namespace {

poplar::Tensor DoCachedConvolutionScaledInplace(
    poplar::Graph& graph, CompilerResources& res, const poplar::Tensor& in,
    const poplar::Tensor& deltas, const poplin::ConvParams& params,
    const ConvClassificationType conv_type, poplar::program::Sequence& seq,
    const HloInstruction* inst) {
  poplar::Tensor in_shuffled = ShuffleConvolutionInputToPoplar(inst, in);

  poplar::Tensor deltas_shuffled =
      ShuffleConvolutionWeightsToPoplar(inst, deltas, false);
  deltas_shuffled = AddGroupsDimensionToWeights(params, deltas_shuffled, false);

  poplar::OptionFlags opts = res.default_conv_options;
  opts.set("pass", ConvClassificationTypeToString(conv_type));

  auto c_out = poplin::convolution(graph, in_shuffled, deltas_shuffled, params,
                                   false, seq, GetDebugName(inst), opts,
                                   &res.convolution_cache);

  return ShuffleConvolutionOutputToTensorflow(inst, c_out);
}

Status DoCachedConvolutionScaledInplaceConstLearningRate(
    poplar::Graph& graph, CompilerResources& res, const poplar::Tensor& w,
    const poplar::Tensor& in, const poplar::Tensor& deltas,
    const poplin::ConvParams& params, const uint64 device_id,
    poplar::program::Sequence& prog, const HloInstruction* inst) {
  auto conv_type = GetConvClassificationType(inst, res.annotations);

  const auto* root_inst =
      inst->fused_instructions_computation()->root_instruction();

  // Get the constant learning rate.
  const auto* const_inst = root_inst->operand(1)->operand(1)->operand(0);
  CHECK_EQ(const_inst->opcode(), HloOpcode::kConstant);

  TF_ASSIGN_OR_RETURN(double const_lr,
                      LiteralScalarToNativeType<double>(const_inst->literal()));

  std::vector<poplar::Tensor> args = {in, deltas, w};
  auto op_type = root_inst->opcode();

  auto key = GetConvolutionScaledInplaceCacheKey(params, conv_type, true,
                                                 const_lr, op_type, device_id);
  auto it = res.conv_scaled_inplace_graph_cache.find(key);
  if (it != res.conv_scaled_inplace_graph_cache.end() &&
      !res.disable_graph_convolution_caching) {
    auto& f = it->second;
    f(args, prog);
    return Status::OK();
  }
  using namespace poputil::graphfn;
  auto f = VoidFunction(
      graph, {input(in, "in"), input(deltas, "deltas"), inout(w, "w")},
      [&](std::vector<poplar::Tensor>& args, poplar::program::Sequence& seq) {
        auto c_out = DoCachedConvolutionScaledInplace(
            graph, res, args[0], args[1], params, conv_type, seq, inst);
        TF_CHECK_OK(ScaledInplaceConstantOrTensor(
            graph, args[2], c_out, const_lr, seq, op_type, GetDebugName(inst)));
      });

  res.conv_scaled_inplace_graph_cache.emplace(key, f);
  f(args, prog);
  return Status::OK();
}

Status DoCachedConvolutionScaledInplaceVariableLearningRate(
    poplar::Graph& graph, CompilerResources& res, const poplar::Tensor& w,
    const poplar::Tensor& in, const poplar::Tensor& deltas,
    const poplar::Tensor& scale, const poplin::ConvParams& params,
    const uint64 device_id, poplar::program::Sequence& prog,
    const HloInstruction* inst) {
  auto conv_type = GetConvClassificationType(inst, res.annotations);

  const auto* root_inst =
      inst->fused_instructions_computation()->root_instruction();

  std::vector<poplar::Tensor> args = {in, deltas, scale, w};
  auto op_type = root_inst->opcode();

  auto key = GetConvolutionScaledInplaceCacheKey(params, conv_type, false, 0.0,
                                                 op_type, device_id);
  auto it = res.conv_scaled_inplace_graph_cache.find(key);
  if (it != res.conv_scaled_inplace_graph_cache.end() &&
      !res.disable_graph_convolution_caching) {
    auto& f = it->second;
    f(args, prog);
    return Status::OK();
  }

  using namespace poputil::graphfn;
  auto f = VoidFunction(
      graph,
      {input(in, "in"), input(deltas, "deltas"), input(scale, "scale"),
       inout(w, "w")},
      [&](std::vector<poplar::Tensor>& args, poplar::program::Sequence& seq) {
        auto c_out = DoCachedConvolutionScaledInplace(
            graph, res, args[0], args[1], params, conv_type, seq, inst);
        TF_CHECK_OK(ScaledInplaceConstantOrTensor(
            graph, args[3], c_out, args[2], seq, op_type, GetDebugName(inst)));
      });
  res.conv_scaled_inplace_graph_cache.emplace(key, f);
  f(args, prog);
  return Status::OK();
}
}  // namespace

Status DoCachedConvolutionScaledInplace(
    poplar::Graph& graph, CompilerResources& res, const poplar::Tensor& w,
    const poplar::Tensor& in, const poplar::Tensor& deltas,
    const poplin::ConvParams& params, const uint64 device_id,
    poplar::program::Sequence& prog, const HloInstruction* inst,
    TensorMap& tensor_map) {
  if (inst->operand_count() == 3) {
    return DoCachedConvolutionScaledInplaceConstLearningRate(
        graph, res, w, in, deltas, params, device_id, prog, inst);

  } else if (inst->operand_count() == 4) {
    TF_ASSIGN_OR_RETURN(
        poplar::Tensor scale,
        FindInstructionInput(tensor_map, res, inst, 3, prog, false));
    return DoCachedConvolutionScaledInplaceVariableLearningRate(
        graph, res, w, in, deltas, scale, params, device_id, prog, inst);
  } else {
    return xla::FailedPrecondition("Unsupported use of scaled inplace op: %s",
                                   inst->name().c_str());
  }
}

namespace {
Status DoCachedBiasApplyConstLearningRate(
    poplar::Graph& graph, CompilerResources& res, const poplar::Tensor& in,
    const poplar::Tensor& deltas,
    const std::vector<std::size_t>& reduction_dims, const uint64 device_id,
    poplar::program::Sequence& prog, const HloInstruction* inst,
    TensorMap& tensor_map) {
  const auto* root_inst =
      inst->fused_instructions_computation()->root_instruction();

  // Get the constant learning rate.
  const auto* const_inst = root_inst->operand(1)->operand(1)->operand(0);
  CHECK_EQ(const_inst->opcode(), HloOpcode::kConstant);

  TF_ASSIGN_OR_RETURN(float const_lr,
                      LiteralScalarToNativeType<float>(const_inst->literal()));

  std::vector<poplar::Tensor> args = {in, deltas};

  auto key = GetBiasApplyCacheKey(in, deltas, reduction_dims, true, const_lr,
                                  device_id);
  auto it = res.bias_apply_graph_cache.find(key);
  if (it != res.bias_apply_graph_cache.end() &&
      !res.disable_graph_convolution_caching) {
    auto& f = it->second;
    f(args, prog);
    return Status::OK();
  }

  using namespace poputil::graphfn;
  auto f = VoidFunction(
      graph, {inout(in, "in"), input(deltas, "deltas")},
      [&](std::vector<poplar::Tensor>& args, poplar::program::Sequence& seq) {
        popops::reduceWithOutput(graph, args[1], args[0], reduction_dims,
                                 {popops::Operation::ADD, -const_lr, true}, seq,
                                 GetDebugName(inst));
      });
  res.bias_apply_graph_cache.emplace(key, f);
  f(args, prog);
  return Status::OK();
}

Status DoCachedBiasApplyVariableLearningRate(
    poplar::Graph& graph, CompilerResources& res, const poplar::Tensor& in,
    const poplar::Tensor& deltas, const poplar::Tensor& scale,
    const std::vector<std::size_t>& reduction_dims, const uint64 device_id,
    poplar::program::Sequence& prog, const HloInstruction* inst,
    TensorMap& tensor_map) {
  std::vector<poplar::Tensor> args = {in, deltas, scale};

  auto key =
      GetBiasApplyCacheKey(in, deltas, reduction_dims, false, 0.0, device_id);
  auto it = res.bias_apply_graph_cache.find(key);
  if (it != res.bias_apply_graph_cache.end() &&
      !res.disable_graph_convolution_caching) {
    auto& f = it->second;
    f(args, prog);
    return Status::OK();
  }

  using namespace poputil::graphfn;
  auto f = VoidFunction(
      graph,
      {inout(in, "input"), input(deltas, "deltas"), input(scale, "scale")},
      [&](std::vector<poplar::Tensor>& args, poplar::program::Sequence& seq) {
        // TODO T6513 - replace this with just a reduceWithOutput
        // Clone the input layout
        auto input_clone = graph.clone(args[0]);
        // Reduce - do not update or scale.
        popops::reduceWithOutput(graph, args[1], input_clone, reduction_dims,
                                 {popops::Operation::ADD, 1.0, false}, seq,
                                 GetDebugName(inst));
        // Do the scale and apply separately
        TF_CHECK_OK(ScaledInplaceConstantOrTensor(
            graph, args[0], input_clone, args[2], seq, HloOpcode::kSubtract,
            GetDebugName(inst)));
      });
  res.bias_apply_graph_cache.emplace(key, f);
  f(args, prog);
  return Status::OK();
}
}  // namespace

Status DoCachedBiasApply(poplar::Graph& graph, CompilerResources& res,
                         const poplar::Tensor& input,
                         const poplar::Tensor& deltas,
                         const std::vector<std::size_t> reduction_dims,
                         const uint64 device_id,
                         poplar::program::Sequence& prog,
                         const HloInstruction* inst, TensorMap& tensor_map) {
  if (inst->operand_count() == 2) {
    return DoCachedBiasApplyConstLearningRate(graph, res, input, deltas,
                                              reduction_dims, device_id, prog,
                                              inst, tensor_map);

  } else if (inst->operand_count() == 3) {
    TF_ASSIGN_OR_RETURN(
        poplar::Tensor scale,
        FindInstructionInput(tensor_map, res, inst, 2, prog, false));
    return DoCachedBiasApplyVariableLearningRate(
        graph, res, input, deltas, scale, reduction_dims, device_id, prog, inst,
        tensor_map);
  } else {
    return xla::FailedPrecondition("Unsupported use of bias apply op: %s",
                                   inst->name().c_str());
  }
}
}  // namespace conv_graph_caching
}  // namespace poplarplugin
}  // namespace xla
