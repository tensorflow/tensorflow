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
#include <popops/Cast.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Reduce.hpp>
#include <popops/ScaledAdd.hpp>
#include <poputil/GraphFunction.hpp>

namespace pe = popops::expr;

namespace xla {
namespace poplarplugin {

namespace conv_graph_caching {
namespace {

poplar::OptionFlags GetConvolutionOptions(
    CompilerResources& res, const ConvClassificationType conv_type) {
  poplar::OptionFlags opts = res.default_conv_options;
  opts.set("pass", ConvClassificationTypeToString(conv_type));
  return opts;
}

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
  PoplarTensorSignature in_sig(params.inputType, std::move(in_shape));

  // Create signature for the weights
  std::vector<std::size_t> weights_shape = {
      params.getNumConvGroups(), params.getNumOutputChansPerConvGroup(),
      params.getNumInputChansPerConvGroup()};
  weights_shape.insert(weights_shape.end(), params.kernelShape.begin(),
                       params.kernelShape.end());
  PoplarTensorSignature weights_sig(params.inputType, std::move(weights_shape));
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
  PoplarTensorSignature in_sig(params.inputType, std::move(in_shape));

  // Create signature for the gradients
  std::vector<std::size_t> grad_shape = {params.getNumConvGroups(),
                                         params.getNumOutputChansPerConvGroup(),
                                         params.getNumInputChansPerConvGroup()};
  grad_shape.insert(grad_shape.end(), params.kernelShape.begin(),
                    params.kernelShape.end());
  PoplarTensorSignature grad_sig(params.inputType, std::move(grad_shape));
  return std::make_tuple(in_sig, grad_sig, poplin::canonicalizeParams(params),
                         conv_type, learning_rate_is_constant, learning_rate,
                         op_type, device_id);
}

BiasApplyCacheKey GetBiasApplyCacheKey(
    const poplar::Tensor& input, const poplar::Tensor& deltas,
    const poplar::Tensor& scale, const std::vector<std::size_t>& reduction_dims,
    const uint64 device_id) {
  return std::make_tuple(graph_caching_util::GetPoplarTensorSignature(input),
                         graph_caching_util::GetPoplarTensorSignature(deltas),
                         graph_caching_util::GetPoplarTensorSignature(scale),
                         reduction_dims, device_id);
}
}  // namespace

poplar::Tensor DoCachedConvolution(
    poplar::Graph& graph, CompilerResources& res, const poplar::Tensor& in,
    const poplar::Tensor& input_weights, const poplin::ConvParams& params,
    const ConvClassificationType& input_conv_type,
    bool input_transpose_and_flip_weights, const uint64 device_id,
    poplar::program::Sequence& prog, const std::string& debug_prefix) {
  // If this is a pass bwd convolution, turn it into a
  // weightsTransposeChansFlipXY and a fwd pass convolution - this allows us to
  // reuse the graph for the convolution and save code space.
  ConvClassificationType conv_type = input_conv_type;
  poplar::Tensor weights = input_weights;
  bool transpose_and_flip_weights = input_transpose_and_flip_weights;
  // If this is a backprop input convolution perform the
  // weightsTransposeChansFlipXY on weights.
  if (conv_type == ConvClassificationType::BACKPROP_INPUT &&
      !res.disable_graph_convolution_caching && transpose_and_flip_weights) {
    conv_type = ConvClassificationType::FORWARD;
    transpose_and_flip_weights = false;
    auto fwd_opts = GetConvolutionOptions(res, conv_type);
    auto bwd_weights = poplin::createWeights(graph, params, "bwd_weights",
                                             fwd_opts, &res.convolution_cache);
    CreateCachedBwdWeights(graph, res, weights, bwd_weights, device_id, prog,
                           debug_prefix);
    weights = bwd_weights;
  }
  // Perform the convolution.
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
  auto opts = GetConvolutionOptions(res, conv_type);

  if (VLOG_IS_ON(2)) {
    std::stringstream stream;
    poplin::reportPlanInfo(stream, graph, params, opts, &res.convolution_cache);
    VLOG(2) << "Convolution " << debug_prefix << ". Type "
            << ConvClassificationTypeToString(conv_type) << ". Plan "
            << stream.str();
  }

  using namespace poputil::graphfn;
  auto f = TensorFunction(
      graph, {input(in, "in"), input(weights, "weights")},
      [&](std::vector<poplar::Tensor>& args, poplar::program::Sequence& prog) {
        return convolution(graph, args[0], args[1], params,
                           transpose_and_flip_weights, prog, debug_prefix, opts,
                           &res.convolution_cache);
      });
  res.conv_graph_cache.emplace(key, f);
  return f(args, prog);
}

Status DoCachedConvolutionScaledInplace(
    poplar::Graph& graph, CompilerResources& res, const poplar::Tensor& w,
    const poplar::Tensor& in, const poplar::Tensor& deltas,
    const poplin::ConvParams& params, const uint64 device_id,
    poplar::program::Sequence& prog, const HloInstruction* inst,
    TensorMap& tensor_map) {
  const auto operand_count = inst->operand_count();
  if (!(operand_count == 3 || operand_count == 4)) {
    return xla::FailedPrecondition(
        "Unsupported use of scaled inplace op: %s. Too many operands, "
        "expected 3 or 4, got %d.",
        inst->name().c_str(), operand_count);
  }

  std::vector<poplar::Tensor> args = {in, deltas, w};
  auto conv_type = GetConvClassificationType(inst, res.annotations);
  auto opts = GetConvolutionOptions(res, conv_type);

  // Handle both constant and variant (Tensor) scale.
  const bool scale_is_constant = operand_count == 3;
  double constant_scale = 0.0;
  poplar::Tensor variable_scale;

  // Get the root of the fusion - it indicates whether this is add or subtract.
  const auto* root_inst =
      inst->fused_instructions_computation()->root_instruction();
  auto op_type = root_inst->opcode();

  if (scale_is_constant) {
    // Get the constant scale.
    const auto* const_inst = root_inst->operand(1)->operand(1)->operand(0);
    CHECK_EQ(const_inst->opcode(), HloOpcode::kConstant);

    TF_ASSIGN_OR_RETURN(constant_scale, LiteralScalarToNativeType<double>(
                                            const_inst->literal()));
  } else {
    // Get the variable scale.
    TF_ASSIGN_OR_RETURN(
        variable_scale,
        FindInstructionInput(tensor_map, res, inst, 3, prog, false));
    args.push_back(variable_scale);
  }

  auto key = GetConvolutionScaledInplaceCacheKey(
      params, conv_type, scale_is_constant, constant_scale, op_type, device_id);
  auto it = res.conv_scaled_inplace_graph_cache.find(key);
  if (it != res.conv_scaled_inplace_graph_cache.end() &&
      !res.disable_graph_convolution_caching) {
    auto& f = it->second;
    f(args, prog);
    return Status::OK();
  }

  using namespace poputil::graphfn;
  Signature signature = {
      input(in, "in"),
      input(deltas, "deltas"),
      inout(w, "w"),
  };
  if (!scale_is_constant) {
    // Add the variable scale to the signature.
    signature.push_back(input(variable_scale, "variable_scale"));
  }

  auto f = VoidFunction(
      graph, signature,
      [&](std::vector<poplar::Tensor>& args, poplar::program::Sequence& seq) {
        auto c_out = poplin::convolution(graph, args[0], args[1], params, false,
                                         seq, GetDebugName(inst), opts,
                                         &res.convolution_cache);
        if (scale_is_constant) {
          TF_CHECK_OK(ScaledInplaceConstantOrTensor(
              graph, args[2], c_out, constant_scale, seq, op_type,
              GetDebugName(inst)));
        } else {
          TF_CHECK_OK(ScaledInplaceConstantOrTensor(graph, args[2], c_out,
                                                    args[3], seq, op_type,
                                                    GetDebugName(inst)));
        }
      });
  res.conv_scaled_inplace_graph_cache.emplace(key, f);
  f(args, prog);
  return Status::OK();
}

Status DoCachedBiasApply(poplar::Graph& graph, CompilerResources& res,
                         const poplar::Tensor& in, const poplar::Tensor& deltas,
                         const std::vector<std::size_t> reduction_dims,
                         const uint64 device_id,
                         poplar::program::Sequence& prog,
                         const HloInstruction* inst, TensorMap& tensor_map) {
  const auto operand_count = inst->operand_count();
  if (!(operand_count == 2 || operand_count == 3)) {
    return xla::FailedPrecondition(
        "Unsupported use of bias apply op: %s. Too many operands, "
        "expected 2 or 3, got %d.",
        inst->name().c_str(), operand_count);
  }

  poplar::Tensor scale;
  const bool scale_is_constant = operand_count == 2;
  if (scale_is_constant) {
    // Get the constant scale.
    const auto* root_inst =
        inst->fused_instructions_computation()->root_instruction();
    const auto* const_inst = root_inst->operand(1)->operand(1)->operand(0);
    CHECK_EQ(const_inst->opcode(), HloOpcode::kConstant);

    TF_ASSIGN_OR_RETURN(Literal lit, const_inst->literal().Convert(F32));

    TF_ASSIGN_OR_RETURN(
        scale, AddConstantTensor(graph, {const_inst, 0}, const_inst->shape(),
                                 lit, res, tensor_map));

  } else {
    // Get the variable scale.
    TF_ASSIGN_OR_RETURN(
        scale, FindInstructionInput(tensor_map, res, inst, 2, prog, false));
  }

  std::vector<poplar::Tensor> args = {in, deltas, scale};

  auto key = GetBiasApplyCacheKey(in, deltas, scale, reduction_dims, device_id);
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
        auto scale_float = args[2];
        if (scale_float.elementType() != poplar::FLOAT) {
          scale_float = popops::cast(graph, scale_float, poplar::FLOAT, seq,
                                     GetDebugName(inst) + "/ScaleToFloat");
        }
        // Reduce with scale and update in place
        popops::mapInPlace(graph, popops::expr::UnaryOpType::NEGATE,
                           scale_float, seq, GetDebugName(inst) + "/negate");
        popops::reduceWithOutput(graph, args[1], args[0], reduction_dims,
                                 {popops::Operation::ADD, true, scale_float},
                                 seq, GetDebugName(inst));
      });
  res.bias_apply_graph_cache.emplace(key, f);
  f(args, prog);
  return Status::OK();
}
}  // namespace conv_graph_caching
}  // namespace poplarplugin
}  // namespace xla
