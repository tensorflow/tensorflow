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

#include "tensorflow/compiler/plugin/poplar/driver/norm_graph_caching.h"
#include "tensorflow/compiler/plugin/poplar/driver/compiler_resources.h"
#include "tensorflow/compiler/plugin/poplar/driver/graph_caching_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/util.h"

#include <poplar/Tensor.hpp>
#include <popnn/BatchNorm.hpp>
#include <popnn/GroupNorm.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Expr.hpp>
#include <poputil/GraphFunction.hpp>

namespace pe = popops::expr;

namespace xla {
namespace poplarplugin {

namespace norm_graph_caching {
namespace {
NormInferenceCacheKey GetNormInferenceCacheKey(
    const NormType& norm_type, const poplar::Tensor& operand_shuffled,
    const poplar::Tensor& scale, const poplar::Tensor& offset,
    const poplar::Tensor& mean, const poplar::Tensor& variance_or_inv_std_dev,
    const double epsilon, absl::optional<uint32> optional_num_groups,
    const uint64 device_id) {
  return std::make_tuple(
      norm_type, graph_caching_util::GetPoplarTensorSignature(operand_shuffled),
      graph_caching_util::GetPoplarTensorSignature(scale),
      graph_caching_util::GetPoplarTensorSignature(offset),
      graph_caching_util::GetPoplarTensorSignature(mean),
      graph_caching_util::GetPoplarTensorSignature(variance_or_inv_std_dev),
      epsilon, optional_num_groups ? *optional_num_groups : 0, device_id);
}

NormTrainingCacheKey GetNormTrainingCacheKey(
    const NormType& norm_type, const poplar::Tensor& operand_shuffled,
    const poplar::Tensor& scale, const poplar::Tensor& offset,
    const double epsilon, absl::optional<uint32> optional_num_groups,
    const uint64 device_id) {
  return std::make_tuple(
      norm_type, graph_caching_util::GetPoplarTensorSignature(operand_shuffled),
      graph_caching_util::GetPoplarTensorSignature(scale),
      graph_caching_util::GetPoplarTensorSignature(offset), epsilon,
      optional_num_groups ? *optional_num_groups : 0, device_id);
}

NormGradCacheKey GetNormGradCacheKey(
    const NormType& norm_type, const poplar::Tensor& operand_shuffled,
    const poplar::Tensor& scale, const poplar::Tensor& mean,
    const poplar::Tensor& variance_or_inv_std_dev,
    const poplar::Tensor& grad_output, const double epsilon,
    absl::optional<uint32> optional_num_groups, const uint64 device_id) {
  return std::make_tuple(
      norm_type, graph_caching_util::GetPoplarTensorSignature(operand_shuffled),
      graph_caching_util::GetPoplarTensorSignature(scale),
      graph_caching_util::GetPoplarTensorSignature(mean),
      graph_caching_util::GetPoplarTensorSignature(variance_or_inv_std_dev),
      graph_caching_util::GetPoplarTensorSignature(grad_output), epsilon,
      optional_num_groups ? *optional_num_groups : 0, device_id);
}

NormStatisticsCacheKey GetNormStatisticsCacheKey(
    const NormType& norm_type, const poplar::Tensor& operand_shuffled,
    const double epsilon, absl::optional<uint32> optional_num_groups,
    const uint64 device_id) {
  return std::make_tuple(
      norm_type, graph_caching_util::GetPoplarTensorSignature(operand_shuffled),
      epsilon, optional_num_groups ? *optional_num_groups : 0, device_id);
}

poplar::Tensor ConvertVarianceToInvStdDev(poplar::Graph& graph,
                                          const poplar::Tensor& variance,
                                          const float epsilon,
                                          poplar::program::Sequence& seq,
                                          const std::string& debug_name) {
  auto expression = pe::VarianceToInvStdDev(pe::_1, pe::Const(epsilon));

  poplar::Tensor inv_sd = graph.clone(variance);
  seq.add(poplar::program::Copy(variance, inv_sd));

  popops::mapInPlace(graph, expression, {inv_sd}, seq,
                     debug_name + "/VarToInvStdDev");
  return inv_sd;
}

poplar::Tensor ConvertInvStdDevToVariance(poplar::Graph& graph,
                                          const poplar::Tensor& inv_sd,
                                          const float epsilon,
                                          poplar::program::Sequence& seq,
                                          const std::string& debug_name) {
  auto expression = pe::InvStdDevToVariance(pe::_1, pe::Const(epsilon));

  poplar::Tensor variance = graph.clone(inv_sd);
  seq.add(poplar::program::Copy(inv_sd, variance));

  popops::mapInPlace(graph, expression, {variance}, seq,
                     debug_name + "/InvStdDevToVar");
  return variance;
}

poplar::Tensor BatchNormalise(
    poplar::Graph& graph, const poplar::Tensor& operand,
    const poplar::Tensor& scale, const poplar::Tensor& offset,
    const poplar::Tensor& mean, const poplar::Tensor& inv_sd,
    poplar::program::Sequence& seq, const std::string& debug_name) {
  auto multiplicand_expression = pe::Mul(pe::_1, pe::_2);
  poplar::Tensor multiplicand =
      popops::map(graph, multiplicand_expression, {scale, inv_sd}, seq,
                  debug_name + "/Multiplicand");
  auto addend_expression = pe::Sub(pe::_1, pe::Mul(pe::_2, pe::_3));
  poplar::Tensor addend =
      popops::map(graph, addend_expression, {offset, multiplicand, mean}, seq,
                  debug_name + "/Addend");
  return popnn::bn::batchNormalise(graph, operand, multiplicand, addend, seq,
                                   debug_name);
}
}  // namespace

poplar::Tensor DoCachedNormInference(
    const NormType& norm_type, poplar::Graph& graph, CompilerResources& res,
    const poplar::Tensor& operand, const poplar::Tensor& scale,
    const poplar::Tensor& offset, const poplar::Tensor& mean,
    const poplar::Tensor& variance_or_inv_std_dev, const double epsilon,
    absl::optional<uint32> optional_num_groups, const uint64 device_id,
    poplar::program::Sequence& prog, const std::string& debug_prefix) {
  auto key = GetNormInferenceCacheKey(norm_type, operand, scale, offset, mean,
                                      variance_or_inv_std_dev, epsilon,
                                      optional_num_groups, device_id);
  std::vector<poplar::Tensor> args = {operand, scale, offset, mean,
                                      variance_or_inv_std_dev};
  auto it = res.norm_inf_graph_cache.find(key);
  if (it != res.norm_inf_graph_cache.end() &&
      !res.disable_graph_convolution_caching) {
    auto& f = it->second;
    return f(args, prog);
  }

  using namespace poputil::graphfn;
  auto f = TensorFunction(
      graph,
      {input(operand, "operand"), input(scale, "scale"),
       input(offset, "offset"), input(mean, "mean"),
       input(variance_or_inv_std_dev, "variance_or_inv_std_dev")},
      [&](std::vector<poplar::Tensor>& args, poplar::program::Sequence& seq) {
        poplar::Tensor out;
        switch (norm_type) {
          case NormType::BatchNorm: {
            // For batch norm variance_or_inv_std_dev is variance, so we need to
            // convert it.
            poplar::Tensor inv_sd = ConvertVarianceToInvStdDev(
                graph, args[4], epsilon, seq, debug_prefix);
            out = BatchNormalise(graph, args[0], args[1], args[2], args[3],
                                 inv_sd, seq, debug_prefix);
            break;
          }
          case NormType::GroupNorm: {
            // For group norm variance_or_inv_std_dev is inv_std_dev, so we
            // don't need to convert it.
            out = popnn::gn::groupNormalise(graph, args[0], args[1], args[2],
                                            args[3], args[4], seq, debug_prefix)
                      .first;
            break;
          }
        }
        return out;
      });
  res.norm_inf_graph_cache.emplace(key, f);
  return f(args, prog);
}

std::tuple<poplar::Tensor, poplar::Tensor, poplar::Tensor> DoCachedNormTraining(
    const NormType& norm_type, poplar::Graph& graph, CompilerResources& res,
    const poplar::Tensor& operand, poplar::Tensor& whitened_operand,
    const poplar::Tensor& scale, const poplar::Tensor& offset,
    const double epsilon, absl::optional<uint32> optional_num_groups,
    const uint64 device_id, poplar::program::Sequence& prog,
    const std::string& debug_prefix) {
  auto key = GetNormTrainingCacheKey(norm_type, operand, scale, offset, epsilon,
                                     optional_num_groups, device_id);
  poplar::Tensor output, mean, variance_or_inv_std_dev;
  std::vector<poplar::Tensor> args = {operand, scale, offset,
                                      output,  mean,  variance_or_inv_std_dev};
  using namespace poputil::graphfn;
  Signature signature = {
      input(operand, "operand"), input(scale, "scale"),
      input(offset, "offset"),   created("output"),
      created("mean"),           created("variance_or_inv_std_dev")};

  const bool output_whitened_operand = norm_type == NormType::GroupNorm;
  if (output_whitened_operand) {
    args.push_back(whitened_operand);
    signature.push_back(created("whitened_operand"));
  }

  auto it = res.norm_tr_graph_cache.find(key);
  if (it != res.norm_tr_graph_cache.end() &&
      !res.disable_graph_convolution_caching) {
    auto& f = it->second;
    f(args, prog);
    if (output_whitened_operand) {
      whitened_operand = args[6];
    }
    return std::make_tuple(args[3], args[4], args[5]);
  }

  auto f = VoidFunction(
      graph, signature,
      [&](std::vector<poplar::Tensor>& args, poplar::program::Sequence& seq) {
        switch (norm_type) {
          case NormType::BatchNorm: {
            poplar::Tensor inv_sd;
            std::tie(args[4], inv_sd) = popnn::bn::batchNormStatistics(
                graph, args[0], epsilon, seq, false, poplar::FLOAT,
                debug_prefix);

            args[3] = BatchNormalise(graph, args[0], args[1], args[2], args[4],
                                     inv_sd, seq, debug_prefix);
            // For batch norm variance_or_inv_std_dev is variance, so we need to
            // convert it.
            args[5] = ConvertInvStdDevToVariance(graph, inv_sd, epsilon, seq,
                                                 debug_prefix);
            break;
          }
          case NormType::GroupNorm: {
            // For group norm variance_or_inv_std_dev is inv_std_dev, so we
            // don't need to convert it.
            std::tie(args[4], args[5]) = popnn::gn::groupNormStatistics(
                graph, args[0], epsilon, seq, *optional_num_groups, false,
                poplar::FLOAT, debug_prefix);

            std::tie(args[3], args[6]) =
                popnn::gn::groupNormalise(graph, args[0], args[1], args[2],
                                          args[4], args[5], seq, debug_prefix);
            break;
          }
        }
      });
  res.norm_tr_graph_cache.emplace(key, f);
  f(args, prog);
  if (output_whitened_operand) {
    whitened_operand = args[6];
  }
  return std::make_tuple(args[3], args[4], args[5]);
}

std::tuple<poplar::Tensor, poplar::Tensor, poplar::Tensor> DoCachedNormGrad(
    const NormType& norm_type, poplar::Graph& graph, CompilerResources& res,
    const poplar::Tensor& operand, const poplar::Tensor& scale,
    const poplar::Tensor& mean, const poplar::Tensor& variance_or_inv_std_dev,
    const poplar::Tensor& grad_output, const double epsilon,
    absl::optional<uint32> optional_num_groups, const uint64 device_id,
    poplar::program::Sequence& prog, const std::string& debug_prefix) {
  auto key = GetNormGradCacheKey(norm_type, operand, scale, mean,
                                 variance_or_inv_std_dev, grad_output, epsilon,
                                 optional_num_groups, device_id);
  poplar::Tensor operand_grad, scale_grad, offset_grad;
  std::vector<poplar::Tensor> args = {
      operand,     scale,        variance_or_inv_std_dev,
      grad_output, operand_grad, scale_grad,
      offset_grad};
  using namespace poputil::graphfn;
  Signature signature = {
      input(operand, "operand"),
      input(scale, "scale"),
      input(variance_or_inv_std_dev, "variance_or_inv_std_dev"),
      input(grad_output, "grad_output"),
      created("operand_grad"),
      created("scale_grad"),
      created("offset_grad")};

  const bool operand_is_whitened = norm_type == NormType::GroupNorm;
  // We only need the mean if the operand is not whitened.
  if (!operand_is_whitened) {
    args.push_back(mean);
    signature.push_back(input(mean, "mean"));
  }

  auto it = res.norm_grad_graph_cache.find(key);
  if (it != res.norm_grad_graph_cache.end() &&
      !res.disable_graph_convolution_caching) {
    auto& f = it->second;
    f(args, prog);
    return std::make_tuple(args[4], args[5], args[6]);
  }
  using namespace poputil::graphfn;
  auto f = VoidFunction(
      graph, signature,
      [&](std::vector<poplar::Tensor>& args, poplar::program::Sequence& seq) {
        switch (norm_type) {
          case NormType::BatchNorm: {
            // For batch norm variance_or_inv_std_dev is variance, so we need to
            // convert it.
            poplar::Tensor inv_sd = ConvertVarianceToInvStdDev(
                graph, args[2], epsilon, seq, debug_prefix);
            poplar::Tensor operand_whitened =
                popnn::bn::batchNormWhiten(graph, args[0], args[7], inv_sd, seq,
                                           debug_prefix + "/WhitenedActs");

            // Compute the grad for the operand.
            args[4] = popnn::bn::batchNormGradients(
                graph, operand_whitened, args[3], inv_sd, args[1], seq,
                poplar::FLOAT, debug_prefix + "/OperandGrad");
            // Compute the grads for the scale and offset.
            std::tie(args[5], args[6]) = popnn::bn::batchNormParamGradients(
                graph, operand_whitened, args[3], seq, poplar::FLOAT,
                debug_prefix + "/ScaleOffsetGrads");
            break;
          }
          case NormType::GroupNorm: {
            // For group norm variance_or_inv_std_dev is inv_std_dev, so we
            // don't need to convert it.
            // For group norm we are using a whitened version of the operand.
            // Compute the grad for the operand.
            args[4] = popnn::gn::groupNormGradients(
                graph, args[0], args[3], args[2], args[1], seq, poplar::FLOAT,
                debug_prefix + "/OperandGrad");
            // Compute the grads for the scale and offset.
            std::tie(args[5], args[6]) = popnn::gn::groupNormParamGradients(
                graph, args[0], args[3], seq, poplar::FLOAT,
                debug_prefix + "/ScaleOffsetGrads");
            break;
          }
        }
      });
  res.norm_grad_graph_cache.emplace(key, f);
  f(args, prog);
  return std::make_tuple(args[4], args[5], args[6]);
}

std::tuple<poplar::Tensor, poplar::Tensor> DoCachedNormStatistics(
    const NormType& norm_type, poplar::Graph& graph, CompilerResources& res,
    const poplar::Tensor& operand, const double epsilon,
    absl::optional<uint32> optional_num_groups, const uint64 device_id,
    poplar::program::Sequence& prog, const std::string& debug_prefix) {
  auto key = GetNormStatisticsCacheKey(norm_type, operand, epsilon,
                                       optional_num_groups, device_id);
  poplar::Tensor mean, variance_or_inv_std_dev;
  std::vector<poplar::Tensor> args = {operand, mean, variance_or_inv_std_dev};
  auto it = res.norm_statistics_graph_cache.find(key);
  if (it != res.norm_statistics_graph_cache.end() &&
      !res.disable_graph_convolution_caching) {
    auto& f = it->second;
    f(args, prog);
    return std::make_tuple(args[1], args[2]);
  }

  using namespace poputil::graphfn;
  auto f = VoidFunction(
      graph,
      {input(operand, "operand"), created("mean"),
       created("variance_or_inv_std_dev")},
      [&](std::vector<poplar::Tensor>& args, poplar::program::Sequence& seq) {
        poplar::Tensor inv_sd;
        switch (norm_type) {
          case NormType::BatchNorm: {
            std::tie(args[1], inv_sd) = popnn::bn::batchNormStatistics(
                graph, args[0], epsilon, seq, false, poplar::FLOAT,
                debug_prefix);
            // For batch norm variance_or_inv_std_dev is variance, so we need to
            // convert it.
            args[2] = ConvertInvStdDevToVariance(graph, inv_sd, epsilon, seq,
                                                 debug_prefix);
            break;
          }
          case NormType::GroupNorm: {
            // For group norm variance_or_inv_std_dev is inv_std_dev, so we
            // don't need to convert it.
            std::tie(args[1], args[2]) = popnn::gn::groupNormStatistics(
                graph, args[0], epsilon, seq, *optional_num_groups, false,
                poplar::FLOAT, debug_prefix);
            break;
          }
        }
      });
  res.norm_statistics_graph_cache.emplace(key, f);
  f(args, prog);
  return std::make_tuple(args[1], args[2]);
}

}  // namespace norm_graph_caching
}  // namespace poplarplugin
}  // namespace xla
