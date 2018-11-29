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

#include "tensorflow/compiler/plugin/poplar/driver/batch_norm_graph_caching.h"
#include "tensorflow/compiler/plugin/poplar/driver/compiler_resources.h"
#include "tensorflow/compiler/plugin/poplar/driver/graph_caching_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/util.h"

#include <poplar/Tensor.hpp>
#include <popnn/BatchNorm.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Expr.hpp>
#include <poputil/GraphFunction.hpp>

namespace pe = popops::expr;

namespace xla {
namespace poplarplugin {

namespace batch_norm_graph_caching {
namespace {
BatchNormInferenceCacheKey GetBatchNormInferenceCacheKey(
    const poplar::Tensor& operand_shuffled, const poplar::Tensor& scale,
    const poplar::Tensor& offset, const poplar::Tensor& mean,
    const poplar::Tensor& variance, const double epsilon) {
  return std::make_tuple(
      graph_caching_util::GetPoplarTensorSignature(operand_shuffled),
      graph_caching_util::GetPoplarTensorSignature(scale),
      graph_caching_util::GetPoplarTensorSignature(offset),
      graph_caching_util::GetPoplarTensorSignature(mean),
      graph_caching_util::GetPoplarTensorSignature(variance), epsilon);
}

BatchNormTrainingCacheKey GetBatchNormTrainingCacheKey(
    const poplar::Tensor& operand_shuffled, const poplar::Tensor& scale,
    const poplar::Tensor& offset, const double epsilon) {
  return std::make_tuple(
      graph_caching_util::GetPoplarTensorSignature(operand_shuffled),
      graph_caching_util::GetPoplarTensorSignature(scale),
      graph_caching_util::GetPoplarTensorSignature(offset), epsilon);
}

BatchNormGradCacheKey GetBatchNormGradCacheKey(
    const poplar::Tensor& operand_shuffled, const poplar::Tensor& scale,
    const poplar::Tensor& mean, const poplar::Tensor& variance,
    const poplar::Tensor& grad_output, const double epsilon) {
  return std::make_tuple(
      graph_caching_util::GetPoplarTensorSignature(operand_shuffled),
      graph_caching_util::GetPoplarTensorSignature(scale),
      graph_caching_util::GetPoplarTensorSignature(mean),
      graph_caching_util::GetPoplarTensorSignature(variance),
      graph_caching_util::GetPoplarTensorSignature(grad_output), epsilon);
}

poplar::Tensor convertVarianceToInvStdDev(poplar::Graph& graph,
                                          const poplar::Tensor& variance,
                                          const float epsilon,
                                          poplar::program::Sequence& seq,
                                          const std::string& debug_name) {
  auto expression =
      pe::Divide(pe::Const(1), pe::Sqrt(pe::Add(pe::_1, pe::Const(epsilon))));

  return popops::map(graph, expression, {variance}, seq,
                     debug_name + "/VarToInvStdDev");
}

poplar::Tensor convertInvStdDevToVariance(poplar::Graph& graph,
                                          const poplar::Tensor& inv_sd,
                                          const float epsilon,
                                          poplar::program::Sequence& seq,
                                          const std::string& debug_name) {
  auto expression =
      pe::Sub(pe::Divide(pe::Const(1), pe::Square(pe::_1)), pe::Const(epsilon));

  return popops::map(graph, expression, {inv_sd}, seq,
                     debug_name + "/InvStdDevToVar");
}

poplar::Tensor batchNormalise(
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

poplar::Tensor DoCachedBatchNormInference(
    poplar::Graph& graph, CompilerResources& res, const poplar::Tensor& operand,
    const poplar::Tensor& scale, const poplar::Tensor& offset,
    const poplar::Tensor& mean, const poplar::Tensor& variance,
    const double epsilon, poplar::program::Sequence& prog,
    const std::string& debug_prefix) {
  auto key = GetBatchNormInferenceCacheKey(operand, scale, offset, mean,
                                           variance, epsilon);
  std::vector<poplar::Tensor> args = {operand, scale, offset, mean, variance};
  auto it = res.bn_inf_graph_cache.find(key);
  if (it != res.bn_inf_graph_cache.end() &&
      !res.disable_graph_convolution_caching) {
    auto& f = it->second;
    return f(args, prog);
  }

  using namespace poputil::graphfn;
  auto f = TensorFunction(
      graph, {input(operand, "operand"), input(scale, "scale"),
              input(offset, "offset"), input(mean, "mean"),
              input(variance, "variance")},
      [&](std::vector<poplar::Tensor>& args, poplar::program::Sequence& seq) {
        poplar::Tensor inv_sd = convertVarianceToInvStdDev(
            graph, args[4], epsilon, seq, debug_prefix);
        return batchNormalise(graph, args[0], args[1], args[2], args[3], inv_sd,
                              seq, debug_prefix);
      });
  res.bn_inf_graph_cache.emplace(key, f);
  return f(args, prog);
}

std::tuple<poplar::Tensor, poplar::Tensor, poplar::Tensor>
DoCachedBatchNormTraining(poplar::Graph& graph, CompilerResources& res,
                          const poplar::Tensor& operand,
                          const poplar::Tensor& scale,
                          const poplar::Tensor& offset, const double epsilon,
                          poplar::program::Sequence& prog,
                          const std::string& debug_prefix) {
  auto key = GetBatchNormTrainingCacheKey(operand, scale, offset, epsilon);
  poplar::Tensor output, mean, variance;
  std::vector<poplar::Tensor> args = {operand, scale, offset,
                                      output,  mean,  variance};
  auto it = res.bn_tr_graph_cache.find(key);
  if (it != res.bn_tr_graph_cache.end() &&
      !res.disable_graph_convolution_caching) {
    auto& f = it->second;
    f(args, prog);
    return std::make_tuple(args[3], args[4], args[5]);
  }

  using namespace poputil::graphfn;
  auto f = VoidFunction(
      graph, {input(operand, "operand"), input(scale, "scale"),
              input(offset, "offset"), created("output"), created("mean"),
              created("variance")},
      [&](std::vector<poplar::Tensor>& args, poplar::program::Sequence& seq) {
        poplar::Tensor inv_sd;
        std::tie(args[4], inv_sd) = popnn::bn::batchNormEstimates(
            graph, args[0], epsilon, seq, false, poplar::FLOAT, debug_prefix);

        args[3] = batchNormalise(graph, args[0], args[1], args[2], args[4],
                                 inv_sd, seq, debug_prefix);

        args[5] = convertInvStdDevToVariance(graph, inv_sd, epsilon, seq,
                                             debug_prefix);
      });
  res.bn_tr_graph_cache.emplace(key, f);
  f(args, prog);
  return std::make_tuple(args[3], args[4], args[5]);
}

std::tuple<poplar::Tensor, poplar::Tensor, poplar::Tensor>
DoCachedBatchNormGrad(poplar::Graph& graph, CompilerResources& res,
                      const poplar::Tensor& operand,
                      const poplar::Tensor& scale, const poplar::Tensor& mean,
                      const poplar::Tensor& variance,
                      const poplar::Tensor& grad_output, const double epsilon,
                      poplar::program::Sequence& prog,
                      const std::string& debug_prefix) {
  auto key = GetBatchNormGradCacheKey(operand, scale, mean, variance,
                                      grad_output, epsilon);
  poplar::Tensor operand_grad, scale_grad, offset_grad;
  std::vector<poplar::Tensor> args = {operand,    scale,       mean,
                                      variance,   grad_output, operand_grad,
                                      scale_grad, offset_grad};
  auto it = res.bn_grad_graph_cache.find(key);
  if (it != res.bn_grad_graph_cache.end() &&
      !res.disable_graph_convolution_caching) {
    auto& f = it->second;
    f(args, prog);
    return std::make_tuple(args[5], args[6], args[7]);
  }
  using namespace poputil::graphfn;
  auto f = VoidFunction(
      graph,
      {input(operand, "operand"), input(scale, "scale"), input(mean, "mean"),
       input(variance, "variance"), input(grad_output, "grad_output"),
       created("operand_grad"), created("scale_grad"), created("offset_grad")},
      [&](std::vector<poplar::Tensor>& args, poplar::program::Sequence& seq) {
        poplar::Tensor inv_sd = convertVarianceToInvStdDev(
            graph, args[3], epsilon, seq, debug_prefix);

        poplar::Tensor operand_whitened =
            popnn::bn::batchNormWhiten(graph, args[0], args[2], inv_sd, seq,
                                       debug_prefix + "/WhitenedActs");

        // Compute the deltas for scaled and offset
        std::tie(args[6], args[7]) =
            popnn::bn::batchNormDeltas(graph, operand_whitened, args[4], seq,
                                       poplar::FLOAT, debug_prefix + "/Deltas");
        // Compute the delta for the operand grad
        args[5] = popnn::bn::batchNormGradients(
            graph, operand_whitened, args[4], args[6], args[7], inv_sd, args[1],
            seq, poplar::FLOAT, debug_prefix + "/Grad");
      });
  res.bn_grad_graph_cache.emplace(key, f);
  f(args, prog);
  return std::make_tuple(args[5], args[6], args[7]);
}

}  // namespace batch_norm_graph_caching
}  // namespace poplarplugin
}  // namespace xla
