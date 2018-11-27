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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_BATCH_NORM_GRAPH_CACHING_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_BATCH_NORM_GRAPH_CACHING_H_

#include "tensorflow/compiler/plugin/poplar/driver/graph_caching_util.h"

#include <poplar/Tensor.hpp>
#include <poputil/GraphFunction.hpp>

#include <map>

namespace xla {
namespace poplarplugin {
struct CompilerResources;

namespace batch_norm_graph_caching {

// The batch norm inference key is:
// * shape of operand
// * shape of scale
// * shape of offset
// * shape of mean
// * shape of variance
// * epsilon
using BatchNormInferenceCacheKey =
    std::tuple<PoplarTensorSignature, PoplarTensorSignature,
               PoplarTensorSignature, PoplarTensorSignature,
               PoplarTensorSignature, double>;
using BatchNormInferenceGraphCache =
    std::map<BatchNormInferenceCacheKey, poputil::graphfn::TensorFunction>;
poplar::Tensor DoCachedBatchNormInference(
    poplar::Graph& graph, CompilerResources& res, const poplar::Tensor& operand,
    const poplar::Tensor& scale, const poplar::Tensor& offset,
    const poplar::Tensor& mean, const poplar::Tensor& variance,
    const double epsilon, poplar::program::Sequence& prog,
    const std::string& debug_prefix);

// The batch norm training key is:
// * shape of operand
// * shape of scale
// * shape of offset
// * epsilon
using BatchNormTrainingCacheKey =
    std::tuple<PoplarTensorSignature, PoplarTensorSignature,
               PoplarTensorSignature, double>;
using BatchNormTrainingGraphCache =
    std::map<BatchNormTrainingCacheKey, poputil::graphfn::VoidFunction>;
std::tuple<poplar::Tensor, poplar::Tensor, poplar::Tensor>
DoCachedBatchNormTraining(poplar::Graph& graph, CompilerResources& res,
                          const poplar::Tensor& operand,
                          const poplar::Tensor& scale,
                          const poplar::Tensor& offset, const double epsilon,
                          poplar::program::Sequence& prog,
                          const std::string& debug_prefix);
// Cached BatchNormGrad

// The batch norm gradient key is:
// * shape of operand
// * shape of scale
// * shape of mean
// * shape of variance
// * shape of grad_output
// * epsilon
using BatchNormGradCacheKey =
    std::tuple<PoplarTensorSignature, PoplarTensorSignature,
               PoplarTensorSignature, PoplarTensorSignature,
               PoplarTensorSignature, double>;
using BatchNormGradGraphCache =
    std::map<BatchNormGradCacheKey, poputil::graphfn::VoidFunction>;
std::tuple<poplar::Tensor, poplar::Tensor, poplar::Tensor>
DoCachedBatchNormGrad(poplar::Graph& graph, CompilerResources& res,
                      const poplar::Tensor& operand,
                      const poplar::Tensor& scale, const poplar::Tensor& mean,
                      const poplar::Tensor& variance,
                      const poplar::Tensor& grad_output, const double epsilon,
                      poplar::program::Sequence& prog,
                      const std::string& debug_prefix);

}  // namespace batch_norm_graph_caching

}  // namespace poplarplugin
}  // namespace xla

#endif
