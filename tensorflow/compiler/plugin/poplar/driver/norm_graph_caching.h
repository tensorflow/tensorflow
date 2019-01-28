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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_NORM_GRAPH_CACHING_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_NORM_GRAPH_CACHING_H_

#include "tensorflow/compiler/plugin/poplar/driver/graph_caching_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops.h"
#include "tensorflow/core/platform/default/integral_types.h"

#include <poplar/Tensor.hpp>
#include <poputil/GraphFunction.hpp>

#include "absl/types/optional.h"

#include <map>

using tensorflow::uint32;
using tensorflow::uint64;

namespace xla {
namespace poplarplugin {
struct CompilerResources;

namespace norm_graph_caching {

// The norm inference key is:
// * type of norm
// * shape of operand
// * shape of scale
// * shape of offset
// * shape of mean
// * shape of variance_or_inv_std_dev
// * epsilon
// * num_groups for GroupNorm, 0 otherwise
// * sharding device ID
using NormInferenceCacheKey =
    std::tuple<NormType, PoplarTensorSignature, PoplarTensorSignature,
               PoplarTensorSignature, PoplarTensorSignature,
               PoplarTensorSignature, double, uint32, uint64>;
using NormInferenceGraphCache =
    std::map<NormInferenceCacheKey, poputil::graphfn::TensorFunction>;
poplar::Tensor DoCachedNormInference(
    const NormType& norm_type, poplar::Graph& graph, CompilerResources& res,
    const poplar::Tensor& operand, const poplar::Tensor& scale,
    const poplar::Tensor& offset, const poplar::Tensor& mean,
    const poplar::Tensor& variance_or_inv_std_dev, const double epsilon,
    absl::optional<uint32> optional_num_groups, const uint64 device,
    poplar::program::Sequence& prog, const std::string& debug_prefix);

// The norm training key is:
// * type of norm
// * shape of operand
// * shape of scale
// * shape of offset
// * epsilon
// * num_groups for GroupNorm, 0 otherwise
// * sharding device ID
using NormTrainingCacheKey =
    std::tuple<NormType, PoplarTensorSignature, PoplarTensorSignature,
               PoplarTensorSignature, double, uint32, uint64>;
using NormTrainingGraphCache =
    std::map<NormTrainingCacheKey, poputil::graphfn::VoidFunction>;
std::tuple<poplar::Tensor, poplar::Tensor, poplar::Tensor> DoCachedNormTraining(
    const NormType& norm_type, poplar::Graph& graph, CompilerResources& res,
    const poplar::Tensor& operand, poplar::Tensor& whitened_operand,
    const poplar::Tensor& scale, const poplar::Tensor& offset,
    const double epsilon, absl::optional<uint32> optional_num_groups,
    const uint64 device, poplar::program::Sequence& prog,
    const std::string& debug_prefix);

// The norm gradient key is:
// * type of norm
// * shape of operand
// * shape of scale
// * shape of mean
// * shape of variance_or_inv_std_dev
// * shape of grad_output
// * epsilon
// * num_groups for GroupNorm, 0 otherwise
// * sharding device ID
using NormGradCacheKey =
    std::tuple<NormType, PoplarTensorSignature, PoplarTensorSignature,
               PoplarTensorSignature, PoplarTensorSignature,
               PoplarTensorSignature, double, uint32, uint64>;
using NormGradGraphCache =
    std::map<NormGradCacheKey, poputil::graphfn::VoidFunction>;
std::tuple<poplar::Tensor, poplar::Tensor, poplar::Tensor> DoCachedNormGrad(
    const NormType& norm_type, poplar::Graph& graph, CompilerResources& res,
    const poplar::Tensor& operand, const poplar::Tensor& scale,
    const poplar::Tensor& mean, const poplar::Tensor& variance_or_inv_std_dev,
    const poplar::Tensor& grad_output, const double epsilon,
    absl::optional<uint32> optional_num_groups, const uint64 device,
    poplar::program::Sequence& prog, const std::string& debug_prefix);

// The norm statistics key is:
// * type of norm
// * shape of operand
// * epsilon
// * num_groups for GroupNorm, 0 otherwise
// * sharding device ID
using NormStatisticsCacheKey =
    std::tuple<NormType, PoplarTensorSignature, double, uint32, uint64>;
using NormStatisticsGraphCache =
    std::map<NormStatisticsCacheKey, poputil::graphfn::VoidFunction>;
std::tuple<poplar::Tensor, poplar::Tensor> DoCachedNormStatistics(
    const NormType& norm_type, poplar::Graph& graph, CompilerResources& res,
    const poplar::Tensor& operand, const double epsilon,
    absl::optional<uint32> optional_num_groups, const uint64 device,
    poplar::program::Sequence& prog, const std::string& debug_prefix);

}  // namespace norm_graph_caching

}  // namespace poplarplugin
}  // namespace xla

#endif
