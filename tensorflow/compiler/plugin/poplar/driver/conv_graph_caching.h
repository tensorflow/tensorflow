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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_CONV_GRAPH_CACHING_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_CONV_GRAPH_CACHING_H_

#include "tensorflow/compiler/plugin/poplar/driver/convolution_classifier.h"
#include "tensorflow/compiler/plugin/poplar/driver/graph_caching_util.h"

#include <poplar/Tensor.hpp>
#include <poplin/ConvUtil.hpp>
#include <poputil/GraphFunction.hpp>

namespace xla {
namespace poplarplugin {
struct CompilerResources;

namespace conv_graph_caching {

// Fwd and bwd convolution caches

// The convolution key is:
// * Shape of the input tensor
// * Shape of the weights tensor
// * ConvolutionDimensionNumbers for the given convolution
// * poplin ConvParams for the given convolution
// * Enum for the type of convolution
// * bool indicating whether to do weightsTransposeChansFlipXY
using ConvolutionCacheKey =
    std::tuple<PoplarTensorSignature, PoplarTensorSignature, poplin::ConvParams,
               ConvClassificationType, bool>;
using ConvolutionGraphCache =
    std::map<ConvolutionCacheKey, poputil::graphfn::TensorFunction>;

using BwdWeightCacheKey =
    std::pair<PoplarTensorSignature, PoplarTensorSignature>;
using BwdWeightGraphCache =
    std::map<BwdWeightCacheKey, poputil::graphfn::VoidFunction>;

poplar::Tensor DoCachedConvolution(
    poplar::Graph& graph, CompilerResources& res, const poplar::Tensor& in,
    const poplar::Tensor& weights, const poplin::ConvParams& params,
    const ConvClassificationType& conv_type, bool transpose_and_flip_weights,
    poplar::program::Sequence& prog, const std::string& debug_prefix);

// The weight update convolution key is:
// * Shape of the input tensor
// * Shape of the gradient tensor
// * ConvolutionDimensionNumbers for the given convolution
// * poplin ConvParams for the given convolution
// * Enum for the type of convolution
// * Learning rate constant
using WeightUpdateConvolutionCacheKey =
    std::tuple<PoplarTensorSignature, PoplarTensorSignature, poplin::ConvParams,
               ConvClassificationType, double>;
using WeightUpdateConvolutionGraphCache =
    std::map<WeightUpdateConvolutionCacheKey, poputil::graphfn::VoidFunction>;

Status DoCachedConvolutionWithScaledAdd(
    poplar::Graph& graph, CompilerResources& res, const poplar::Tensor& weights,
    const poplar::Tensor& in, const poplar::Tensor& deltas,
    const poplin::ConvParams& params, poplar::program::Sequence& prog,
    const HloInstruction* root, const HloInstruction* conv);
}  // namespace conv_graph_caching

}  // namespace poplarplugin
}  // namespace xla

#endif
