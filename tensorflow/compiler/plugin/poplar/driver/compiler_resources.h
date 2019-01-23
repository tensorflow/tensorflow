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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_COMPILER_RESOURCES_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_COMPILER_RESOURCES_H_

#include "tensorflow/compiler/plugin/poplar/driver/compiler_annotations.h"
#include "tensorflow/compiler/plugin/poplar/driver/conv_graph_caching.h"
#include "tensorflow/compiler/plugin/poplar/driver/convolution_classifier.h"
#include "tensorflow/compiler/plugin/poplar/driver/norm_graph_caching.h"
#include "tensorflow/compiler/plugin/poplar/driver/visitor_subcomputation.h"

#include <poplar/OptionFlags.hpp>
#include <poplin/Convolution.hpp>
#include <poplin/MatMul.hpp>
#include <poprand/RandomGen.hpp>
#include <poputil/GraphFunction.hpp>

namespace xla {
namespace poplarplugin {

using ComputationMap = std::map<const HloComputation*, SubComputationVisitor>;

// This structure contains additional information required to lower the graph
// from an XLA graph to a poplar graph.
struct CompilerResources {
  poplar::Graph main_graph;

  std::vector<poplar::Graph> shard_graphs;

  ComputationMap computation_map;

  CompilerAnnotations annotations;

  poplin::PlanningCache convolution_cache;

  poplin::matmul::PlanningCache dot_cache;

  poprand::Random random;

  const poplar::OptionFlags default_conv_options;

  const poplar::OptionFlags default_pooling_options;

  bool disable_graph_convolution_caching;

  std::map<std::string, TensorMap> tensor_maps;

  conv_graph_caching::ConvolutionGraphCache conv_graph_cache;

  conv_graph_caching::BwdWeightGraphCache bwd_weight_graph_cache;

  conv_graph_caching::ConvolutionScaledInplaceGraphCache
      conv_scaled_inplace_graph_cache;

  conv_graph_caching::BiasApplyGraphCache bias_apply_graph_cache;

  norm_graph_caching::NormInferenceGraphCache norm_inf_graph_cache;

  norm_graph_caching::NormTrainingGraphCache norm_tr_graph_cache;

  norm_graph_caching::NormGradGraphCache norm_grad_graph_cache;

  norm_graph_caching::NormStatisticsGraphCache norm_statistics_graph_cache;

  CompilerResources(const poplar::Device& dev, uint64 seed,
                    poprand::RandomGenMode mode,
                    const poplar::OptionFlags& conv_options,
                    const poplar::OptionFlags& pooling_options,
                    bool disable_graph_convolution_caching, HloModule* module)
      : main_graph(dev),
        annotations(module),
        random(mode, seed),
        default_conv_options(conv_options),
        default_pooling_options(pooling_options),
        disable_graph_convolution_caching(disable_graph_convolution_caching) {}
};

}  // namespace poplarplugin
}  // namespace xla

#endif
