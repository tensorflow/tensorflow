/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
#ifndef TENSORFLOW_COMPILER_TF2TENSORRT_CONVERT_PASSES_PREPERATION_PASS_H_
#define TENSORFLOW_COMPILER_TF2TENSORRT_CONVERT_PASSES_PREPERATION_PASS_H_
#if GOOGLE_CUDA && GOOGLE_TENSORRT

#include <memory>
#include <string>

#include "tensorflow/compiler/tf2tensorrt/convert/trt_parameters.h"
#include "tensorflow/compiler/tf2tensorrt/convert/utils.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/grappler/costs/graph_properties.h"
#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer.h"
#include "tensorflow/core/grappler/optimizers/graph_optimizer_stage.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
namespace tensorrt {
namespace convert {

class TRTPreperationContext;

class TRTPreperationPass : public grappler::CustomGraphOptimizer {
 public:
  TRTPreperationPass(const string& name = "TRTPreperationPass")
      : name_(name), precision_mode_(TrtPrecisionMode::FP32) {}

  string name() const override { return name_; };
  bool UsesFunctionLibrary() const override { return true; }
  Status Init(
      const RewriterConfig_CustomGraphOptimizer* config = nullptr) override;

  Status ConditionGraphConversion(bool can_use_shapes);

  Status RunPipeline(grappler::GraphOptimizerStagePipeline<string>& pipeline,
                     const grappler::GraphOptimizerContext& context,
                     const TRTPreperationContext& trt_context);
  Status Optimize(grappler::Cluster* cluster,
                  const grappler::GrapplerItem& item,
                  GraphDef* optimized_graph) override;

  // Specifies which pipeline stages should be enabled during graph
  // conditioning prior to TRT segmentation and conversion.
  struct ConditioningOptions {
    // Attempt to enable "cast-free" FP16 network inputs by rewriting casts to
    // FP32 to two different casts.
    bool rewrite_fp32_casts{true};
  };

 private:
  const string name_;
  TrtPrecisionMode precision_mode_;
  ConditioningOptions conditioning_options_{};

  bool fetch_nodes_known_ = false;
  std::unordered_set<string> nodes_to_preserve_;
  std::unique_ptr<grappler::NodeMap> node_map_;
  std::unique_ptr<grappler::GraphProperties> graph_properties_;
  GraphDef* optimized_graph_ = nullptr;  // Not owned.
  gtl::FlatSet<string> feed_nodes_;
  RewriterConfig::Toggle opt_level_;
};
}  // namespace convert
}  // namespace tensorrt
}  // namespace tensorflow
#endif  // GOOGLE_CUDA && GOOGLE_TENSORRT
#endif  // TENSORFLOW_COMPILER_TF2TENSORRT_CONVERT_PASSES_PREPERATION_PASS_H_
