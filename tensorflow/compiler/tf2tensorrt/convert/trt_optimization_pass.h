/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_TF2TENSORRT_CONVERT_TRT_OPTIMIZATION_PASS_H_
#define TENSORFLOW_COMPILER_TF2TENSORRT_CONVERT_TRT_OPTIMIZATION_PASS_H_

#include <memory>
#include <string>

#include "tensorflow/compiler/tf2tensorrt/convert/trt_parameters.h"
#include "tensorflow/compiler/tf2tensorrt/convert/utils.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/grappler/costs/graph_properties.h"
#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/platform/logging.h"

#if GOOGLE_CUDA && GOOGLE_TENSORRT

#if !IS_TRT_VERSION_GE(7, 0, 0, 0)
#error From version 2.6, we only support NVIDIA TensorRT version 7 or newer.
#error Please update your environment and relaunch the compilation.
#endif

namespace tensorflow {
namespace tensorrt {
namespace convert {

class TRTOptimizationPass : public grappler::CustomGraphOptimizer {
 public:
  struct ConversionParams {
    string trt_logger_name = "DefaultLogger";
    size_t max_batch_size = -1;
    size_t max_workspace_size_bytes = 1 << 30;
    TrtPrecisionMode precision_mode = TrtPrecisionMode::FP32;
    int minimum_segment_size = 3;
    // Whether to create engine on conversion or execution time
    bool is_dynamic_op = false;
    // maximum number of cached engines
    int max_cached_engines = 1;
    bool use_calibration = true;
    bool use_implicit_batch = true;
    ProfileStrategy profile_strategy = ProfileStrategy::kRange;
    bool allow_build_at_runtime = true;
    bool use_explicit_precision = false;
    void SerializeToString(std::string* myString) const {
        std::stringstream ss;
        ss << trt_logger_name << max_batch_size << max_workspace_size_bytes
           << (int)precision_mode << minimum_segment_size << is_dynamic_op
           << max_cached_engines << use_calibration << use_implicit_batch
           << (int)profile_strategy << allow_build_at_runtime
           << use_explicit_precision;

        *myString = ss.str();
    }
  };

  TRTOptimizationPass(const string& name = "TRTOptimizationPass")
      : name_(name) {}

  string name() const override { return name_; };

  bool UsesFunctionLibrary() const override { return true; }

  Status Init(
      const RewriterConfig_CustomGraphOptimizer* config = nullptr) override;

  Status Optimize(grappler::Cluster* cluster,
                  const grappler::GrapplerItem& item,
                  GraphDef* optimized_graph) override;

 private:
  const string name_;

  ConversionParams params_;

  std::vector<int> batches_;
};

}  // namespace convert
}  // namespace tensorrt
}  // namespace tensorflow

#endif  // GOOGLE_CUDA && GOOGLE_TENSORRT
#endif  // TENSORFLOW_COMPILER_TF2TENSORRT_CONVERT_TRT_OPTIMIZATION_PASS_H_
