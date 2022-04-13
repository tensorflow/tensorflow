/* Copyright 20121 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_TF2TENSORRT_CONVERT_TRT_LAYOUT_OPTIMIZATION_PASS_H_
#define TENSORFLOW_COMPILER_TF2TENSORRT_CONVERT_TRT_LAYOUT_OPTIMIZATION_PASS_H_

#include <string>

#include "tensorflow/compiler/tf2tensorrt/convert/utils.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer.h"
#include "tensorflow/core/platform/logging.h"

#if GOOGLE_CUDA && GOOGLE_TENSORRT

#if !IS_TRT_VERSION_GE(7, 0, 0, 0)
#error From version 2.6, we only support NVIDIA TensorRT version 7 or newer.
#error Please update your environment and relaunch the compilation.
#endif

namespace tensorflow {
namespace tensorrt {
namespace convert {
class TRTLayoutOptimizationPass : public grappler::CustomGraphOptimizer {
 public:
  TRTLayoutOptimizationPass(const string& name = "TRTLayoutOptimizationPass");

  string name() const override { return name_; };

  bool UsesFunctionLibrary() const override { return true; }

  Status Init(
      const RewriterConfig_CustomGraphOptimizer* config = nullptr) override;

  Status Optimize(grappler::Cluster* cluster,
                  const grappler::GrapplerItem& item,
                  GraphDef* optimized_graph) override;

  /*  void PrintDebugInfo(grappler::Cluster* cluster,
                        const grappler::GrapplerItem& item);
  */

 private:
  const string name_;
  string trt_logger_name_;
  int minimum_segment_size_;
  bool is_dynamic_op_;
  int max_cached_batches_;
  int64_t max_workspace_size_bytes_;
};

}  // namespace convert
}  // namespace tensorrt
}  // namespace tensorflow

#endif  // GOOGLE_CUDA && GOOGLE_TENSORRT
#endif  // TENSORFLOW_COMPILER_TF2TENSORRT_CONVERT_TRT_LAYOUT_OPTIMIZATION_PASS_H_
