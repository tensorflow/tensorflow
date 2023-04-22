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

#ifndef TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_AUTO_MIXED_PRECISION_H_
#define TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_AUTO_MIXED_PRECISION_H_

#include "tensorflow/core/grappler/optimizers/graph_optimizer.h"
#include "tensorflow/core/protobuf/rewriter_config.pb.h"

namespace tensorflow {
namespace grappler {

enum class AutoMixedPrecisionMode { CUDA, MKL };

// Convert data types to float16 or bfloat16 where appropriate to improve
// performance on GPUs or CPUs.
class AutoMixedPrecision : public GraphOptimizer {
 public:
  // If 'mode' is CUDA, converts nodes to float16 on Nvidia GPUs. If MKL,
  // converts nodes to bfloat16 on CPUs in order to take advantage of MKL
  // performance improvements with bfloat16.
  explicit AutoMixedPrecision(
      AutoMixedPrecisionMode mode = AutoMixedPrecisionMode::CUDA)
      : mode_(mode) {}

  ~AutoMixedPrecision() override {}

  string name() const override {
    return mode_ == AutoMixedPrecisionMode::CUDA ? "auto_mixed_precision_cuda"
                                                 : "auto_mixed_precision_mkl";
  };

  bool UsesFunctionLibrary() const override { return false; }

  Status Optimize(Cluster* cluster, const GrapplerItem& item,
                  GraphDef* output) override;

 private:
  const AutoMixedPrecisionMode mode_;
};

}  // end namespace grappler
}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_AUTO_MIXED_PRECISION_H_
