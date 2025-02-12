/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_GENERIC_LAYOUT_OPTIMIZER_H_
#define TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_GENERIC_LAYOUT_OPTIMIZER_H_

#include <string>

#include "tensorflow/core/grappler/optimizers/graph_optimizer.h"
#include "tensorflow/core/protobuf/rewriter_config.pb.h"

namespace tensorflow {
namespace grappler {

// Optimize the data layout for convolutional models.
class GenericLayoutOptimizer : public GraphOptimizer {
 public:
  explicit GenericLayoutOptimizer(string enforced_layout = "")
      : GenericLayoutOptimizer(RewriterConfig::DEFAULT,
                               RewriterConfig::NO_CONVERSION_ON_CPU,
                               enforced_layout) {}
  explicit GenericLayoutOptimizer(RewriterConfig::Toggle opt_level,
                                  string enforced_layout = "")
      : GenericLayoutOptimizer(opt_level, RewriterConfig::NO_CONVERSION_ON_CPU,
                               enforced_layout) {}
  explicit GenericLayoutOptimizer(RewriterConfig::Toggle opt_level,
                                  RewriterConfig::CpuLayout layout_conversion,
                                  string enforced_layout = "")
      : opt_level_(opt_level),
        cpu_layout_conversion_(layout_conversion),
        enforced_layout_(enforced_layout) {}
  ~GenericLayoutOptimizer() override = default;

  string name() const override { return "layout"; };

  bool UsesFunctionLibrary() const override { return false; }

  absl::Status Optimize(Cluster* cluster, const GrapplerItem& item,
                        GraphDef* output) override;

 private:
  RewriterConfig::Toggle opt_level_;
  RewriterConfig::CpuLayout cpu_layout_conversion_;
  const string enforced_layout_;
};

}  // namespace grappler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_GENERIC_LAYOUT_OPTIMIZER_H_
