/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_DATA_INJECT_PREFETCH_H_
#define TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_DATA_INJECT_PREFETCH_H_

#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/grappler/optimizers/data/optimizer_base.h"

namespace tensorflow {
namespace grappler {

constexpr char kAutotune[] = "autotune";

// If autotune is ON and the last transformation in the input pipeline is not
// `prefetch()`, this optimization adds `prefetch(AUTOTUNE)` after it.
class InjectPrefetch : public TFDataOptimizerBase {
 public:
  InjectPrefetch() = default;
  ~InjectPrefetch() override = default;

  std::string name() const override { return "inject_prefetch"; };

  bool UsesFunctionLibrary() const override { return false; }

  Status Init(
      const tensorflow::RewriterConfig_CustomGraphOptimizer* config) override {
    if (!config) return OkStatus();

    const std::string& autotune = config->parameter_map().at(kAutotune).s();
    if (autotune == "true") {
      autotune_ = true;
    } else if (autotune == "false") {
      autotune_ = false;
    } else {
      return errors::InvalidArgument("Received an invalid value for parameter ",
                                     kAutotune, ": ", autotune);
    }
    return OkStatus();
  }

  Status OptimizeAndCollectStats(Cluster* cluster, const GrapplerItem& item,
                                 GraphDef* output,
                                 OptimizationStats* stats) override;

 protected:
  bool autotune_ = true;
};

}  // namespace grappler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_DATA_INJECT_PREFETCH_H_
