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

#ifndef TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_DATA_ENABLE_GPU_COMPATIBLE_MEMORY_H_
#define TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_DATA_ENABLE_GPU_COMPATIBLE_MEMORY_H_

#include "tensorflow/core/lib/gtl/flatset.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/grappler/optimizers/data/optimizer_base.h"

namespace tensorflow {
namespace grappler {

constexpr char kAutotune[] = "autotune";

class GPUTensorOpList {
 public:
  gtl::FlatSet<string> AllowList() {
    auto list = gtl::FlatSet<string>{
        "map_and_batch_fusion",
        "noop_elimination",
        "map_parallelization",
        "shuffle_and_repeat_fusion",
        "filter_fusion",
        "map_and_filter_fusion",
        "map_fusion",
        "parallel_batch",
        "autotune_buffer_sizes",
        "disable_prefetch_legacy_autotune",
        "make_sloppy",
        "use_choose_fastest",
        "batch_parallelization",
        "enable_gradient_descent",
        "inject_prefetch",
        "inject_prefetch_eligible",
        "autotune",
        "slack",
        "slack_period",
        "make_deterministic",
        "enable_gpu_compatible_memory",
    };
    return list;
  }
};

// If prefetch_to_device op exists, this op will try to set the set
// the memory type to the gpu compatible type (pinned memory)
class EnableGPUCompatibleMemory : public TFDataOptimizerBase {
 public:
  EnableGPUCompatibleMemory() = default;
  ~EnableGPUCompatibleMemory() override = default;

  std::string name() const override { return "enable_gpu_compatible_memory"; };

  bool UsesFunctionLibrary() const override { return false; }

  Status Init(
      const tensorflow::RewriterConfig_CustomGraphOptimizer* config) override {
    if (!config) return Status::OK();

    const std::string& autotune = config->parameter_map().at(kAutotune).s();
    if (autotune == "true") {
      autotune_ = true;
    } else if (autotune == "false") {
      autotune_ = false;
    } else {
      return errors::InvalidArgument("Received an invalid value for parameter ",
                                     kAutotune, ": ", autotune);
    }
    return Status::OK();
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
