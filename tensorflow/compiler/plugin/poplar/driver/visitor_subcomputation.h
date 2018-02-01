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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_VISITOR_SUBCOMPUTATION_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_VISITOR_SUBCOMPUTATION_H_

#include "tensorflow/compiler/plugin/poplar/driver/visitor_full.h"

namespace poplar {
class Graph;
class Tensor;
}

namespace xla {
namespace poplarplugin {

class SubComputationVisitor : public FullVisitor {
public:
  SubComputationVisitor(poplar::Graph* graph,
                        CompilerResources& res,
                        const ArgVectors& inputs);

  Status HandleParameter(HloInstruction* inst) override;
  Status FinishVisit(HloInstruction* inst) override;

  const ArgVectors& inputs() {
    return inputs_;
  }

  const OutVector& outputs() {
    return outputs_;
  }

  bool input_valid(unsigned int param, unsigned int index) {
    return (param < input_valid_.size() &&
            index < input_valid_[param].size() &&
            input_valid_[param][index]);
  }

private:
  ArgVectors temp_inputs_;
  ArgVectors inputs_;
  OutVector outputs_;

  std::vector<std::vector<bool>> input_valid_;
};

}  // namespace poplarplugin
}  // namespace xla

#endif
