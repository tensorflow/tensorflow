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
#include "tensorflow/contrib/lite/toco/graph_transformations/graph_transformations.h"

#include <algorithm>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "tensorflow/contrib/lite/toco/toco_port.h"
#include "tensorflow/contrib/lite/toco/tooling_util.h"
#include "tensorflow/core/platform/logging.h"

namespace toco {

namespace {

void PrintModelStats(const string& label, const Model& model) {
  int quantized_arrays = 0;
  for (const auto& array : model.arrays) {
    if (array.second->quantization_params) {
      quantized_arrays++;
    }
  }
  LOG(INFO) << label << ": " << model.operators.size() << " operators, "
            << model.arrays.size() << " arrays (" << quantized_arrays
            << " quantized)";
}

bool GraphTransformationsPass(int increment, Model* model,
                              const GraphTransformationsSet& transformations) {
  CHECK(increment == 1 || increment == -1);
  bool changed = false;
  CHECK(!model->operators.empty());
  int op_index = increment == 1 ? 0 : model->operators.size() - 1;
  while (true) {
    bool changed_now = false;
    // Loop over all transformations at the current position in the graph.
    for (const auto& transformation : transformations) {
      CHECK(!changed_now);
      CHECK(transformation->Messages().empty());
      changed_now = transformation->Run(model, op_index);
      if (changed_now) {
        DumpGraphvizVideoFrame(*model);
        CHECK(!model->operators.empty());
        op_index = std::min<int>(op_index, model->operators.size() - 1);
        // Uncomment for debugging
        // CheckInvariants(*model);
      }
      const char* made_a_change_msg =
          changed_now ? "made a change" : "did NOT make a change";
      const int log_level =
          changed_now ? kLogLevelModelChanged : kLogLevelModelUnchanged;
      for (const string& message : transformation->Messages()) {
        VLOG(log_level) << transformation->Name() << " " << made_a_change_msg
                        << " at op_index=" << op_index << "/"
                        << model->operators.size() - 1 << ": " << message;
      }
      transformation->ClearMessages();
      if (changed_now) {
        break;
      }
    }
    if (changed_now) {
      changed = true;
    } else {
      const int op_index_last =
          increment == 1 ? model->operators.size() - 1 : 0;
      if (op_index == op_index_last) {
        break;
      }
      op_index += increment;
    }
  }
  return changed;
}

}  // namespace

void RunGraphTransformations(Model* model, const string& msg,
                             const GraphTransformationsSet& transformations) {
  PrintModelStats(toco::port::StringF("Before %s", msg), *model);
  int pass_index = 0;
  while (GraphTransformationsPass((pass_index % 2) ? -1 : 1, model,
                                  transformations)) {
    pass_index++;
    const auto& label =
        toco::port::StringF("After %s pass %d", msg, pass_index);
    PrintModelStats(label, *model);
    CheckInvariants(*model);
  }
}

}  // namespace toco
