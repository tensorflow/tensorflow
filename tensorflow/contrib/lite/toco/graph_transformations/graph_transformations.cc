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

// Some graphs have RNN back-edges that are discardable, having been
// created typically by TensorFlow import rather than specified by the user.
// Such graphs might have cycles (closed by RNN back-edges) that may be pruned.
// Local graph transformations can't identify such global features,
// so this function performs this global transformation.
//
// The other (and related) thing that is peculiar about RNN back-edges
// is that they do not prevent the arrays that they touch, from being
// pruned. Thus, they may refer to array names which no longer exist.
// The intent is for that to result in the eventual pruning of such
// 'dangling' RNN back-edges. We perform this pruning at the end of this
// function, as the pruning of connected components done here may leave
// more RNN back-edges dangling.
void DiscardUselessConnectedComponentsAndRNNBackEdges(Model* model) {
  // Identify the set of arrays that are in 'useful' connected components
  // of the graph, which means connected to output arrays.
  std::unordered_set<string> useful_arrays;
  for (const string& output_array : model->flags.output_arrays()) {
    useful_arrays.insert(output_array);
  }
  bool found_new_useful_arrays;
  do {
    found_new_useful_arrays = false;
    for (const auto& op : model->operators) {
      bool op_touches_useful_arrays = false;
      for (const string& output : op->outputs) {
        op_touches_useful_arrays |= useful_arrays.count(output);
      }
      if (op_touches_useful_arrays) {
        for (const string& input : op->inputs) {
          found_new_useful_arrays |= !useful_arrays.count(input);
          useful_arrays.insert(input);
        }
        for (const string& output : op->outputs) {
          found_new_useful_arrays |= !useful_arrays.count(output);
          useful_arrays.insert(output);
        }
      }
    }
    for (const auto& rnn_state : model->flags.rnn_states()) {
      bool rnn_back_edge_touches_useful_arrays =
          useful_arrays.count(rnn_state.state_array());
      if (rnn_back_edge_touches_useful_arrays) {
        found_new_useful_arrays |=
            !useful_arrays.count(rnn_state.back_edge_source_array());
        useful_arrays.insert(rnn_state.back_edge_source_array());
      }
    }
  } while (found_new_useful_arrays);
  // Erase arrays that aren't useful, and that are discardable.
  for (auto it = model->arrays.begin(); it != model->arrays.end();) {
    if (useful_arrays.count(it->first) ||
        !IsDiscardableArray(*model, it->first)) {
      ++it;
    } else {
      it = model->arrays.erase(it);
    }
  }
  // Erase operators that do not produce a useful output array.
  for (auto it = model->operators.begin(); it != model->operators.end();) {
    // Only need to test the first output, as we simultaneously added all of
    // an operator's outputs to the list of output arrays.
    if (useful_arrays.count((*it)->outputs[0])) {
      ++it;
    } else {
      for (const string& output : (*it)->outputs) {
        CHECK(!useful_arrays.count(output));
      }
      it = model->operators.erase(it);
    }
  }
  // Erase RNN back-edges that are 'dangling' i.e. that touch an array
  // that no longer exists. This should only happen for discardable RNN
  // back-edges.
  std::vector<RnnState> rnn_states_to_keep;
  for (const auto& rnn_state : model->flags.rnn_states()) {
    const bool dangling =
        !model->arrays.count(rnn_state.back_edge_source_array()) ||
        !model->arrays.count(rnn_state.state_array());
    if (dangling) {
      CHECK(rnn_state.discardable());
    } else {
      rnn_states_to_keep.push_back(rnn_state);
    }
  }
  model->flags.clear_rnn_states();
  for (const auto& rnn_state : rnn_states_to_keep) {
    *model->flags.add_rnn_states() = rnn_state;
  }
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
  DiscardUselessConnectedComponentsAndRNNBackEdges(model);
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
