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

#include <unordered_set>

#include "tensorflow/c/eager/tape.h"

namespace tensorflow {
namespace eager {

bool GradientTape::ShouldRecord(gtl::ArraySlice<int64> tensor_ids) {
  for (int64 i : tensor_ids) {
    if (tensor_tape_.find(i) != tensor_tape_.end()) {
      return true;
    }
  }
  return false;
}

void GradientTape::Watch(int64 tensor_id) {
  tensor_tape_.emplace(tensor_id, -1);
}

void GradientTape::RecordOperation(
    const string& op_type, gtl::ArraySlice<TapeTensor> output_tensors,
    gtl::ArraySlice<int64> input_tensor_id, void* backward_function,
    const std::function<void()>& backward_function_deleter) {
  if (!ShouldRecord(input_tensor_id)) {
    backward_function_deleter();
    return;
  }
  std::vector<int64> ids;
  ids.reserve(input_tensor_id.size());
  for (int64 i : input_tensor_id) {
    tensor_usage_[i]++;
    ids.push_back(i);
  }
  const int64 op_id = next_op_id_++;
  std::vector<TapeTensor> tensors;
  tensors.reserve(output_tensors.size());
  for (const TapeTensor& o : output_tensors) {
    // Note: the tensor can have already been watched and hence be in the tape,
    // so we cannot check that we're inserting it here.
    tensor_tape_[o.id] = op_id;
    tensor_usage_[o.id] = 1;
    tensors.push_back(o);
  }
  op_tape_[op_id] = OpTapeEntry{op_type, tensors, ids, backward_function,
                                backward_function_deleter};
}

void GradientTape::DeleteTrace(int64 tensor_id) {
  auto it = tensor_usage_.find(tensor_id);
  if (it == tensor_usage_.end()) {
    return;
  }
  it->second--;
  if (it->second != 0) {
    return;
  }
  tensor_usage_.erase(it);
  auto tensor_op_it = tensor_tape_.find(tensor_id);
  if (tensor_op_it == tensor_tape_.end()) {
    return;
  }
  const int64 op_id = tensor_op_it->second;
  if (op_id == -1) {
    // Do not delete watched tensors.
    return;
  }
  tensor_tape_.erase(tensor_op_it);
  auto op_it = op_tape_.find(op_id);
  CHECK(op_it != op_tape_.end());
  for (const auto& output : op_it->second.output_tensor_info) {
    if (tensor_usage_.find(output.id) != tensor_usage_.end()) {
      // Found a usage for an output, so cannot delete the op.
      return;
    }
  }
  for (int64 id : op_it->second.input_tensor_id) {
    DeleteTrace(id);
  }
  op_it->second.backward_function_deleter();
  op_tape_.erase(op_it);
}

// Terminology:
//
//  - op: a possibly composite operation, which has an entry in the tape
//  - target: dy in dx/dy
//  - source: dx in dx/dy
//  - tensor: one of the many inputs or outputs of an operation
//
// Below here we do the gradient algorithm. It works as follows:
//
// First we filter the tape to just the subset of operations we want to
// differentiate. In the process of doing so we count how many times each Tensor
// is used as an input to an op (so we know when we're done computing gradients
// for that Tensor). We also count, for each tape entry, how many of its output
// Tensors need gradients to be computed (Tensors which are not used do not need
// any gradients to be computed).
//
// Finally, we start a backprop stack with a set of tape entries for which we
// have all gradients available. This set usually is a subset of the set of
// targets (not all since targets which have outputs in the tape will not have
// gradients available initially).
//
// Then we repeatedly pop an entry from the stack, run its backprop, and update
// the gradients of its inputs. Once we have computed all gradients for a single
// input we can mark this input as done, and this can trigger adding an entry to
// the stack if all outputs of that entry are now done.
//
// When the stack is empty we have gradients for all tensors we're interested
// in.

struct BackpropInitialState {
  OpTape op_tape;

  // Map from tensor ID to how many references still exist for this tensor in
  // the tape.
  std::unordered_map<int64, int64> tensor_usage_counts;

  // Maps from op ID to how many output tensors of this op still need to have
  // their gradients computed.
  std::unordered_map<int64, int64> op_missing_tensor;
};

BackpropInitialState PrepareBackprop(
    gtl::ArraySlice<int64> target, const TensorTape& tensor_tape,
    OpTape op_tape, const std::unordered_set<int64>& sources_set) {
  std::vector<int64> tensor_stack;
  tensor_stack.reserve(target.size());
  for (auto t : target) {
    tensor_stack.push_back(t);
  }
  BackpropInitialState result;
  while (!tensor_stack.empty()) {
    int64 tensor_id = tensor_stack.back();
    tensor_stack.pop_back();
    auto op_id_it = tensor_tape.find(tensor_id);
    if (op_id_it == tensor_tape.end()) {
      continue;
    }
    int64 op_id = op_id_it->second;
    auto op_it = op_tape.find(op_id);
    auto result_op_it = result.op_tape.find(op_id);
    if (op_id == -1 || op_it == op_tape.end() ||
        result_op_it != result.op_tape.end()) {
      continue;
    }
    CHECK(result.op_tape.emplace(op_id, op_it->second).second);
    for (auto it : op_it->second.input_tensor_id) {
      auto count_it = result.tensor_usage_counts.find(it);
      if (count_it != result.tensor_usage_counts.end()) {
        count_it->second++;
      } else {
        result.tensor_usage_counts[it] = 1;
        if (sources_set.find(it) == sources_set.end() &&
            tensor_tape.find(it) != tensor_tape.end()) {
          tensor_stack.push_back(it);
        }
      }
    }
    op_tape.erase(op_it);
  }
  for (auto& pair : result.tensor_usage_counts) {
    auto it = tensor_tape.find(pair.first);
    if (it != tensor_tape.end() && it->second != -1) {
      result.op_missing_tensor[it->second] += 1;
    }
  }
  // Call destructors for all unneeded gradient functions.
  for (const auto& op_pair : op_tape) {
    op_pair.second.backward_function_deleter();
  }
  return result;
}

std::vector<int64> InitialStack(
    const OpTape& op_tape,
    const std::unordered_map<int64, int64>& op_missing_tensor) {
  std::vector<int64> result;
  for (auto& op_entry : op_tape) {
    if (op_missing_tensor.find(op_entry.first) == op_missing_tensor.end()) {
      result.push_back(op_entry.first);
    }
  }
  return result;
}

Status InitialGradients(const VSpace& vspace, gtl::ArraySlice<void*> target,
                        gtl::ArraySlice<void*> output_gradients,
                        std::unordered_map<int64, int64> tensor_usage_counts,
                        std::unordered_map<int64, std::vector<void*>>* result) {
  for (int i = 0; i < target.size(); ++i) {
    int64 id = vspace.TensorId(target[i]);
    if (tensor_usage_counts.find(id) != tensor_usage_counts.end()) {
      if (!output_gradients.empty() && output_gradients[i] != nullptr) {
        // TODO(apassos) figure out how to print debugging information here.
        return errors::InvalidArgument(
            "A gradient was provided for a tensor which is used as part of the "
            "computation.");
      }
    } else {
      if (output_gradients.empty() || output_gradients[i] == nullptr) {
        (*result)[id].push_back(vspace.OnesLike(target[i]));
      } else {
        (*result)[id].push_back(output_gradients[i]);
      }
    }
  }
  return Status::OK();
}

// If over kMinAggregateCount gradients are accumulated and the total
// memory consumption is over kMinAggregateBytes, do an early aggregation
// so as to release the gradient tensor to save memory.
static const int kMinAggregateCount = 4;
static const int kMinAggregateBytes = 128 * 1024 * 1024;

Status GradientTape::Gradient(const VSpace& vspace,
                              gtl::ArraySlice<void*> target,
                              gtl::ArraySlice<void*> sources,
                              gtl::ArraySlice<void*> output_gradients,
                              std::vector<void*>* result) {
  std::vector<int64> id_sources;
  id_sources.reserve(sources.size());
  for (void* s : sources) {
    id_sources.push_back(vspace.TensorId(s));
  }
  std::unordered_set<int64> sources_set(id_sources.begin(), id_sources.end());
  std::vector<int64> id_targets;
  id_sources.reserve(target.size());
  for (void* t : target) {
    id_targets.push_back(vspace.TensorId(t));
  }
  BackpropInitialState state = PrepareBackprop(
      id_targets, tensor_tape_, std::move(op_tape_), sources_set);
  std::vector<int64> op_stack =
      InitialStack(state.op_tape, state.op_missing_tensor);
  std::unordered_map<int64, std::vector<void*>> gradients;
  Status s = InitialGradients(vspace, target, output_gradients,
                              state.tensor_usage_counts, &gradients);
  auto cleanup = [&state]() {
    // Release all backprop functions
    for (const auto& pair : state.op_tape) {
      pair.second.backward_function_deleter();
    }
  };
  if (!s.ok()) {
    cleanup();
    return s;
  }
  std::unordered_map<int64, int64> gradients_size;
  // TODO(apassos) multiple threads could be dequeuing from op_stack at the same
  // time, for better CPU backprop performance.
  VLOG(1) << "Initial stack:";
  if (VLOG_IS_ON(1)) {
    for (auto t : op_stack) {
      VLOG(1) << "  " << t;
    }
  }
  std::unordered_map<string, std::unordered_set<int>>
      functions_accept_none_for_indices({
          {"SoftmaxCrossEntropyWithLogits", {1}},
          {"FusedBatchNorm", {1, 2, 3, 4}},
      });
  while (!op_stack.empty()) {
    const int64 op = op_stack.back();
    VLOG(1) << "Popped " << op;
    op_stack.pop_back();
    auto op_it = state.op_tape.find(op);
    if (op_it == state.op_tape.end()) {
      // It is possible for ops to end up on the stack if they are unrelated to
      // the target; we should just skip them.
      continue;
    }
    auto trace = std::move(op_it->second);
    state.op_tape.erase(op_it);
    std::vector<void*> out_gradients;
    out_gradients.reserve(trace.output_tensor_info.size());
    for (int i = 0; i < trace.output_tensor_info.size(); ++i) {
      const int64 id = trace.output_tensor_info[i].id;
      auto grad_it = gradients.find(id);
      if (grad_it == gradients.end()) {
        auto func_name_it =
            functions_accept_none_for_indices.find(trace.op_type);
        if (func_name_it != functions_accept_none_for_indices.end() &&
            func_name_it->second.find(i) != func_name_it->second.end()) {
          out_gradients.push_back(nullptr);
        } else {
          out_gradients.push_back(
              vspace.Zeros(trace.output_tensor_info[i].shape,
                           trace.output_tensor_info[i].dtype));
        }
      } else {
        out_gradients.push_back(vspace.AggregateGradients(grad_it->second));
        if (sources_set.find(grad_it->first) == sources_set.end()) {
          gradients.erase(grad_it);
        }
      }
    }
    std::vector<void*> in_gradients;
    Status s = vspace.CallBackwardFunction(trace.backward_function,
                                           out_gradients, &in_gradients);
    if (!s.ok()) {
      VLOG(1) << "Gradient function failed.";
      cleanup();
      return s;
    }
    VLOG(1) << "Got " << in_gradients.size() << " in_gradients for "
            << trace.input_tensor_id.size() << " sources";
    for (int i = 0; i < in_gradients.size(); ++i) {
      const int64 id = trace.input_tensor_id[i];
      if (in_gradients[i] != nullptr) {
        auto& unaggregated_grads = gradients[id];
        unaggregated_grads.push_back(in_gradients[i]);
        if (unaggregated_grads.size() > kMinAggregateCount) {
          auto size_it = gradients_size.find(id);
          int64 size;
          if (size_it == gradients_size.end()) {
            size = vspace.NumElements(unaggregated_grads[0]);
            gradients_size.emplace(id, size);
          } else {
            size = size_it->second;
          }
          if (unaggregated_grads.size() * size * 4 > kMinAggregateBytes) {
            void* tensor = vspace.AggregateGradients(unaggregated_grads);
            unaggregated_grads.clear();
            unaggregated_grads.push_back(tensor);
          }
        }
      }
      auto usage_count_it = state.tensor_usage_counts.find(id);
      if (usage_count_it == state.tensor_usage_counts.end()) {
        VLOG(1) << "Tensor " << id << " not used";
        continue;
      }
      usage_count_it->second--;
      if (usage_count_it->second > 0) {
        VLOG(1) << "Tensor " << id << " usage count " << usage_count_it->second;
        continue;
      }
      auto tape_it = tensor_tape_.find(id);
      if (tape_it == tensor_tape_.end()) {
        VLOG(1) << "Tensor " << id
                << " has no associated op. Deleting gradient";
        auto grad_it = gradients.find(id);
        if (grad_it != gradients.end()) {
          for (auto g : grad_it->second) {
            vspace.DeleteTensor(g);
          }
          gradients.erase(grad_it);
        }
        continue;
      }
      const int64 op_id = tape_it->second;
      if (op_id == -1) {
        VLOG(1) << "Tensor " << id << " is source";
        continue;
      }
      auto missing_it = state.op_missing_tensor.find(op_id);
      if (missing_it != state.op_missing_tensor.end()) {
        missing_it->second--;
        VLOG(1) << "Op " << op_id << " missing " << missing_it->second
                << " output gradients";
        if (missing_it->second == 0) {
          op_stack.push_back(op_id);
        }
      }
    }
  }
  CHECK(state.op_tape.empty());
  result->reserve(sources.size());
  for (auto is : id_sources) {
    auto grad_it = gradients.find(is);
    if (grad_it == gradients.end()) {
      result->push_back(nullptr);
    } else {
      if (grad_it->second.size() == 1) {
        result->push_back(grad_it->second[0]);
      } else {
        result->push_back(vspace.AggregateGradients(grad_it->second));
      }
      gradients.erase(grad_it);
    }
  }
  VLOG(1) << "Final gradients size: " << gradients.size();
  for (auto grad_pair : gradients) {
    for (const auto& g : grad_pair.second) {
      vspace.DeleteTensor(g);
    }
  }
  return Status::OK();
}

}  // namespace eager
}  // namespace tensorflow
