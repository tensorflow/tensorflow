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
#ifndef TENSORFLOW_C_EAGER_TAPE_H_
#define TENSORFLOW_C_EAGER_TAPE_H_

// Language-agnostic gradient tape. Does not perform backpropagation, just
// maintains the data structures required to do so.

#include <vector>
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/lib/gtl/flatmap.h"
#include "tensorflow/core/lib/gtl/flatset.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace eager {

// Represents an entry in the tape.
template <typename BackwardFunction, typename TapeTensor>
struct OpTapeEntry {
  string op_type;
  std::vector<TapeTensor> output_tensor_info;
  std::vector<int64> input_tensor_id;

  // TODO(apassos) consider narrowing down this interface.
  BackwardFunction* backward_function;

  // Should be called before deleting the backward function. TODO(apassos) use
  // unique_ptrs to ensure this happens.
  std::function<void(BackwardFunction*)> backward_function_deleter;
};

// Map from tensor_id to internally-defined operation-id of the operation which
// produced this tensor. A value of -1 means that the tensor was directly
// watched and not the result of any operation in the tape.
using TensorTape = gtl::FlatMap<int64, int64>;

// Map from operation-id to tape entry.
template <typename BackwardFunction, typename TapeTensor>
using OpTape = gtl::FlatMap<int64, OpTapeEntry<BackwardFunction, TapeTensor>>;

// Operations the tape needs to perform on tensors to do backpropagation. Named
// "vspace" because a subset of these are related to a vector space, such as
// adding gradients, getting zeroes, etc. Currently cannot be implemented
// without using tensorflow python code, hence left unspecified here.
//
// Gradient is the type returned by gradient functions. In Python TF it's either
// Tensor or IndexedSlices or None, which here we map to nullptr. Gradients need
// to allow their size to be computed and they need to be passable to a backward
// function and deleted (as the backprop code creates lots of gradients the user
// is not interested in).
//
// BackwardFunction needs to be a closure which stores intermediate activations
// from the forward computation and calls a vector-jacobian product function
// (also known as adjoint function) to compute, given downstream gradients,
// upstream gradients.
//
// TODO(apassos) provide concrete template instantiations for TFE_TensorHandle
// specialization, which is blocked by quite a few things needing to loop back
// into python now.
template <typename Gradient, typename BackwardFunction, typename TapeTensor>
class VSpace {
 public:
  virtual ~VSpace() {}

  // Returns the number of elements in the gradient tensor.
  virtual int64 NumElements(Gradient* tensor) const = 0;

  // Consumes references to the tensors in the gradient_tensors list and returns
  // a tensor with the result.
  virtual Gradient* AggregateGradients(
      gtl::ArraySlice<Gradient*> gradient_tensors) const = 0;

  // Returns a tensor of the right shape and dtype filled with zeros.
  virtual Gradient* Zeros(const TapeTensor& tensor) const = 0;

  // Returns a Tensor which is filled with ones and like the input.
  virtual Gradient* Ones(const TapeTensor& tensor) const = 0;

  // Calls the passed-in backward function.
  virtual Status CallBackwardFunction(
      BackwardFunction* backward_function,
      gtl::ArraySlice<Gradient*> output_gradients,
      std::vector<Gradient*>* result) const = 0;

  // Marks the following gradient as a result so it's not consumed by backward
  // functions.
  virtual void MarkAsResult(Gradient* gradient) const = 0;

  // Deletes the input tensor.
  virtual void DeleteGradient(Gradient* gradient) const = 0;
};

// Traces the execution of operations, doing eager garbage collection, and
// exporting a full trace so other code can do backpropagation. Not thread-safe.
template <typename Gradient, typename BackwardFunction, typename TapeTensor>
class GradientTape {
 public:
  // If `persistent` is true, GradientTape will not eagerly delete backward
  // functions (and hence the tensors they keep alive). Instead, everything
  // is deleted in ~GradientTape. Persistent GradientTapes are useful when
  // users want to compute multiple gradients over the same tape.
  GradientTape(bool persistent) : persistent_(persistent) {}
  ~GradientTape() {
    for (const auto& pair : op_tape_) {
      pair.second.backward_function_deleter(pair.second.backward_function);
    }
  }

  bool ShouldRecord(gtl::ArraySlice<int64> tensor_ids,
                    gtl::ArraySlice<tensorflow::DataType> dtypes);

  void Watch(int64 tensor_id);

  void RecordOperation(
      const string& op_type, std::vector<TapeTensor>& output_tensors,
      gtl::ArraySlice<int64> input_tensor_id,
      gtl::ArraySlice<tensorflow::DataType> input_dtypes,
      BackwardFunction* backward_function,
      const std::function<void(BackwardFunction*)>& backward_function_deleter);

  void DeleteTrace(int64 tensor_id);

  // Consumes the internal state of the tape (so cannot be called more than
  // once) and produces the gradient of the target tensors with respect to the
  // source tensors. The output gradients are used if not empty and not
  // null. The result is populated with one tensor per target element.
  Status ComputeGradient(
      const VSpace<Gradient, BackwardFunction, TapeTensor>& vspace,
      gtl::ArraySlice<int64> target_tensor_ids,
      gtl::ArraySlice<int64> source_tensor_id,
      gtl::ArraySlice<Gradient*> output_gradients,
      std::vector<Gradient*>* result);

  bool IsPersistent() const { return persistent_; }

 private:
  TensorTape tensor_tape_;
  OpTape<BackwardFunction, TapeTensor> op_tape_;
  int64 next_op_id_{0};

  // Map from tensor id to number of remaining usages (i.e. how many entries in
  // the tape refer to it); to aid in tape garbage collection.
  gtl::FlatMap<int64, int64> tensor_usage_;

  // If false, all activations are deleted in the first call to ComputeGradient.
  // Else, only when this is destructed.
  bool persistent_;
};

// Template instantiations here

inline bool IsDtypeTrainable(DataType dtype) {
  switch (dtype) {
    case DT_HALF:
    case DT_BFLOAT16:
    case DT_FLOAT:
    case DT_DOUBLE:
    case DT_COMPLEX64:
    case DT_COMPLEX128:
    case DT_RESOURCE:
    case DT_VARIANT:
      return true;
    default:
      return false;
  }
}

template <typename Gradient, typename BackwardFunction, typename TapeTensor>
bool GradientTape<Gradient, BackwardFunction, TapeTensor>::ShouldRecord(
    gtl::ArraySlice<int64> tensor_ids,
    gtl::ArraySlice<tensorflow::DataType> dtypes) {
  CHECK_EQ(tensor_ids.size(), dtypes.size());
  for (int i = 0; i < tensor_ids.size(); ++i) {
    if (tensor_tape_.find(tensor_ids[i]) != tensor_tape_.end()) {
      if (IsDtypeTrainable(dtypes[i])) {
        return true;
      }
    }
  }
  return false;
}

template <typename Gradient, typename BackwardFunction, typename TapeTensor>
void GradientTape<Gradient, BackwardFunction, TapeTensor>::Watch(
    int64 tensor_id) {
  tensor_tape_.emplace(tensor_id, -1);
}

template <typename Gradient, typename BackwardFunction, typename TapeTensor>
void GradientTape<Gradient, BackwardFunction, TapeTensor>::RecordOperation(
    const string& op_type, std::vector<TapeTensor>& output_tensors,
    gtl::ArraySlice<int64> input_tensor_id,
    gtl::ArraySlice<tensorflow::DataType> input_dtypes,
    BackwardFunction* backward_function,
    const std::function<void(BackwardFunction*)>& backward_function_deleter) {
  if (!ShouldRecord(input_tensor_id, input_dtypes)) {
    backward_function_deleter(backward_function);
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
    tensor_tape_[o.GetID()] = op_id;
    tensor_usage_[o.GetID()] = 1;
    tensors.push_back(o);
  }
  op_tape_[op_id] = OpTapeEntry<BackwardFunction, TapeTensor>{
      op_type, std::move(tensors), ids, backward_function,
      backward_function_deleter};
}

template <typename Gradient, typename BackwardFunction, typename TapeTensor>
void GradientTape<Gradient, BackwardFunction, TapeTensor>::DeleteTrace(
    int64 tensor_id) {
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
    if (tensor_usage_.find(output.GetID()) != tensor_usage_.end()) {
      // Found a usage for an output, so cannot delete the op.
      return;
    }
  }
  for (int64 id : op_it->second.input_tensor_id) {
    DeleteTrace(id);
  }
  op_it->second.backward_function_deleter(op_it->second.backward_function);
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

namespace {

template <typename BackwardFunction, typename TapeTensor>
struct BackpropInitialState {
  OpTape<BackwardFunction, TapeTensor> op_tape;

  // Map from tensor ID to how many references still exist for this tensor in
  // the tape.
  gtl::FlatMap<int64, int64> tensor_usage_counts;

  // Maps from op ID to how many output tensors of this op still need to have
  // their gradients computed.
  gtl::FlatMap<int64, int64> op_missing_tensor;
};

// If `persistent_tape` is true, op_tape is not changed and none of the
// backwards functions are deleted.
// If `persistent_tape` is false, op_tape is cleared and backwards functions
// not needed for gradient computation are deleted. Backwards functions that
// are needed, are copied and returned in BackpropInitialState.
template <typename BackwardFunction, typename TapeTensor>
BackpropInitialState<BackwardFunction, TapeTensor> PrepareBackprop(
    gtl::ArraySlice<int64> target, const TensorTape& tensor_tape,
    OpTape<BackwardFunction, TapeTensor>* op_tape,
    const gtl::FlatSet<int64>& sources_set, bool persistent_tape) {
  std::vector<int64> tensor_stack;
  tensor_stack.reserve(target.size());
  for (auto t : target) {
    tensor_stack.push_back(t);
  }
  BackpropInitialState<BackwardFunction, TapeTensor> result;
  while (!tensor_stack.empty()) {
    int64 tensor_id = tensor_stack.back();
    tensor_stack.pop_back();
    auto op_id_it = tensor_tape.find(tensor_id);
    if (op_id_it == tensor_tape.end()) {
      continue;
    }
    int64 op_id = op_id_it->second;
    auto op_it = op_tape->find(op_id);
    auto result_op_it = result.op_tape.find(op_id);
    if (op_id == -1 || op_it == op_tape->end() ||
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
        if (tensor_tape.find(it) != tensor_tape.end()) {
          tensor_stack.push_back(it);
        }
      }
    }
    if (!persistent_tape) {
      op_tape->erase(op_it);
    }
  }
  for (auto& pair : result.tensor_usage_counts) {
    auto it = tensor_tape.find(pair.first);
    if (it != tensor_tape.end() && it->second != -1) {
      result.op_missing_tensor[it->second] += 1;
    }
  }
  if (!persistent_tape) {
    // Call destructors for all unneeded gradient functions and
    // clear the op_tape. We can clear the tape because ownership of
    // backward functions that will be used for gradient computation
    // has been transferred to `result`.
    for (const auto& op_pair : *op_tape) {
      op_pair.second.backward_function_deleter(
          op_pair.second.backward_function);
    }
    op_tape->clear();
  }
  return result;
}

template <typename BackwardFunction, typename TapeTensor>
std::vector<int64> InitialStack(
    const OpTape<BackwardFunction, TapeTensor>& op_tape,
    const gtl::FlatMap<int64, int64>& op_missing_tensor) {
  std::vector<int64> result;
  for (auto& op_entry : op_tape) {
    if (op_missing_tensor.find(op_entry.first) == op_missing_tensor.end()) {
      result.push_back(op_entry.first);
    }
  }
  return result;
}

template <typename Gradient, typename BackwardFunction, typename TapeTensor>
Status InitialGradients(
    const VSpace<Gradient, BackwardFunction, TapeTensor>& vspace,
    gtl::ArraySlice<int64> target_tensor_ids,
    gtl::ArraySlice<Gradient*> output_gradients, const TensorTape& tensor_tape,
    const OpTape<BackwardFunction, TapeTensor>& op_tape,
    gtl::FlatMap<int64, std::vector<Gradient*>>* result) {
  for (int i = 0; i < target_tensor_ids.size(); ++i) {
    const int64 id = target_tensor_ids[i];
    if (output_gradients.empty() || output_gradients[i] == nullptr) {
      auto tensor_it = tensor_tape.find(id);
      if (tensor_it != tensor_tape.end() && tensor_it->second != -1) {
        auto op_it = op_tape.find(tensor_it->second);
        if (op_it == op_tape.end()) {
          return errors::Internal(
              "Internal state of the gradient tape is invalid: "
              "failed to find operation producing a tensor");
        }
        bool found = false;
        for (int j = 0; j < op_it->second.output_tensor_info.size(); ++j) {
          if (op_it->second.output_tensor_info[j].GetID() == id) {
            found = true;
            (*result)[id].push_back(
                vspace.Ones(op_it->second.output_tensor_info[j]));
            break;
          }
        }
        if (!found) {
          return errors::Internal(
              "Internal state of the gradient tape is invalid: "
              "none of operations outputs match expected tensor");
        }
      } else {
        // No record of the target tensor found on the tape, so no gradient
        // needs to be computed from it. Do nothing.
      }
    } else {
      (*result)[id].push_back(output_gradients[i]);
    }
  }
  return Status::OK();
}

// TODO(agarwal): use an automatic mechanism for handling None arguments to
// gradient functions.
//
// Some gradient functions can accept None arguments for gradients. The
// following maps the operation name to the indices at which the corresponding
// gradient function can accept None values. e.g. FusedBatchNorm outputs 5
// values and hence receives 5 gradient values during backprop. However the
// gradient function uses only the first of those values and ignores the rest.
// The entry, "FusedBatchNorm": [1, 2, 3, 4], indicates that only the gradient
// corresponding to index 0 is used, and the gradient values at indices 1-4 are
// ignored (and hence can be None). The backprop algorithm can then leverage
// this by not constructing zeros to pass for those indices.
gtl::FlatMap<string, gtl::FlatSet<int>>* FunctionsAcceptingNoneForIndicesMap() {
  static auto* const m = new gtl::FlatMap<string, gtl::FlatSet<int>>({
      {"SoftmaxCrossEntropyWithLogits", {1}},
      {"SparseSoftmaxCrossEntropyWithLogits", {1}},
      {"FusedBatchNorm", {1, 2, 3, 4}},
  });
  return m;
}

}  // namespace

// If over kMinAggregateCount gradients are accumulated and the total
// memory consumption is over kMinAggregateBytes, do an early aggregation
// so as to release the gradient tensor to save memory.
constexpr int kMinAggregateCount = 4;
constexpr int kMinAggregateBytes = 128 * 1024 * 1024;

template <typename Gradient, typename BackwardFunction, typename TapeTensor>
Status GradientTape<Gradient, BackwardFunction, TapeTensor>::ComputeGradient(
    const VSpace<Gradient, BackwardFunction, TapeTensor>& vspace,
    gtl::ArraySlice<int64> target_tensor_ids,
    gtl::ArraySlice<int64> source_tensor_ids,
    gtl::ArraySlice<Gradient*> output_gradients,
    std::vector<Gradient*>* result) {
  gtl::FlatSet<int64> sources_set(source_tensor_ids.begin(),
                                  source_tensor_ids.end());
  BackpropInitialState<BackwardFunction, TapeTensor> state = PrepareBackprop(
      target_tensor_ids, tensor_tape_, &op_tape_, sources_set, persistent_);
  std::vector<int64> op_stack =
      InitialStack(state.op_tape, state.op_missing_tensor);
  gtl::FlatMap<int64, std::vector<Gradient*>> gradients;
  Status s = InitialGradients(vspace, target_tensor_ids, output_gradients,
                              tensor_tape_, state.op_tape, &gradients);
  auto cleanup = [this, &state]() {
    if (!persistent_) {
      // Release all backprop functions
      for (const auto& pair : state.op_tape) {
        pair.second.backward_function_deleter(pair.second.backward_function);
      }
    }
  };
  if (!s.ok()) {
    cleanup();
    return s;
  }
  gtl::FlatMap<int64, int64> gradients_size;
  // TODO(apassos) multiple threads could be dequeuing from op_stack at the same
  // time, for better CPU backprop performance.
  VLOG(1) << "Initial stack:";
  if (VLOG_IS_ON(1)) {
    for (auto t : op_stack) {
      VLOG(1) << "  " << t;
    }
  }
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
    std::vector<Gradient*> out_gradients;
    out_gradients.reserve(trace.output_tensor_info.size());
    bool any_gradient_nonzero = false;
    for (int i = 0; i < trace.output_tensor_info.size(); ++i) {
      const int64 id = trace.output_tensor_info[i].GetID();
      auto grad_it = gradients.find(id);
      if (grad_it == gradients.end()) {
        auto func_name_it =
            FunctionsAcceptingNoneForIndicesMap()->find(trace.op_type);
        if (func_name_it != FunctionsAcceptingNoneForIndicesMap()->end() &&
            func_name_it->second.find(i) != func_name_it->second.end()) {
          out_gradients.push_back(nullptr);
        } else {
          out_gradients.push_back(vspace.Zeros(trace.output_tensor_info[i]));
        }
      } else {
        any_gradient_nonzero = true;
        Gradient* new_gradients = nullptr;
        if (grad_it->second.size() == 1) {
          new_gradients = grad_it->second.at(0);
        } else {
          new_gradients = vspace.AggregateGradients(grad_it->second);
        }
        if (sources_set.find(grad_it->first) == sources_set.end()) {
          gradients.erase(grad_it);
        } else {
          grad_it->second.clear();
          grad_it->second.push_back(new_gradients);
          vspace.MarkAsResult(new_gradients);
        }
        out_gradients.push_back(new_gradients);
      }
    }
    std::vector<Gradient*> in_gradients;
    if (any_gradient_nonzero) {
      Status s = vspace.CallBackwardFunction(trace.backward_function,
                                             out_gradients, &in_gradients);
      if (!persistent_) {
        trace.backward_function_deleter(trace.backward_function);
      }
      if (!s.ok()) {
        cleanup();
        return s;
      }
    } else {
      in_gradients.resize(trace.input_tensor_id.size());
      if (!persistent_) {
        trace.backward_function_deleter(trace.backward_function);
      }
      for (Gradient* grad : out_gradients) {
        if (grad != nullptr) {
          vspace.DeleteGradient(grad);
        }
      }
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
            Gradient* grad = vspace.AggregateGradients(unaggregated_grads);
            unaggregated_grads.clear();
            unaggregated_grads.push_back(grad);
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
            vspace.DeleteGradient(g);
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
  result->reserve(source_tensor_ids.size());
  gtl::FlatSet<int64> used_gradient_ids(source_tensor_ids.size());
  for (auto is : source_tensor_ids) {
    auto grad_it = gradients.find(is);
    if (grad_it == gradients.end()) {
      result->push_back(nullptr);
    } else {
      if (grad_it->second.size() > 1) {
        Gradient* grad = vspace.AggregateGradients(grad_it->second);
        grad_it->second.clear();
        grad_it->second.push_back(grad);
      }
      result->push_back(grad_it->second[0]);
      used_gradient_ids.insert(is);
    }
  }
  VLOG(1) << "Final gradients size: "
          << gradients.size() - used_gradient_ids.size();
  for (auto grad_pair : gradients) {
    if (used_gradient_ids.find(grad_pair.first) == used_gradient_ids.end()) {
      for (const auto& g : grad_pair.second) {
        vspace.DeleteGradient(g);
      }
    }
  }
  return Status::OK();
}

}  // namespace eager
}  // namespace tensorflow

#endif  // TENSORFLOW_C_EAGER_TAPE_H_
