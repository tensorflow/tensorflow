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

#include <stack>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
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
using TensorTape = std::unordered_map<int64, int64>;

// Map from operation-id to tape entry.
template <typename BackwardFunction, typename TapeTensor>
using OpTape =
    std::unordered_map<int64, OpTapeEntry<BackwardFunction, TapeTensor>>;

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

  // Calls the passed-in backward function.
  virtual Status CallBackwardFunction(
      BackwardFunction* backward_function,
      const std::vector<int64>& unneeded_gradients,
      gtl::ArraySlice<Gradient*> output_gradients,
      std::vector<Gradient*>* result) const = 0;

  // Looks up the ID of a Gradient.
  virtual int64 TensorId(Gradient* tensor) const = 0;

  // Converts a Gradient to a TapeTensor.
  virtual TapeTensor TapeTensorFromGradient(Gradient* gradient) const = 0;

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
      const string& op_type, const std::vector<TapeTensor>& output_tensors,
      gtl::ArraySlice<int64> input_tensor_id,
      gtl::ArraySlice<tensorflow::DataType> input_dtypes,
      const std::function<BackwardFunction*()>& backward_function_getter,
      const std::function<void(BackwardFunction*)>& backward_function_deleter);

  void DeleteTrace(int64 tensor_id);

  // Consumes the internal state of the tape (so cannot be called more than
  // once) and produces the gradient of the target tensors with respect to the
  // source tensors. The output gradients are used if not empty and not
  // null. The result is populated with one tensor per target element.
  Status ComputeGradient(
      const VSpace<Gradient, BackwardFunction, TapeTensor>& vspace,
      const gtl::ArraySlice<int64> target_tensor_ids,
      const gtl::ArraySlice<int64> source_tensor_ids,
      const std::unordered_map<int64, TapeTensor>& sources_that_are_targets,
      gtl::ArraySlice<Gradient*> output_gradients,
      std::vector<Gradient*>* result);

  bool IsPersistent() const { return persistent_; }

 private:
  TensorTape tensor_tape_;
  OpTape<BackwardFunction, TapeTensor> op_tape_;
  int64 next_op_id_{0};

  // Map from tensor id to number of remaining usages (i.e. how many entries in
  // the tape refer to it); to aid in tape garbage collection.
  std::unordered_map<int64, int64> tensor_usage_;

  // If false, all activations are deleted in the first call to ComputeGradient.
  // Else, only when this is destructed.
  bool persistent_;
};

// Describes a callback for special-cased and more efficient jvp computation.
//
// Could just be a simple typedef in ForwardAccumulator, but MSVC chokes on
// that.
template <typename Gradient>
class ForwardFunction
    : public std::function<Status(const std::vector<Gradient*>&,
                                  std::vector<Gradient*>*)> {
 public:
  template <typename lambda_type>
  explicit ForwardFunction(lambda_type lambda)
      : std::function<Status(const std::vector<Gradient*>&,
                             std::vector<Gradient*>*)>(lambda) {}
};

// Computes Jacobian-vector products using forward-mode automatic
// differentiation.
//
// While GradientTape's RecordOperation is trivial, ForwardAccumulator's
// Accumulate runs the gradient computation immediately.
//
// Keeps references to Tensors watched via Watch and computed in Accumulate
// corresponding to output_tensors, and releases these references in its
// destructor. However, waiting until the destructor runs loses the memory
// efficiency of forward-mode autodiff. Instead, language bindings should call
// DeleteGradient as soon as a Tensor which was `Watch`ed or was an output
// Tensor passed to Accumulate goes out of scope.
//
// Not thread-safe.
template <typename Gradient, typename BackwardFunction, typename TapeTensor>
class ForwardAccumulator {
 public:
  // Does not take ownership of `vspace`, which must outlive the
  // ForwardAccumulator.
  explicit ForwardAccumulator(
      const VSpace<Gradient, BackwardFunction, TapeTensor>& vspace)
      : vspace_(vspace) {
    call_state_.emplace(nullptr, false);
  }

  virtual ~ForwardAccumulator() {
    for (auto accumulated : accumulated_gradients_) {
      vspace_.DeleteGradient(accumulated.second);
    }
  }

  // Tell the forward accumulator to watch tensor_id, with a Tensor tangent
  // vector `tangent` of matching shape and dtype. Tangents are the "vector" in
  // "Jacobian-vector product"; `Watch`ing a new Tensor and immediately calling
  // FetchJVP for it would return `tangent`.
  void Watch(int64 tensor_id, Gradient* tangent);

  // Removes the gradient associated with tensor_id. Should be called when the
  // Tensor associated with `tensor_id` is deleted.
  void DeleteGradient(int64 tensor_id);

  // Runs forward autodiff. Should be called whenever a new operation is
  // available and the accumulator is active.
  //
  // Like GradientTape::RecordOperation, this method takes the operation type
  // `op_type` (e.g. "Add"), the operation's inputs (`input_tensors`,
  // `input_tensor_id`, and `input_dtypes`; the latter two are somewhat
  // redundant but taken as arguments to avoid repeatedly fetching these values
  // between calls to ShouldRecord and Accumulator), and its outputs
  // (`output_tensors`).
  //
  // If provided, a non-null `forward_function` will be used instead of the
  // backward function (`backward_function_getter` /
  // `backward_function_deleter`) to compute jvps for this operation. If
  // `forward_function` is null, a GradientTape is used on the backward function
  // to compute the jvp, which will waste computation when executing eagerly.
  //
  // Unlike GradientTape::RecordOperation, Accumulate runs gradient computation
  // immediately. It stores the results, which feed into Accumulate for future
  // operations and may be fetched by calling FetchJVP. ForwardAccumulator
  // maintains a reference to these JVPs: if an `output_tensors` Tensor is
  // deleted, `DeleteGradient` should be called as soon as possible to free the
  // (now inaccessible) corresponding JVPs, but ForwardAccumulator's destructor
  // will release remaining references.
  //
  // This method is not thread-safe (and in general ForwardAccumulator is not
  // thread-safe).
  Status Accumulate(
      const string& op_type, const std::vector<TapeTensor>& input_tensors,
      const std::vector<TapeTensor>& output_tensors,
      gtl::ArraySlice<int64> input_tensor_id,
      gtl::ArraySlice<tensorflow::DataType> input_dtypes,
      const ForwardFunction<Gradient>* forward_function,
      const std::function<BackwardFunction*()>& backward_function_getter,
      const std::function<void(BackwardFunction*)>& backward_function_deleter);

  // Returns true if `Accumulate` is active somewhere above on the stack and
  // there isn't an intervening PushState. This is useful for ordering
  // ForwardAccumulators, where more deeply nested accumulators should not see
  // computations from less deeply nested accumulators.
  bool BusyAccumulating() const { return call_state_.top().accumulating; }

  // Fetches the current Jacobian-vector product associated with `tensor_id`, or
  // a nullptr if none is available.
  //
  // Returns a borrowed reference, i.e. does not run VSpace::MarkAsResult on its
  // return value. The caller should increment the reference count before
  // deleting the ForwardAccumulator or calling DeleteGradient if keeping a
  // persistent reference to a non-null result.
  Gradient* FetchJVP(int64 tensor_id);

  // Indicates whether the forward accumulator should run on an operation with
  // the specified inputs and dtypes.
  bool ShouldRecord(gtl::ArraySlice<int64> tensor_ids,
                    gtl::ArraySlice<tensorflow::DataType> dtypes);

  // Temporarily push or pop transient state for this accumulator.
  //
  // Allows an accumulator which is currently processing an operation to
  // temporarily reset its state. Without pushing and popping, accumulators
  // ignore operations executed as a direct result of their own jvp
  // computations.
  void PushState() { call_state_.emplace(nullptr, false); }
  void PopState() { call_state_.pop(); }

 private:
  // Helper for Accumulate: uses a GradientTape to compute forward gradients
  // from a backward gradient function. Fills `out_grads` corresponding to
  // `output_tensors`. `out_grads` must not be null.
  //
  // Executes the backward function in order to trace its gradient, which will
  // waste computation if executing eagerly (when graph building the unneeded
  // computation is pruned). Temporarily sets `backward_tape` so that
  // Accumulate will forward op executions to the tape while the backward
  // function is running; this effectively adds the backward tape to the active
  // set (but does not require complicated callbacks to the language bindings).
  Status ForwardpropFromTape(
      const std::vector<TapeTensor>& output_tensors,
      const std::function<BackwardFunction*()>& backward_function_getter,
      const std::function<void(BackwardFunction*)>& backward_function_deleter,
      const std::vector<Gradient*>& in_grads,
      std::vector<Gradient*>* out_grads);

  // Maps from tensor IDs to corresponding JVPs.
  std::unordered_map<int64, Gradient*> accumulated_gradients_;
  // Not owned; provides operations on Tensors which are currently only
  // available in language bindings (e.g. Python).
  const VSpace<Gradient, BackwardFunction, TapeTensor>& vspace_;

  struct AccumulatorCallState {
    AccumulatorCallState(
        GradientTape<Gradient, BackwardFunction, TapeTensor>* backward_tape,
        bool accumulating)
        : backward_tape(backward_tape), accumulating(accumulating) {}
    // Set temporarily while in the Accumulate method; if backward_tape is not
    // nullptr then we forward op executions to it so Accumulate can compute a
    // backward pass on its backward function.
    //
    // Not owned by the ForwardAccumulator. The method which sets
    // `backward_tape` keeps ownership.
    GradientTape<Gradient, BackwardFunction, TapeTensor>* backward_tape;
    // While the Accumulate method is running (accumulating is True), any op
    // executions not forwarded to backward_tape should be ignored.
    bool accumulating;
  };
  // A deque-backed stack, whose element references are not invalidated by
  // pushes and pops at the back.
  std::stack<AccumulatorCallState> call_state_;
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
    const string& op_type, const std::vector<TapeTensor>& output_tensors,
    gtl::ArraySlice<int64> input_tensor_id,
    gtl::ArraySlice<tensorflow::DataType> input_dtypes,
    const std::function<BackwardFunction*()>& backward_function_getter,
    const std::function<void(BackwardFunction*)>& backward_function_deleter) {
  if (!ShouldRecord(input_tensor_id, input_dtypes)) {
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
      op_type, std::move(tensors), std::move(ids), backward_function_getter(),
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
  std::unordered_map<int64, int64> tensor_usage_counts;

  // Maps from op ID to how many output tensors of this op still need to have
  // their gradients computed.
  std::unordered_map<int64, int64> op_missing_tensor;
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
    const std::unordered_set<int64>& sources_set, bool persistent_tape) {
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
    const std::unordered_map<int64, int64>& op_missing_tensor) {
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
    const std::unordered_map<int64, TapeTensor>& sources_that_are_targets,
    gtl::ArraySlice<Gradient*> output_gradients, const TensorTape& tensor_tape,
    const OpTape<BackwardFunction, TapeTensor>& op_tape,
    std::unordered_map<int64, std::vector<Gradient*>>* result) {
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
                op_it->second.output_tensor_info[j].OnesLike());
            break;
          }
        }
        if (!found) {
          return errors::Internal(
              "Internal state of the gradient tape is invalid: "
              "none of operations outputs match expected tensor");
        }
      } else {
        // This target tensor was not generated by any operation recorded on
        // the tape, so no gradient needs to be computed from it unless this
        // target is also a source.
        auto source_tensor = sources_that_are_targets.find(id);
        if (source_tensor != sources_that_are_targets.end()) {
          (*result)[id].push_back(source_tensor->second.OnesLike());
        }
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
std::unordered_map<string, std::unordered_set<int>>*
FunctionsAcceptingNoneForIndicesMap() {
  static auto* const m =
      new std::unordered_map<string, std::unordered_set<int>>({
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
    const gtl::ArraySlice<int64> target_tensor_ids,
    const gtl::ArraySlice<int64> source_tensor_ids,
    const std::unordered_map<int64, TapeTensor>& sources_that_are_targets,
    gtl::ArraySlice<Gradient*> output_gradients,
    std::vector<Gradient*>* result) {
  std::unordered_set<int64> sources_set(source_tensor_ids.begin(),
                                        source_tensor_ids.end());
  BackpropInitialState<BackwardFunction, TapeTensor> state = PrepareBackprop(
      target_tensor_ids, tensor_tape_, &op_tape_, sources_set, persistent_);
  std::vector<int64> op_stack =
      InitialStack(state.op_tape, state.op_missing_tensor);
  std::unordered_map<int64, std::vector<Gradient*>> gradients;
  Status s = InitialGradients(vspace, target_tensor_ids,
                              sources_that_are_targets, output_gradients,
                              tensor_tape_, state.op_tape, &gradients);
  auto cleanup = gtl::MakeCleanup([this, &state]() {
    if (!persistent_) {
      // Release all backprop functions
      for (const auto& pair : state.op_tape) {
        pair.second.backward_function_deleter(pair.second.backward_function);
      }
    }
  });
  if (!s.ok()) {
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
    std::vector<int64> unneeded_gradients;
    for (int i = 0; i < trace.input_tensor_id.size(); i++) {
      const auto& in_tensor_id = trace.input_tensor_id[i];
      if (tensor_tape_.find(in_tensor_id) == tensor_tape_.end() &&
          sources_set.find(in_tensor_id) == sources_set.end()) {
        unneeded_gradients.push_back(i);
      }
    }

    bool any_gradient_nonzero = false;
    std::vector<int> zero_indices;
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
          out_gradients.push_back(nullptr);
          zero_indices.push_back(i);
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
      for (const auto i : zero_indices) {
        out_gradients[i] = trace.output_tensor_info[i].ZerosLike();
      }
      Status s;
      s = vspace.CallBackwardFunction(trace.backward_function,
                                      unneeded_gradients, out_gradients,
                                      &in_gradients);
      if (in_gradients.size() != trace.input_tensor_id.size()) {
        return tensorflow::errors::Internal(
            "Recorded operation '", trace.op_type,
            "' returned too few gradients. Expected ",
            trace.input_tensor_id.size(), " but received ",
            in_gradients.size());
      }
      if (!persistent_) {
        trace.backward_function_deleter(trace.backward_function);
      }
      if (!s.ok()) {
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
          op_stack.insert(op_stack.begin(), op_id);
        }
      }
    }
  }
  if (!state.op_tape.empty()) {
    return tensorflow::errors::Internal("Invalid tape state.");
  }
  result->reserve(source_tensor_ids.size());
  std::unordered_set<int64> used_gradient_ids(source_tensor_ids.size());
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

template <typename Gradient, typename BackwardFunction, typename TapeTensor>
bool ForwardAccumulator<Gradient, BackwardFunction, TapeTensor>::ShouldRecord(
    gtl::ArraySlice<int64> tensor_ids,
    gtl::ArraySlice<tensorflow::DataType> dtypes) {
  if (call_state_.top().backward_tape != nullptr) {
    // If we're forwarding Accumulate calls to backward_tape's RecordOperation,
    // we should also delegate ShouldRecord.
    return call_state_.top().backward_tape->ShouldRecord(tensor_ids, dtypes);
  }
  if (call_state_.top().accumulating) {
    return false;
  }
  for (int i = 0; i < tensor_ids.size(); ++i) {
    if (accumulated_gradients_.find(tensor_ids[i]) !=
        accumulated_gradients_.end()) {
      if (IsDtypeTrainable(dtypes[i])) {
        return true;
      }
    }
  }
  return false;
}

template <typename Gradient, typename BackwardFunction, typename TapeTensor>
Status
ForwardAccumulator<Gradient, BackwardFunction, TapeTensor>::ForwardpropFromTape(
    const std::vector<TapeTensor>& output_tensors,
    const std::function<BackwardFunction*()>& backward_function_getter,
    const std::function<void(BackwardFunction*)>& backward_function_deleter,
    const std::vector<Gradient*>& in_grads, std::vector<Gradient*>* out_grads) {
  /* This function is approximately equivalent to this Python code:

  forwardprop_aids = tf.ones_like(output_tensors)
  with tf.GradientTape() as g:
    g.watch(forwardprop_aids)
    grad = backward_function(forwardprop_aids)
  forward_grads = g.gradient(grad, forwardprop_aids, output_gradients=in_grads)
  accumulated_gradients_[ID(output_tensors)] = forward_grads
  */
  std::unique_ptr<GradientTape<Gradient, BackwardFunction, TapeTensor>> tape(
      new GradientTape<Gradient, BackwardFunction, TapeTensor>(false));
  AccumulatorCallState& call_state = call_state_.top();
  call_state.backward_tape = tape.get();
  auto pop_backward_tape =
      gtl::MakeCleanup([&call_state] { call_state.backward_tape = nullptr; });
  std::vector<Gradient*> forwardprop_aids;
  std::vector<int64> sources;
  std::unordered_set<int64> sources_set;
  sources.reserve(output_tensors.size());
  for (const TapeTensor& output_tensor : output_tensors) {
    // Ownership of `aid` transferred to CallBackwardFunction below.
    Gradient* aid;
    if (output_tensor.GetDType() == tensorflow::DT_VARIANT) {
      // Note: Needs to be zeros rather than ones since there's currently no
      // ones_like for variants.
      aid = output_tensor.ZerosLike();
    } else {
      // TODO(allenl): Figure out why using zeros_like everywhere causes issues
      // for some gradient functions and if there's another way to work around
      // it (e.g. conds instead of ifs). The value shouldn't really matter.
      aid = output_tensor.OnesLike();
    }
    if (TF_PREDICT_FALSE(aid == nullptr)) {
      return tensorflow::errors::Internal(
          "Failed to create ones tensor for tensor ", output_tensor.GetID(),
          " with dtype ", output_tensor.GetDType());
    }
    forwardprop_aids.push_back(aid);
    int64 aid_id = vspace_.TensorId(aid);
    sources.push_back(aid_id);
    sources_set.insert(aid_id);
    tape->Watch(aid_id);
  }
  std::vector<Gradient*> grad;
  auto delete_grad = gtl::MakeCleanup([&grad, this] {
    for (Gradient* tensor : grad) {
      this->vspace_.DeleteGradient(tensor);
    }
  });
  {
    std::vector<int64> unneeded_gradients;
    std::unique_ptr<BackwardFunction, std::function<void(BackwardFunction*)>>
        backward_function(backward_function_getter(),
                          backward_function_deleter);
    TF_RETURN_IF_ERROR(vspace_.CallBackwardFunction(
        backward_function.get(), unneeded_gradients, forwardprop_aids, &grad));
  }

  // Stop the tape from recording
  pop_backward_tape.release()();

  if (grad.size() != in_grads.size()) {
    return tensorflow::errors::Internal("Wrong number of gradients returned.");
  }

  std::vector<int64> targets;
  std::vector<Gradient*> used_in_grads;
  // We may end up with slightly fewer elements than we reserve, but grad.size()
  // should be a reasonably tight upper bound.
  targets.reserve(grad.size());
  used_in_grads.reserve(grad.size());
  std::unordered_map<int64, TapeTensor> sources_that_are_targets;
  for (int grad_index = 0; grad_index < grad.size(); ++grad_index) {
    Gradient* grad_tensor = grad[grad_index];
    if (grad_tensor != nullptr) {
      int64 tensor_id = vspace_.TensorId(grad_tensor);
      targets.push_back(tensor_id);
      if (sources_set.find(tensor_id) != sources_set.end()) {
        sources_that_are_targets.emplace(
            tensor_id, vspace_.TapeTensorFromGradient(grad_tensor));
      }
      Gradient* in_grad = in_grads[grad_index];
      if (in_grad != nullptr) {
        // ComputeGradient steals a reference
        vspace_.MarkAsResult(in_grad);
      }
      used_in_grads.push_back(in_grad);
    }
  }

  return tape->ComputeGradient(vspace_, targets, sources,
                               sources_that_are_targets, used_in_grads,
                               out_grads);
}

template <typename Gradient, typename BackwardFunction, typename TapeTensor>
Status ForwardAccumulator<Gradient, BackwardFunction, TapeTensor>::Accumulate(
    const string& op_type, const std::vector<TapeTensor>& input_tensors,
    const std::vector<TapeTensor>& output_tensors,
    gtl::ArraySlice<int64> input_tensor_id,
    gtl::ArraySlice<tensorflow::DataType> input_dtypes,
    const ForwardFunction<Gradient>* forward_function,
    const std::function<BackwardFunction*()>& backward_function_getter,
    const std::function<void(BackwardFunction*)>& backward_function_deleter) {
  if (call_state_.top().backward_tape != nullptr) {
    // If backward_tape is not null, then this call to Accumulate is the result
    // of a still-active call to Accumulate which is running operations. We
    // forward these operations to backward_tape so the outer Accumulate call
    // can do its work.
    //
    // Rather than re-entering and delegating Accumulate like this, we could
    // instead allow ForwardAccumulator some control over the current tape set
    // (so it can deactivate itself and activate its GradientTape). Currently
    // that is managed by the language binding and would require relatively
    // messy callbacks.
    call_state_.top().backward_tape->RecordOperation(
        op_type, output_tensors, input_tensor_id, input_dtypes,
        backward_function_getter, backward_function_deleter);
    return Status::OK();
  }
  if (!ShouldRecord(input_tensor_id, input_dtypes)) {
    return Status::OK();
  }

  // We may need to allocate zero inputs for trainable dtypes we don't have JVPs
  // for. Make sure they get cleaned up.
  std::vector<Gradient*> new_zeros;
  auto delete_new_zeros = gtl::MakeCleanup([&new_zeros, this] {
    for (Gradient* tensor : new_zeros) {
      this->vspace_.DeleteGradient(tensor);
    }
  });
  std::vector<Gradient*> in_grads;
  in_grads.reserve(input_tensors.size());
  for (int target_index = 0; target_index < input_tensors.size();
       ++target_index) {
    const auto current_grad =
        accumulated_gradients_.find(input_tensors[target_index].GetID());
    if (current_grad == accumulated_gradients_.end()) {
      if (IsDtypeTrainable(input_tensors[target_index].GetDType())) {
        // ForwardAccumulator defaults to zeros for unwatched Tensors, unlike
        // GradientTape which uses ones.
        Gradient* zero = input_tensors[target_index].ZerosLike();
        new_zeros.push_back(zero);
        in_grads.push_back(zero);
      } else {
        in_grads.push_back(nullptr);
      }
    } else {
      in_grads.push_back(current_grad->second);
    }
  }

  // Avoid infinite recursion. Whichever forward function we run, it'll end up
  // executing ops, and we don't want to watch those with this accumulator.
  call_state_.emplace(nullptr, true);
  auto pop_call_state = gtl::MakeCleanup([this] { this->call_state_.pop(); });

  std::vector<Gradient*> forward_grads;
  if (forward_function == nullptr) {
    // We have no special-cased forward gradient. Fall back to running the
    // backward function under a gradient tape.
    TF_RETURN_IF_ERROR(ForwardpropFromTape(
        output_tensors, backward_function_getter, backward_function_deleter,
        in_grads, &forward_grads));
  } else {
    TF_RETURN_IF_ERROR((*forward_function)(in_grads, &forward_grads));
  }
  for (int i = 0; i < forward_grads.size(); ++i) {
    if (forward_grads[i] != nullptr) {
      int64 tensor_id = output_tensors[i].GetID();
      auto existing = accumulated_gradients_.find(tensor_id);
      if (existing != accumulated_gradients_.end()) {
        // This is a somewhat odd case to be in, since it means we have two
        // operations which supposedly both created the same Tensor. It comes up
        // in recompute_grad, where the gradients have the same value. However,
        // only the original gradient is connected to everything else, so we
        // should still use that.
        vspace_.DeleteGradient(forward_grads[i]);
      } else {
        accumulated_gradients_[output_tensors[i].GetID()] = forward_grads[i];
      }
    }
  }
  return Status::OK();
}

template <typename Gradient, typename BackwardFunction, typename TapeTensor>
void ForwardAccumulator<Gradient, BackwardFunction, TapeTensor>::Watch(
    int64 tensor_id, Gradient* tangent) {
  typename std::unordered_map<int64, Gradient*>::iterator existing =
      accumulated_gradients_.find(tensor_id);
  vspace_.MarkAsResult(tangent);
  if (existing == accumulated_gradients_.end()) {
    accumulated_gradients_.emplace(tensor_id, tangent);
  } else {
    std::array<Gradient*, 2> to_aggregate;
    to_aggregate[0] = tangent;
    to_aggregate[1] = existing->second;
    // AggregateGradients steals a reference to each of its arguments. We
    // MarkAsResult on `tangent` above so we don't steal a reference to it.
    existing->second = vspace_.AggregateGradients(to_aggregate);
  }
}

template <typename Gradient, typename BackwardFunction, typename TapeTensor>
void ForwardAccumulator<Gradient, BackwardFunction, TapeTensor>::DeleteGradient(
    int64 tensor_id) {
  auto existing = accumulated_gradients_.find(tensor_id);
  if (existing != accumulated_gradients_.end()) {
    vspace_.DeleteGradient(existing->second);
    accumulated_gradients_.erase(existing);
  }
}

template <typename Gradient, typename BackwardFunction, typename TapeTensor>
Gradient* ForwardAccumulator<Gradient, BackwardFunction, TapeTensor>::FetchJVP(
    int64 tensor_id) {
  auto lookup = accumulated_gradients_.find(tensor_id);
  if (lookup == accumulated_gradients_.end()) {
    return nullptr;
  } else {
    return lookup->second;
  }
}

}  // namespace eager
}  // namespace tensorflow

#endif  // TENSORFLOW_C_EAGER_TAPE_H_
