/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xrt/xrt_util.h"

#include <stdlib.h>
#include <string.h>

#include "tensorflow/compiler/xla/debug_options_flags.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
namespace {

// The ScopedHandles data structure is used in the ExecuteChained() API and its
// task is to track tuple allocation registrations. It is used both the track
// intermediate results of a chained computation, or its final results. Anything
// which is marked to be released, will be released using the XRTMemoryManager
// once the object is destroyed (unless an explicit call to Drop() or Release()
// is made).
class ScopedHandles {
 public:
  explicit ScopedHandles(RefPtr<XRTMemoryManager> memory_manager)
      : memory_manager_(std::move(memory_manager)) {}

  ~ScopedHandles() {
    for (size_t i = 0; i < handles_.size(); ++i) {
      if (handles_release_[i]) {
        memory_manager_->Release(handles_[i]).IgnoreError();
      }
    }
  }

  int64 operator[](size_t index) const { return handles_.at(index); }

  size_t size() const { return handles_.size(); }

  // Adds the given handle at the index position, by marking it releasable
  // according to the release argument. If an existing, and to-be-released
  // handle already exists at the same index, it will be released.
  Status Add(size_t index, int64 handle, bool release) {
    if (index >= handles_.size()) {
      handles_.resize(index + 1, XRTMemoryManager::InvalidKey());
      handles_release_.resize(index + 1, false);
    }
    if (handles_release_[index]) {
      Status status = memory_manager_->Release(handles_[index]);
      if (!status.ok()) {
        if (release) {
          memory_manager_->Release(handle).IgnoreError();
        }
        return status;
      }
    }
    handles_[index] = handle;
    handles_release_[index] = release;
    return Status::OK();
  }

  // Adds a to-be-released tuple allocation at the given index.
  Status Add(size_t index, RefPtr<XRTTupleAllocation> tuple) {
    return Add(index, memory_manager_->Register(std::move(tuple)),
               /*release=*/true);
  }

  // Drops the handle at the given index, and releases it using the
  // XRTMemoryManager::Release() if marked as to-be-released.
  Status Drop(size_t index) {
    if (handles_release_.at(index)) {
      TF_RETURN_IF_ERROR(memory_manager_->Release(handles_[index]));
    }
    Release(index);
    return Status::OK();
  }

  // Releases the handle at the given index. The destructor will not use that
  // XRTMemoryManager::Release() API on such handle.
  int64 Release(size_t index) {
    int64 handle = handles_.at(index);
    handles_[index] = XRTMemoryManager::InvalidKey();
    handles_release_[index] = false;
    return handle;
  }

  // Looks up the handle stored at the given index, and returns the matching
  // tuple allocation.
  xla::StatusOr<RefPtr<XRTTupleAllocation>> Lookup(size_t index) const {
    return memory_manager_->Lookup(handles_.at(index));
  }

 private:
  RefPtr<XRTMemoryManager> memory_manager_;
  std::vector<int64> handles_;
  std::vector<bool> handles_release_;
};

bool DebugOptionsPassThroughEnabled() {
  const char* env = getenv("TF_XLA_DEBUG_OPTIONS_PASSTHROUGH");
  bool enabled =
      env != nullptr && (strcmp(env, "1") == 0 || strcmp(env, "true") == 0);
  if (enabled) {
    LOG(WARNING) << "Passing through XLA debug options!";
  } else {
    LOG(WARNING) << "TF_XLA_DEBUG_OPTIONS_PASSTHROUGH not set, not all options "
                    "will be retained";
  }
  return enabled;
}

string SafeDebugPath(const string& path) {
  if (path.empty() || path.compare(0, 5, "gs://") == 0 ||
      path.compare(0, 11, "bigstore://") == 0) {
    return path;
  }
  LOG(WARNING) << "Invalid config path (will be dropped): " << path;
  return string();
}

Status MakeOutput(const RefPtr<XRTTupleAllocation>& output, int64 index,
                  RefPtr<XRTTupleAllocation>* result) {
  if (index == 0) {
    *result = output;
  } else {
    XRTTupleAllocation* tuple;
    TF_RETURN_IF_ERROR(
        XRTTupleAllocation::MakeSubBuffer(output.get(), {index - 1}, &tuple,
                                          /*alias_parent_allocation=*/true));
    result->reset(tuple);
  }
  return Status::OK();
}

Status PopulateOpWorkingSet(xla::Backend* backend,
                            const xrt::XRTChainedExecuteOp& op,
                            int current_index, const ScopedHandles& outputs,
                            XRTMemoryManager::WorkingSet* working_set) {
  for (int i = 0; i < op.inputs_size(); ++i) {
    auto& input = op.inputs(i);
    if (input.op_index() >= current_index) {
      return errors::InvalidArgument(
          "Input index ", input.op_index(),
          " is above the current position: ", current_index);
    }
    TF_RETURN_IF_ERROR(
        working_set->LookupAndPin(backend, outputs[input.op_index()]));
  }
  return Status::OK();
}

}  // namespace

xla::DebugOptions BuildXlaDebugOptions(const xla::DebugOptions& ref_options) {
  static const bool options_passthrough = DebugOptionsPassThroughEnabled();
  if (options_passthrough) {
    return ref_options;
  }
  xla::DebugOptions options = xla::GetDebugOptionsFromFlags();
  options.set_xla_dump_to(SafeDebugPath(ref_options.xla_dump_to()));
  options.set_xla_dump_hlo_as_proto(ref_options.xla_dump_hlo_as_proto());
  options.set_xla_dump_hlo_as_text(ref_options.xla_dump_hlo_as_text());
  options.set_xla_dump_hlo_snapshots(ref_options.xla_dump_hlo_snapshots());
  options.set_xla_dump_hlo_pass_re(ref_options.xla_dump_hlo_pass_re());
  for (auto& pass : ref_options.xla_disable_hlo_passes()) {
    options.add_xla_disable_hlo_passes(pass);
  }
  return options;
}

xla::StatusOr<std::vector<InputCoords>> GetComputationInputs(
    OpKernelContext* context, const char* input_name) {
  OpInputList arg_list;
  TF_RETURN_IF_ERROR(context->input_list(input_name, &arg_list));
  // Concatenate all input uids from list of scalars-or-vectors carrying them.
  std::vector<InputCoords> input_coords;
  for (int i = 0; i < arg_list.size(); ++i) {
    const Tensor& arg = arg_list[i];
    if (TensorShapeUtils::IsScalar(arg.shape())) {
      input_coords.emplace_back(arg.scalar<int64>()());
    } else {
      TF_RET_CHECK(TensorShapeUtils::IsVector(arg.shape()));
      auto arg_vec = arg.vec<int64>();
      const int64 num_elts = arg.shape().dim_size(0);
      for (int i = 0; i < num_elts; ++i) {
        input_coords.emplace_back(arg_vec(i));
      }
    }
  }
  return std::move(input_coords);
}

Status CreateExecuteOutput(OpKernelContext* context,
                           XRTMemoryManager* memory_manager,
                           RefPtr<XRTTupleAllocation> output_tuple,
                           bool return_exploded_tuple) {
  if (return_exploded_tuple && output_tuple->on_host_shape().IsTuple()) {
    int64 tuple_element_count =
        xla::ShapeUtil::TupleElementCount(output_tuple->on_device_shape());
    Tensor* output_tensor;
    TF_RETURN_IF_ERROR(context->allocate_output(
        0, TensorShape({tuple_element_count}), &output_tensor));

    for (int64 i = 0; i < tuple_element_count; ++i) {
      XRTTupleAllocation* suballocation;
      TF_RETURN_IF_ERROR(XRTTupleAllocation::MakeSubBuffer(
          output_tuple.get(), {i}, &suballocation,
          /*alias_parent_allocation=*/false));
      output_tensor->vec<int64>()(i) = memory_manager->Register(suballocation);
    }
  } else {
    Tensor* output_tensor;
    TF_RETURN_IF_ERROR(
        context->allocate_output(0, TensorShape({}), &output_tensor));
    output_tensor->scalar<int64>()() =
        memory_manager->Register(std::move(output_tuple));
  }
  return Status::OK();
}

Status ExecuteChained(OpKernelContext* context,
                      const RefPtr<XRTMemoryManager>& memory_manager,
                      xla::Backend* backend, int device_ordinal,
                      const xrt::XRTChainedExecutePlan& plan,
                      const xrt::XRTChainedExecuteConfig& config,
                      const ChainedExecuteFn& execute_op) {
  // Create the vector which tracks the uses of the intermediate chained
  // operations outputs.
  std::vector<int64> uses(plan.ops_size(), 0);
  for (auto& op : plan.ops()) {
    for (auto& input : op.inputs()) {
      uses[input.op_index()] += 1;
    }
  }

  ScopedHandles outputs(memory_manager);
  ScopedHandles results(memory_manager);
  for (int i = 0; i < plan.ops_size(); ++i) {
    auto& op = plan.ops(i);
    if (op.op_oneof_case() == xrt::XRTChainedExecuteOp::kDataHandle) {
      // This operation is a device data load. Set the handle as output and
      // leave the release flag off, since this is not an intermediate output.
      TF_RETURN_IF_ERROR(outputs.Add(i, op.data_handle(), /*release=*/false));
    } else if (op.op_oneof_case() ==
               xrt::XRTChainedExecuteOp::kComputationHandle) {
      // This is an XRT execute operation, forward to the device specific
      // handler. Populating the working set makes sure the input allocations
      // for this execute operations are pinned to device memory.
      XRTMemoryManager::WorkingSet working_set(memory_manager);
      TF_RETURN_IF_ERROR(
          PopulateOpWorkingSet(backend, op, i, outputs, &working_set));
      TF_ASSIGN_OR_RETURN(auto tuple,
                          execute_op(op, working_set.PinnedTuples()));
      TF_RETURN_IF_ERROR(outputs.Add(i, std::move(tuple)));
    } else {
      return errors::InvalidArgument(
          "Undefined operation kind at post-order position ", i);
    }
    // If the result of this chained operation is an output result, feed the
    // results at the desired position.
    for (auto& output : op.outputs()) {
      TF_ASSIGN_OR_RETURN(auto tuple, outputs.Lookup(i));
      RefPtr<XRTTupleAllocation> result;
      TF_RETURN_IF_ERROR(MakeOutput(tuple, output.output_index(), &result));
      TF_RETURN_IF_ERROR(results.Add(output.result_index(), std::move(result)));
    }
    // Drop intermediate results which have no more users.
    for (auto& input : op.inputs()) {
      uses[input.op_index()] -= 1;
      if (uses[input.op_index()] == 0) {
        TF_RETURN_IF_ERROR(outputs.Drop(input.op_index()));
      }
    }
  }

  Tensor* output_tensor;
  TF_RETURN_IF_ERROR(context->allocate_output(
      0, TensorShape({static_cast<int64>(results.size())}), &output_tensor));
  for (size_t i = 0; i < results.size(); ++i) {
    output_tensor->vec<int64>()(i) = results.Release(i);
  }
  return Status::OK();
}

}  // namespace tensorflow
