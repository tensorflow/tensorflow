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
    OpKernelContext* context, ResourceMgr* rm, const char* input_name) {
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

Status CreateExecuteOutput(OpKernelContext* context, ResourceMgr* rm,
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
      int64 key;
      TF_RETURN_IF_ERROR(suballocation->Intern(rm, &key));
      output_tensor->vec<int64>()(i) = key;
    }
  } else {
    Tensor* output_tensor;
    TF_RETURN_IF_ERROR(
        context->allocate_output(0, TensorShape({}), &output_tensor));
    int64 key;
    TF_RETURN_IF_ERROR(output_tuple->Intern(rm, &key));
    output_tuple.release();
    output_tensor->scalar<int64>()() = key;
  }
  return Status::OK();
}

Status ExecuteChained(OpKernelContext* context, ResourceMgr* rm,
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
  std::vector<RefPtr<XRTTupleAllocation>> ops_outputs(plan.ops_size());
  std::vector<RefPtr<XRTTupleAllocation>> results;
  for (int i = 0; i < plan.ops_size(); ++i) {
    auto& op = plan.ops(i);
    if (op.op_oneof_case() == xrt::XRTChainedExecuteOp::kDataHandle) {
      // This operation is a device data load. Fetch the proper
      // XRTTupleAllocation behind the user handle and fill up the op output at
      // the current position.
      XRTTupleAllocation* tuple;
      TF_RETURN_IF_ERROR(
          XRTTupleAllocation::Lookup(rm, op.data_handle(), &tuple));
      ops_outputs[i].reset(tuple);
    } else if (op.op_oneof_case() ==
               xrt::XRTChainedExecuteOp::kComputationHandle) {
      // This is an XRT execute operation, forward to the device specific
      // handler.
      TF_ASSIGN_OR_RETURN(ops_outputs[i], execute_op(op, i, ops_outputs));
    } else {
      return errors::InvalidArgument(
          "Undefined operation kind at post-order position ", i);
    }
    // If the result of this chained operation is an output result, feed the
    // results vector at the desired position.
    for (auto& output : op.outputs()) {
      if (output.result_index() >= results.size()) {
        results.resize(output.result_index() + 1);
      }
      TF_RETURN_IF_ERROR(MakeOutput(ops_outputs[i], output.output_index(),
                                    &results[output.result_index()]));
    }
    // Drop intermediate results which have no more users.
    for (auto& input : op.inputs()) {
      uses[input.op_index()] -= 1;
      if (uses[input.op_index()] == 0) {
        ops_outputs[input.op_index()].reset();
      }
    }
  }

  Tensor* output_tensor;
  TF_RETURN_IF_ERROR(context->allocate_output(
      0, TensorShape({static_cast<int64>(results.size())}), &output_tensor));
  for (size_t i = 0; i < results.size(); ++i) {
    int64 key = XRTTupleAllocation::InvalidKey();
    if (results[i] != nullptr) {
      TF_RETURN_IF_ERROR(results[i]->Intern(rm, &key));
      results[i].release();
    }
    output_tensor->vec<int64>()(i) = key;
  }
  return Status::OK();
}

}  // namespace tensorflow
