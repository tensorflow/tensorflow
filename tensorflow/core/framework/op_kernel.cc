/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/framework/op_kernel.h"

#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>  // NOLINT
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "absl/base/call_once.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/match.h"
#include "tensorflow/core/framework/allocation_description.pb.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/framework/device_attributes.pb.h"
#include "tensorflow/core/framework/device_factory.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/kernel_def.pb.h"
#include "tensorflow/core/framework/kernel_def_util.h"
#include "tensorflow/core/framework/log_memory.h"
#include "tensorflow/core/framework/memory_types.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/node_properties.h"
#include "tensorflow/core/framework/op_def_util.h"
#include "tensorflow/core/framework/tensor_reference.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/cpu_info.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/platform_strings.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/lib/scoped_memory_debug_annotation.h"
#include "tensorflow/core/profiler/lib/traceme.h"

namespace tensorflow {

const char* kJitKernelLabel = "JITCompiledKernel";
const char* kDisableJitKernelsEnvVar = "TF_DISABLE_JIT_KERNELS";

namespace {

absl::Status MatchSignatureHelper(const DataTypeSlice expected_inputs,
                                  const DataTypeSlice expected_outputs,
                                  const DataTypeSlice inputs,
                                  const DataTypeSlice outputs) {
  bool signature_mismatch = false;

  if (inputs.size() != expected_inputs.size()) signature_mismatch = true;
  for (size_t i = 0; !signature_mismatch && i < inputs.size(); ++i) {
    if (!TypesCompatible(expected_inputs[i], inputs[i])) {
      signature_mismatch = true;
    }
  }

  if (outputs.size() != expected_outputs.size()) signature_mismatch = true;
  for (size_t i = 0; !signature_mismatch && i < outputs.size(); ++i) {
    if (!TypesCompatible(expected_outputs[i], outputs[i])) {
      signature_mismatch = true;
    }
  }

  if (signature_mismatch) {
    return errors::InvalidArgument(
        "Signature mismatch, have: ", DataTypeSliceString(inputs), "->",
        DataTypeSliceString(outputs),
        " expected: ", DataTypeSliceString(expected_inputs), "->",
        DataTypeSliceString(expected_outputs));
  }
  return absl::OkStatus();
}

const absl::flat_hash_set<std::string>* GetOpNodeDefsToLogFromEnv() {
  auto* result = new absl::flat_hash_set<std::string>;
  const char* env = getenv("TF_DEBUG_OPS_TO_LOG_NODEDEFS");
  if (!env) {
    return result;
  }

  std::vector<absl::string_view> ops = absl::StrSplit(env, ',');
  LOG(INFO) << "Will log NodeDefs from the following ops: ";
  for (absl::string_view op : ops) {
    result->insert(std::string(op));
    LOG(INFO) << "  |" << op << "|";
  }

  return result;
}

// Returns true if the NodeDef for the OpKernel should be logged. The
// envionrmental variable TF_DEBUG_OPS_TO_LOG_NODEDEFS can be set to a
// comma-separated list of op types. The NodeDef for each is printed, which is
// useful for debugging purposes.
bool ShouldLogNodeDef(OpKernel* op_kernel) {
  static const absl::flat_hash_set<std::string>& ops_to_log_nodedefs =
      *GetOpNodeDefsToLogFromEnv();
  return ops_to_log_nodedefs.count(op_kernel->type_string());
}

}  // namespace

// OpKernel ------------------------------------------------------------------

OpKernel::OpKernel(OpKernelConstruction* context) : OpKernel(context, false) {}

OpKernel::OpKernel(OpKernelConstruction* context, bool is_deferred)
    : props_(context->props_),
      input_memory_types_(context->input_memory_types().begin(),
                          context->input_memory_types().end()),
      output_memory_types_(context->output_memory_types().begin(),
                           context->output_memory_types().end()),
      input_name_map_(context->num_inputs()),
      output_name_map_(context->num_outputs()),
      name_view_(props_->node_def.name()),
      type_string_view_(props_->node_def.op()),
      graph_def_version_(context->graph_def_version()),
      is_deferred_(is_deferred) {
  OP_REQUIRES_OK(context,
                 NameRangesForNode(props_->node_def, *props_->op_def,
                                   &input_name_map_, &output_name_map_));
  OP_REQUIRES_OK(context, CheckOpDeprecation(*props_->op_def,
                                             context->graph_def_version()));

  // Kernels executing on GPU tie very few resources on the CPU where the
  // scheduler runs: we consider them as inexpensive.
  expensive_ = context->device_type() != DeviceType(DEVICE_GPU) &&
               !DeviceFactory::IsPluggableDevice(
                   DeviceTypeString(context->device_type()));

  if (ShouldLogNodeDef(this)) {
    LOG(INFO) << "NodeDef for " << name() << ":\n" << def().ShortDebugString();
  }
}

OpKernel::OpKernel(OpKernelConstruction* context, NodeDef&& custom_def,
                   bool is_deferred)
    : props_(std::make_shared<const NodeProperties>(
          context->props_->op_def, std::move(custom_def),
          context->props_->input_types, context->props_->output_types)),
      input_memory_types_(context->input_memory_types().begin(),
                          context->input_memory_types().end()),
      output_memory_types_(context->output_memory_types().begin(),
                           context->output_memory_types().end()),
      input_name_map_(context->num_inputs()),
      output_name_map_(context->num_outputs()),
      name_view_(props_->node_def.name()),
      type_string_view_(props_->node_def.op()),
      graph_def_version_(context->graph_def_version()),
      is_deferred_(is_deferred) {
  OP_REQUIRES_OK(context,
                 NameRangesForNode(props_->node_def, *props_->op_def,
                                   &input_name_map_, &output_name_map_));
  OP_REQUIRES_OK(context, CheckOpDeprecation(*props_->op_def,
                                             context->graph_def_version()));

  // Kernels executing on GPU tie very few resources on the CPU where the
  // scheduler runs: we consider them as inexpensive.
  expensive_ = context->device_type() != DeviceType(DEVICE_GPU) &&
               !DeviceFactory::IsPluggableDevice(
                   DeviceTypeString(context->device_type()));
}

OpKernel::~OpKernel() {}

absl::Status OpKernel::InputRange(absl::string_view input_name, int* start,
                                  int* stop) const {
  const auto result = input_name_map_.find(input_name);
  if (result == input_name_map_.end()) {
    return errors::InvalidArgument("Unknown input name: ", input_name);
  } else {
    *start = result->second.first;
    *stop = result->second.second;
    return absl::OkStatus();
  }
}

absl::Status OpKernel::OutputRange(absl::string_view output_name, int* start,
                                   int* stop) const {
  const auto result = output_name_map_.find(output_name);
  if (result == output_name_map_.end()) {
    return errors::InvalidArgument("Unknown output name: ", output_name);
  } else {
    *start = result->second.first;
    *stop = result->second.second;
    return absl::OkStatus();
  }
}

string OpKernel::ShapeTraceString(const OpKernelContext& ctx) const {
  int num_inputs = ctx.num_inputs();
  if (num_inputs == 0) return "";
  std::vector<string> tensor_shapes;
  tensor_shapes.reserve(num_inputs);
  for (int i = 0; i < num_inputs; i++) {
    if (!ctx.has_input(i)) {
      tensor_shapes.emplace_back();  // Placeholder
      continue;
    }
    DataType input_dtype = ctx.input_dtype(i);
    if (input_dtype == DataType::DT_RESOURCE ||
        input_dtype == DataType::DT_VARIANT || IsRefType(input_dtype)) {
      tensor_shapes.emplace_back();  // Placeholder
      continue;
    }
    tensor_shapes.emplace_back(strings::StrCat(
        DataTypeString(input_dtype), ctx.input(i).shape().DebugString()));
  }
  return strings::StrCat("(", absl::StrJoin(tensor_shapes, ";"), ")");
}

string OpKernel::TraceString(const OpKernelContext& ctx, bool verbose) const {
  string trace_string =
      tsl::profiler::TraceMeOp(name_view(), type_string_view());
  if (verbose) {
    string shape = ShapeTraceString(ctx);
    if (!shape.empty()) {
      trace_string = tsl::profiler::TraceMeEncode(std::move(trace_string),
                                                  {{"shape", shape}});
    }
  }
  return trace_string;
}

void AsyncOpKernel::Compute(OpKernelContext* context) {
  Notification n;
  ComputeAsync(context, [&n]() { n.Notify(); });
  n.WaitForNotification();
}

// OpKernelConstruction ------------------------------------------------------

OpKernelConstruction::OpKernelConstruction(
    DeviceType device_type, DeviceBase* device, Allocator* allocator,
    FunctionLibraryRuntime* flib, ResourceMgr* resource_mgr,
    const std::shared_ptr<const NodeProperties>& props,
    const MemoryTypeSlice& input_memory_types,
    const MemoryTypeSlice& output_memory_types, int graph_def_version,
    absl::Status* status)
    : device_type_(std::move(device_type)),
      device_(device),
      allocator_(allocator),
      flib_(flib),
      resource_mgr_(resource_mgr),
      props_(props),
      input_memory_types_(input_memory_types),
      output_memory_types_(output_memory_types),
      graph_def_version_(graph_def_version),
      status_(status) {}

bool OpKernelConstruction::HasAttr(absl::string_view attr_name) const {
  return HasNodeAttr(def(), attr_name);
}

void OpKernelConstruction::SetStatus(const absl::Status& status) {
  status_->Update(status);
}

absl::Status OpKernelConstruction::MatchSignature(
    const DataTypeSlice expected_inputs, const DataTypeSlice expected_outputs) {
  return MatchSignatureHelper(expected_inputs, expected_outputs,
                              props_->input_types, props_->output_types);
}

absl::Status OpKernelConstruction::allocate_temp(DataType type,
                                                 const TensorShape& shape,
                                                 Tensor* out_temp) {
  AllocationAttributes attr;
  attr.allocation_will_be_logged = true;
  Tensor new_temp(allocator_, type, shape, attr);

  if (!new_temp.IsInitialized()) {
    return errors::ResourceExhausted(
        "OOM when allocating temporary tensor with shape", shape.DebugString());
  }
  if (LogMemory::IsEnabled()) {
    LogMemory::RecordTensorAllocation(
        def().name(), LogMemory::OP_KERNEL_CONSTRUCTION_STEP_ID, new_temp);
  }
  *out_temp = new_temp;
  return absl::OkStatus();
}

absl::Status OpKernelConstruction::allocate_temp(
    DataType type, const TensorShape& shape, Tensor* out_temp,
    AllocatorAttributes allocator_attr) {
  if (allocator_attr.scope_id != 0) {
    return errors::InvalidArgument(
        "ScopedAllocator cannot be used via OpKernelConstruction.");
  }
  Allocator* a = device_->GetAllocator(allocator_attr);
  AllocationAttributes attr;
  attr.allocation_will_be_logged = true;
  Tensor new_temp(a, type, shape, attr);

  if (!new_temp.IsInitialized()) {
    return errors::ResourceExhausted(
        "OOM when allocating temporary tensor with shape", shape.DebugString());
  }
  if (LogMemory::IsEnabled()) {
    LogMemory::RecordTensorAllocation(
        def().name(), LogMemory::OP_KERNEL_CONSTRUCTION_STEP_ID, new_temp);
  }
  *out_temp = new_temp;
  return absl::OkStatus();
}

// OpKernelContext -----------------------------------------------------------

const int OpKernelContext::Params::kNeverForward;
const int OpKernelContext::Params::kNoReservation;

OpKernelContext::OpKernelContext(Params* params)
    : OpKernelContext(
          params, static_cast<int>(params->op_kernel->output_types().size())) {}

OpKernelContext::OpKernelContext(Params* params, int num_outputs)
    : params_(params), outputs_(num_outputs) {
  if (params_->track_allocations) {
    tracking_state_ = std::make_unique<TrackingState>();
  }

  params_->ensure_eigen_gpu_device();
  if (params_->eigen_gpu_device != nullptr) {
    Allocator* eigen_gpu_allocator = get_allocator(AllocatorAttributes());
    absl::Status s = params_->device->ReinitializeGpuDevice(
        this, params_->eigen_gpu_device, params_->op_device_context,
        eigen_gpu_allocator);
    if (!s.ok()) {
      SetStatus(s);
    }
  }
}

OpKernelContext::~OpKernelContext() {
  for (TensorValue& value : outputs_) {
    if (!value.is_ref()) {
      delete value.tensor;
    }
  }
  if (params_->track_allocations &&
      !tracking_state_->wrapped_allocators.empty()) {
    LOG(WARNING) << "OpKernelContext is tracking allocations but they are not "
                 << "being consumed by the StepStatsCollector.";
    for (auto& wrapped_allocator : tracking_state_->wrapped_allocators) {
      wrapped_allocator.second->GetRecordsAndUnRef();
    }
  }
}

Allocator* OpKernelContext::get_allocator(AllocatorAttributes attr) {
  Allocator* allocator = nullptr;
  if (TF_PREDICT_FALSE(attr.scope_id > 0)) {
    allocator = params_->device->GetScopedAllocator(attr, step_id());
    CHECK(allocator);
  } else {
    allocator = params_->device->GetAllocator(attr);
  }
  if (TF_PREDICT_FALSE(track_allocations())) {
    DCHECK(tracking_state_);
    mutex_lock lock(tracking_state_->mu);
    for (const auto& wrapped : tracking_state_->wrapped_allocators) {
      if (wrapped.first == allocator) {
        return wrapped.second;
      }
    }
    TrackingAllocator* wrapped_allocator =
        new TrackingAllocator(allocator, params_->track_allocations);
    tracking_state_->wrapped_allocators.push_back(
        std::make_pair(allocator, wrapped_allocator));
    return wrapped_allocator;
  } else {
    return allocator;
  }
}

void OpKernelContext::SetStatus(const absl::Status& status) {
  status_.Update(status);
}

absl::Status OpKernelContext::input(absl::string_view name,
                                    const Tensor** tensor) {
  int index;
  TF_RETURN_IF_ERROR(get_input_index(name, &index));
  if (input_is_ref(index)) {
    return errors::InvalidArgument("OpKernel used ref input name '", name,
                                   "' when non-ref input was expected");
  }
  *tensor = params_->inputs[index].tensor;
  return absl::OkStatus();
}

absl::Status OpKernelContext::input_dtype(absl::string_view name,
                                          DataType* dtype) const {
  int index;
  TF_RETURN_IF_ERROR(get_input_index(name, &index));
  const TensorValue& value(params_->inputs[index]);
  *dtype = value.dtype();
  return absl::OkStatus();
}

absl::Status OpKernelContext::input_ref_mutex(absl::string_view name,
                                              mutex** out_mutex) {
  int index;
  TF_RETURN_IF_ERROR(get_input_index(name, &index));
  *out_mutex = input_ref_mutex(index);
  return absl::OkStatus();
}

absl::StatusOr<const Tensor*> OpKernelContext::get_input(int index) const {
  if (index < 0 || index >= num_inputs() || input_is_ref(index)) {
    return absl::InvalidArgumentError(
        absl::StrCat("Given index was ", index,
                     ", but index of input must be greater than "
                     "0, less than the number of inputs (",
                     num_inputs(), "), and not a ref."));
  }
  return params_->inputs[index].tensor;
}

const Tensor& OpKernelContext::input(int index) const {
  CHECK_GE(index, 0);
  CHECK_LT(index, num_inputs()) << " name: " << op_kernel().name();
  CHECK(!input_is_ref(index));
  const Tensor& tensor = *params_->inputs[index].tensor;
  return tensor;
}

Tensor OpKernelContext::mutable_input(int index, bool lock_held) {
  CHECK_GE(index, 0);
  CHECK_LT(index, num_inputs());
  CHECK(input_is_ref(index));
  // return a copy of the Ref acquired while holding the mutex
  if (lock_held) {
    Tensor& tensor = *params_->inputs[index].tensor;
    return tensor;
  } else {
    tf_shared_lock l(*input_ref_mutex(index));
    Tensor& tensor = *params_->inputs[index].tensor;
    return tensor;
  }
}

void OpKernelContext::replace_ref_input(int index, const Tensor& tensor,
                                        bool lock_held) {
  CHECK_GE(index, 0);
  CHECK_LT(index, num_inputs());
  CHECK(input_is_ref(index));
  // should only modify the tensor while holding the mutex
  if (lock_held) {
    *params_->inputs[index].tensor = tensor;
  } else {
    mutex_lock l(*input_ref_mutex(index));
    *params_->inputs[index].tensor = tensor;
  }
}

void OpKernelContext::forward_ref_input_to_ref_output(int input_index,
                                                      int output_index) {
  CHECK_GE(input_index, 0);
  CHECK_LT(input_index, num_inputs());
  CHECK(input_is_ref(input_index));
  set_output_ref(output_index, params_->inputs[input_index].mutex_if_ref,
                 params_->inputs[input_index].tensor);
}

bool OpKernelContext::forward_input_to_output_with_shape(
    int input_index, int output_index, const TensorShape& output_shape,
    Tensor** output) {
  const auto output_attr = params_->output_attr_array == nullptr
                               ? AllocatorAttributes()
                               : output_alloc_attr(output_index);
  std::unique_ptr<Tensor> new_tensor = forward_input(
      input_index, output_index, expected_output_dtype(output_index),
      output_shape, output_memory_type(output_index), output_attr);
  if (new_tensor != nullptr) {
    // Transfer ownership to the output slot in OpKernelContext.
    outputs_[output_index] = TensorValue(new_tensor.release());
    *output = outputs_[output_index].tensor;
    return true;
  } else {
    return false;
  }
}

absl::Status OpKernelContext::forward_input_to_output_with_shape(
    absl::string_view input_name, absl::string_view output_name,
    const TensorShape& output_shape, Tensor** output) {
  int input_index, output_index;
  TF_RETURN_IF_ERROR(get_input_index(input_name, &input_index));
  TF_RETURN_IF_ERROR(get_output_index(output_name, &output_index));
  if (!forward_input_to_output_with_shape(input_index, output_index,
                                          output_shape, output)) {
    return errors::FailedPrecondition("OpKernel could not forward input '",
                                      input_name, "' to output '", output_name);
  }
  return absl::OkStatus();
}

std::unique_ptr<Tensor> OpKernelContext::forward_input(
    int input_index, int output_index, DataType output_dtype,
    const TensorShape& output_shape, MemoryType output_memory_type,
    const AllocatorAttributes& output_attr) {
  CHECK_GE(input_index, 0);
  CHECK_LT(input_index, num_inputs());
  const TensorValue& input = params_->inputs[input_index];
  // Check whether at graph construction time this output was marked
  // either for no forwarding or with a reservation for this input.
  // If it's reserved for this input we'll skip the refcount and
  // AllocatorAttribute checks.
  // TODO(tucker): Maybe we should skip all of the checks?
  bool never_forward =
      (params_->forward_from_array != nullptr && output_index >= 0 &&
       params_->forward_from_array[output_index] == Params::kNeverForward);
  if (never_forward) return nullptr;
  bool forward_expected =
      (params_->forward_from_array != nullptr && output_index >= 0 &&
       params_->forward_from_array[output_index] == input_index);
  if (!forward_expected && params_->forward_from_array != nullptr) {
    // Check for possibly conflicting forward.
    for (int i = 0; i < num_outputs(); ++i) {
      if (params_->forward_from_array[i] == input_index) {
        // This input is reserved for output i.
        return nullptr;
      }
    }
  }
  // Check that input tensor exists and is not a ref.
  if (input.tensor == nullptr || input.is_ref()) {
    CHECK(!forward_expected);
    return nullptr;
  }
  // Check that input type matches.
  if (input_dtype(input_index) != output_dtype) {
    CHECK(!forward_expected);
    return nullptr;
  }
  // Check that the input and output sizes are compatible.
  if (input.tensor->shape().num_elements() != output_shape.num_elements()) {
    CHECK(!forward_expected);
    return nullptr;
  }
  // Check that input and output memory types match, i.e.
  // that they either both live in host or both live in device memory.
  if (input_memory_type(input_index) != output_memory_type) {
    CHECK(!forward_expected);
    return nullptr;
  }
  if (!forward_expected) {
    if (!input->RefCountIsOne()) {
      return nullptr;
    }
    // Check that output allocator attributes are not more restrictive than
    // input allocator attributes.
    const auto input_attr = params_->input_alloc_attrs.empty()
                                ? AllocatorAttributes()
                                : input_alloc_attr(input_index);
    if (!output_attr.IsEqualOrLessRestrictiveThan(input_attr)) {
      return nullptr;
    }
  }

  auto output_tensor = std::make_unique<Tensor>();
  CHECK(output_tensor->CopyFrom(*input.tensor, output_shape));
  return output_tensor;
}

absl::Status OpKernelContext::forward_input_or_allocate_temp(
    absl::Span<const int> candidate_input_indices, DataType type,
    const TensorShape& shape, const AllocatorAttributes& allocator_attr,
    Tensor* out_temp) {
  for (int input_index : candidate_input_indices) {
    std::unique_ptr<Tensor> new_tensor =
        forward_input(input_index, Params::kNoReservation /*output_index*/,
                      type, shape, DEVICE_MEMORY, allocator_attr);
    if (new_tensor != nullptr) {
      *out_temp = std::move(*new_tensor);
      return absl::OkStatus();
    }
  }
  return allocate_temp(type, shape, out_temp, allocator_attr);
}

absl::Status OpKernelContext::forward_input_or_allocate_output(
    absl::Span<const int> candidate_input_indices, int output_index,
    const TensorShape& output_shape, Tensor** output, int* forwarded_input) {
  for (int input_index : candidate_input_indices) {
    if (forward_input_to_output_with_shape(input_index, output_index,
                                           output_shape, output)) {
      if (forwarded_input != nullptr) {
        *forwarded_input = input_index;
      }
      return absl::OkStatus();
    }
  }
  if (forwarded_input != nullptr) {
    *forwarded_input = -1;
  }
  return allocate_output(output_index, output_shape, output);
}

absl::Status OpKernelContext::forward_input_or_allocate_output(
    absl::Span<const absl::string_view> candidate_input_names,
    absl::string_view output_name, const TensorShape& output_shape,
    Tensor** output) {
  for (const absl::string_view& input_name : candidate_input_names) {
    if (forward_input_to_output_with_shape(input_name, output_name,
                                           output_shape, output)
            .ok()) {
      return absl::OkStatus();
    }
  }
  return allocate_output(output_name, output_shape, output);
}

void OpKernelContext::delete_ref_input(int index, bool lock_held) {
  CHECK_GE(index, 0);
  CHECK_LT(index, num_inputs());
  CHECK(input_is_ref(index));
  // should only modify the tensor while holding the mutex
  if (lock_held) {
    delete params_->inputs[index].tensor;
  } else {
    mutex_lock l(*input_ref_mutex(index));
    delete params_->inputs[index].tensor;
  }
}

absl::Status OpKernelContext::mutable_input(absl::string_view name,
                                            Tensor* tensor, bool lock_held) {
  int index;
  TF_RETURN_IF_ERROR(get_input_index(name, &index));
  if (!input_is_ref(index)) {
    return errors::InvalidArgument("OpKernel used non-ref input name '", name,
                                   "' when ref input was expected");
  }
  // return a copy of the Ref acquired while holding the mutex
  if (lock_held) {
    *tensor = *params_->inputs[index].tensor;
  } else {
    tf_shared_lock l(*input_ref_mutex(index));
    *tensor = *params_->inputs[index].tensor;
  }
  return absl::OkStatus();
}

absl::Status OpKernelContext::replace_ref_input(absl::string_view name,
                                                const Tensor& tensor,
                                                bool lock_held) {
  int index;
  TF_RETURN_IF_ERROR(get_input_index(name, &index));
  if (!input_is_ref(index)) {
    return errors::InvalidArgument("OpKernel used immutable input name '", name,
                                   "' when ref input was expected");
  }
  replace_ref_input(index, tensor, lock_held);
  return absl::OkStatus();
}

absl::Status OpKernelContext::input_list(absl::string_view name,
                                         OpInputList* list) {
  int start, stop;
  TF_RETURN_IF_ERROR(params_->op_kernel->InputRange(name, &start, &stop));
  *list = OpInputList(this, start, stop);
  return absl::OkStatus();
}

absl::Status OpKernelContext::mutable_input_list(absl::string_view name,
                                                 OpMutableInputList* list) {
  int start, stop;
  TF_RETURN_IF_ERROR(params_->op_kernel->InputRange(name, &start, &stop));
  *list = OpMutableInputList(this, start, stop);
  return absl::OkStatus();
}

absl::Status OpKernelContext::output_list(absl::string_view name,
                                          OpOutputList* list) {
  int start, stop;
  TF_RETURN_IF_ERROR(params_->op_kernel->OutputRange(name, &start, &stop));
  *list = OpOutputList(this, start, stop);
  return absl::OkStatus();
}

void OpKernelContext::maybe_initialize_scope_id_set() {
  if (allocated_scope_ids_ == nullptr) {
    allocated_scope_ids_ = std::make_unique<std::unordered_set<int32>>();
  }
}

absl::Status OpKernelContext::allocate_output(int index,
                                              const TensorShape& shape,
                                              Tensor** tensor) {
  if (index < 0) {
    return errors::Internal("allocate_output with bad index=", index,
                            " kernel=", params_->op_kernel->name());
  }
  if (index >= num_outputs()) {
    return errors::Internal("allocate_output with bad index=", index,
                            " num_outputs=", num_outputs(),
                            " kernel=", params_->op_kernel->name());
  }
  bool forward_expected =
      (params_->forward_from_array != nullptr && index >= 0 &&
       params_->forward_from_array[index] >= 0);
  if (forward_expected) {
    return errors::Internal(
        "Explicit allocate_output call where input forwarding required.  Try "
        "turning off the ScopedAllocator optimizer.");
  }
  AllocatorAttributes attr = output_alloc_attr(index);
  return allocate_output(index, shape, tensor, attr);
}

absl::Status OpKernelContext::allocate_output(absl::string_view name,
                                              const TensorShape& shape,
                                              Tensor** tensor) {
  int start, stop;
  TF_RETURN_IF_ERROR(params_->op_kernel->OutputRange(name, &start, &stop));
  if (stop != start + 1) {
    return errors::InvalidArgument("OpKernel used list-valued output name '",
                                   name,
                                   "' when single-valued output was "
                                   "expected");
  }
  return allocate_output(start, shape, tensor);
}

absl::Status OpKernelContext::allocate_output(absl::string_view name,
                                              const TensorShape& shape,
                                              Tensor** tensor,
                                              AllocatorAttributes attr) {
  int start, stop;
  TF_RETURN_IF_ERROR(params_->op_kernel->OutputRange(name, &start, &stop));
  if (stop != start + 1) {
    return errors::InvalidArgument("OpKernel used list-valued output name '",
                                   name,
                                   "' when single-valued output was "
                                   "expected");
  }
  return allocate_output(start, shape, tensor, attr);
}

absl::Status OpKernelContext::allocate_tensor(
    DataType type, const TensorShape& shape, Tensor* out_tensor,
    AllocatorAttributes attr, const AllocationAttributes& allocation_attr) {
  Allocator* a = get_allocator(attr);
  Tensor new_tensor(
      a, type, shape,
      AllocationAttributes(
          /*retry_on_failure=*/allocation_attr.retry_on_failure,
          /*allocation_will_be_logged=*/true, allocation_attr.freed_by_func));

  if (!new_tensor.IsInitialized()) {
    return errors::ResourceExhausted(
        "OOM when allocating tensor with shape", shape.DebugString(),
        " and type ", DataTypeString(type), " on ", params_->device->name(),
        " by allocator ", a->Name());
  }
  if (params_->log_memory) {
    LogMemory::RecordTensorAllocation(params_->op_kernel->name(),
                                      params_->step_id, new_tensor);
  }
  *out_tensor = std::move(new_tensor);
  return absl::OkStatus();
}

absl::Status OpKernelContext::allocate_output(int index,
                                              const TensorShape& shape,
                                              Tensor** output,
                                              AllocatorAttributes attr) {
  if (index < 0) {
    return errors::Internal("allocate_output with bad index=", index,
                            " kernel=", params_->op_kernel->name());
  }
  if (index >= num_outputs()) {
    return errors::Internal("allocate_output with bad index=", index,
                            " num_outputs=", outputs_.size(),
                            " kernel=", params_->op_kernel->name());
  }
  const DataType type = params_->op_kernel->output_type(index);
  if (IsRefType(type)) {
    return errors::Internal("allocate_output with ref type. index=", index,
                            " type=", type,
                            " kernel=", params_->op_kernel->name());
  }
  if (mutable_output(index) != nullptr) {
    return errors::Internal("allocate_output on same index multiple times.",
                            " index = ", index,
                            " mutable_output(index) = ", mutable_output(index),
                            " kernel=", params_->op_kernel->name());
  }
  if (attr.scope_id > 0) {
    maybe_initialize_scope_id_set();
    if (!allocated_scope_ids_->insert(attr.scope_id).second) {
      return errors::Internal(
          "OpKernel ", params_->op_kernel->name(),
          " called allocate_output at index ", index, " with scope_id ",
          attr.scope_id,
          " more than once.  Try turning off the ScopedAllocator optimizer.");
    }
  }
  tsl::profiler::ScopedMemoryDebugAnnotation op_annotation(
      op_kernel().name_view().data(), step_id(), "output", type,
      [&shape]() { return shape.DebugString(); });
  auto output_tensor = std::make_unique<Tensor>();
  absl::Status s = allocate_tensor(type, shape, output_tensor.get(), attr);
  if (s.ok()) {
    outputs_[index] = TensorValue(output_tensor.release());
    *output = outputs_[index].tensor;
  }
  return s;
}

absl::Status OpKernelContext::allocate_temp(
    DataType type, const TensorShape& shape, Tensor* out_temp,
    AllocatorAttributes allocator_attr,
    const AllocationAttributes& allocation_attr) {
  if (allocator_attr.scope_id > 0) {
    // We do not allow ScopedAllocator calls from allocate_temp.
    // Here we clear the scope_id and return a temporary buffer.
    // This is because it is legal for a kernel to call allocate_temp
    // and then set_output with the temp tensor.
    //
    // We achieve memory correctness by forcing an allocation in set_output and
    // copying over the tensor from the temp buffer.  Kernels which would like
    // to avoid this performance penalty should switch to calling
    // allocate_output.
    VLOG(2) << "Warning: OpKernel " << params_->op_kernel->name()
            << " called allocate_temp with scope_id " << allocator_attr.scope_id
            << ".  Switch to allocate_output to avoid performance penalty.";
    allocator_attr.scope_id = -1;
  }
  tsl::profiler::ScopedMemoryDebugAnnotation op_annotation(
      op_kernel().name_view().data(), step_id(), "temp", type,
      [&shape]() { return shape.DebugString(); });
  absl::Status s =
      allocate_tensor(type, shape, out_temp, allocator_attr, allocation_attr);
  if (track_allocations() && s.ok() && out_temp->TotalBytes() > 0) {
    Allocator* a = get_allocator(allocator_attr);
    if (a->TracksAllocationSizes()) {
      int64_t alloc_size = a->AllocatedSize(out_temp->tensor_data().data());
      record_temp_memory_allocation(alloc_size, *out_temp);
    }
  } else if (record_memory_consumption_) {
    DCHECK(tracking_state_);
    mutex_lock l(tracking_state_->stats_mu);
    tracking_state_->temp_memory_allocated += out_temp->TotalBytes();
  }
  return s;
}

absl::Status OpKernelContext::allocate_temp(
    DataType type, const TensorShape& shape, Tensor* out_temp,
    AllocatorAttributes allocator_attr) {
  return allocate_temp(type, shape, out_temp, allocator_attr,
                       AllocationAttributes());
}

absl::Status OpKernelContext::allocate_temp(DataType type,
                                            const TensorShape& shape,
                                            Tensor* out_temp) {
  return allocate_temp(type, shape, out_temp, AllocatorAttributes());
}

absl::Status OpKernelContext::get_input_index(absl::string_view name,
                                              int* out_index) const {
  int start, stop;
  TF_RETURN_IF_ERROR(params_->op_kernel->InputRange(name, &start, &stop));
  if (stop != start + 1) {
    return errors::InvalidArgument("OpKernel used list-valued input name '",
                                   name,
                                   "' when single-valued input was "
                                   "expected");
  }
  *out_index = start;
  return absl::OkStatus();
}

absl::Status OpKernelContext::get_output_index(absl::string_view name,
                                               int* out_index) const {
  int start, stop;
  TF_RETURN_IF_ERROR(params_->op_kernel->OutputRange(name, &start, &stop));
  if (stop != start + 1) {
    return errors::InvalidArgument("OpKernel used list-valued output name '",
                                   name,
                                   "' when single-valued output was "
                                   "expected");
  }
  *out_index = start;
  return absl::OkStatus();
}

absl::Status OpKernelContext::set_output(absl::string_view name,
                                         const Tensor& tensor) {
  int index;
  TF_RETURN_IF_ERROR(get_output_index(name, &index));
  set_output(index, tensor);
  return absl::OkStatus();
}

absl::Status OpKernelContext::set_output(absl::string_view name,
                                         Tensor&& tensor) {
  int index;
  TF_RETURN_IF_ERROR(get_output_index(name, &index));
  set_output(index, std::move(tensor));
  return absl::OkStatus();
}

bool OpKernelContext::maybe_set_output_by_allocate_and_copy(
    int index, const Tensor& tensor) {
  bool allocate_and_copy = false;
  const bool never_forward =
      (params_->forward_from_array != nullptr &&
       params_->forward_from_array[index] == Params::kNeverForward);
  if (TF_PREDICT_FALSE(never_forward)) {
    maybe_initialize_scope_id_set();
    if (allocated_scope_ids_->find(output_alloc_attr(index).scope_id) ==
        allocated_scope_ids_->end()) {
      allocate_and_copy = true;
    } else {
      // The output at `index` must have been previously allocated via a call to
      // `allocate_output(index, ...)`.  That call would ensure that we return
      // the correct slice of the ScopedAllocated buffer, so we do not
      // re-allocate and copy here.
      LOG(WARNING)
          << "OpKernel " << params_->op_kernel->name()
          << " called both allocate_output and set_output with scope_id "
          << output_alloc_attr(index).scope_id;
    }
  }

  if (TF_PREDICT_FALSE(allocate_and_copy)) {
    // This output was marked to not be forwarded either during graph
    // construction or grappler passes.  Force an allocation and copy input to
    // output.
    VLOG(1) << "OpKernelContext set_output index " << index << " tensor "
            << tensor.DebugString() << " never_forward " << never_forward
            << " params_->forward_from_array[index] "
            << params_->forward_from_array[index] << " alloc_attr.scope_id "
            << output_alloc_attr(index).scope_id;
    tsl::profiler::ScopedMemoryDebugAnnotation op_annotation(
        op_kernel().name_view().data(), step_id(), "output", tensor.dtype(),
        [&tensor]() { return tensor.shape().DebugString(); });
    auto new_tensor = std::make_unique<Tensor>();
    absl::Status s =
        allocate_tensor(tensor.dtype(), tensor.shape(), new_tensor.get(),
                        output_alloc_attr(index));
    TF_CHECK_OK(s);
    device()->CopyTensorInSameDevice(&tensor, new_tensor.get(),
                                     op_device_context(),
                                     [](const absl::Status&) {});
    outputs_[index] = TensorValue(new_tensor.release());
  }
  return allocate_and_copy;
}

void OpKernelContext::maybe_track_allocations_for_set_output(
    const Tensor& tensor) {
  if (TF_PREDICT_FALSE(track_allocations()) && tensor.TotalBytes() > 0) {
    DCHECK(tracking_state_);
    mutex_lock l(tracking_state_->stats_mu);
    const auto it = std::find_if(
        tracking_state_->temp_tensor_buffer_and_size.begin(),
        tracking_state_->temp_tensor_buffer_and_size.end(),
        [&tensor](const std::pair<const void*, int64>& e) {
          return e.first ==
                 static_cast<const void*>(tensor.tensor_data().data());
        });
    if (it != tracking_state_->temp_tensor_buffer_and_size.end()) {
      tracking_state_->temp_memory_allocated -= it->second;
      tracking_state_->temp_tensor_buffer_and_size.erase(it);
    }
  }
}

void OpKernelContext::set_output(int index, const Tensor& tensor) {
  CHECK_GE(index, 0);
  CHECK_LT(index, outputs_.size());
  const DataType type = params_->op_kernel->output_type(index);
  CHECK(!IsRefType(type));
  CHECK_EQ(outputs_[index].tensor, nullptr);
  if (TF_PREDICT_TRUE(!maybe_set_output_by_allocate_and_copy(index, tensor))) {
    // Input can be forwarded to output; incref on `tensor` and set output at
    // `index` to this tensor.
    outputs_[index] = TensorValue(new Tensor(tensor));
    maybe_track_allocations_for_set_output(*outputs_[index].tensor);
  }
}

void OpKernelContext::set_output(int index, Tensor&& tensor) {
  CHECK_GE(index, 0);
  CHECK_LT(index, outputs_.size());
  const DataType type = params_->op_kernel->output_type(index);
  CHECK(!IsRefType(type));
  CHECK_EQ(outputs_[index].tensor, nullptr);
  if (TF_PREDICT_TRUE(!maybe_set_output_by_allocate_and_copy(index, tensor))) {
    // Input can be forwarded to output; set output at `index` to this tensor.
    outputs_[index] = TensorValue(new Tensor(std::move(tensor)));
    maybe_track_allocations_for_set_output(*outputs_[index].tensor);
  }
}

void OpKernelContext::set_output_ref(int index, mutex* mu,
                                     Tensor* tensor_for_ref) {
  CHECK_GE(index, 0);
  CHECK_LT(index, outputs_.size());
  CHECK(IsRefType(params_->op_kernel->output_type(index)));
  outputs_[index] = TensorValue(mu, tensor_for_ref);
}

absl::Status OpKernelContext::set_output_ref(absl::string_view name, mutex* mu,
                                             Tensor* tensor_for_ref) {
  int index;
  TF_RETURN_IF_ERROR(get_output_index(name, &index));
  set_output_ref(index, mu, tensor_for_ref);
  return absl::OkStatus();
}

absl::Status OpKernelContext::mutable_output(absl::string_view name,
                                             Tensor** tensor) {
  int index;
  TF_RETURN_IF_ERROR(get_output_index(name, &index));
  *tensor = mutable_output(index);
  return absl::OkStatus();
}

bool OpKernelContext::ValidateInputsAreSameShape(OpKernel* op) {
  const auto& inputs = params_->inputs;
  for (size_t i = 1; i < inputs.size(); ++i) {
    if (!inputs[0]->IsSameSize(*(inputs[i].tensor))) {
      SetStatus(errors::InvalidArgument(
          "Inputs to operation ", op->name(), " of type ", op->type_string(),
          " must have the same size and shape.  Input 0: ",
          inputs[0]->shape().DebugString(), " != input ", i, ": ",
          inputs[i]->shape().DebugString()));
      return false;
    }
  }
  return true;
}

absl::Status OpKernelContext::MatchSignature(
    const DataTypeSlice expected_inputs, const DataTypeSlice expected_outputs) {
  DataTypeVector inputs;
  for (const TensorValue& t : params_->inputs) {
    inputs.push_back(t.dtype());
  }
  DataTypeVector outputs = params_->op_kernel->output_types();
  return MatchSignatureHelper(expected_inputs, expected_outputs, inputs,
                              outputs);
}

void OpKernelContext::record_temp_memory_allocation(int64_t size,
                                                    const Tensor& t) {
  if (tracking_state_) {
    mutex_lock l(tracking_state_->stats_mu);
    tracking_state_->temp_memory_allocated += size;
    tracking_state_->temp_tensor_buffer_and_size.emplace_back(
        static_cast<const void*>(t.tensor_data().data()), size);
  }
}

int64_t OpKernelContext::temp_memory_allocated() const {
  if (tracking_state_) {
    mutex_lock l(tracking_state_->stats_mu);
    return tracking_state_->temp_memory_allocated;
  } else {
    return 0;
  }
}

void OpKernelContext::record_persistent_memory_allocation(int64_t size,
                                                          int64_t alloc_id) {
  if (tracking_state_) {
    mutex_lock l(tracking_state_->stats_mu);
    tracking_state_->persistent_memory_allocated += size;
    if (alloc_id >= 0) {
      tracking_state_->persistent_alloc_ids.push_back(alloc_id);
    }
  }
}

int64_t OpKernelContext::persistent_memory_allocated() const {
  if (tracking_state_) {
    mutex_lock l(tracking_state_->stats_mu);
    return tracking_state_->persistent_memory_allocated;
  } else {
    return 0;
  }
}

std::vector<int64_t> OpKernelContext::persistent_alloc_ids() const {
  if (tracking_state_) {
    mutex_lock l(tracking_state_->stats_mu);
    return std::vector<int64_t>(tracking_state_->persistent_alloc_ids.begin(),
                                tracking_state_->persistent_alloc_ids.end());
  } else {
    return std::vector<int64_t>();
  }
}

void OpKernelContext::clear_recorded_memory() {
  if (tracking_state_) {
    mutex_lock l(tracking_state_->stats_mu);
    tracking_state_->temp_memory_allocated = 0;
    tracking_state_->persistent_memory_allocated = 0;
    tracking_state_->temp_tensor_buffer_and_size.clear();
    tracking_state_->persistent_alloc_ids.clear();
  }
}

void OpKernelContext::set_record_memory_consumption(bool v) {
  record_memory_consumption_ = v;
  if (v && !tracking_state_) {
    tracking_state_ = std::make_unique<TrackingState>();
  }
}

const string& OpKernelContext::executor_type() const {
  if (params_->executor_type) {
    return *params_->executor_type;
  } else {
    static const string& kEmptyString = *new string("");
    return kEmptyString;
  }
}

// OpKernel registration ------------------------------------------------------

struct KernelRegistration {
  KernelRegistration(const KernelDef& d, absl::string_view c,
                     std::unique_ptr<kernel_factory::OpKernelFactory> f)
      : def(d), kernel_class_name(c), factory(std::move(f)) {}

  const KernelDef def;
  const string kernel_class_name;
  std::unique_ptr<kernel_factory::OpKernelFactory> factory;
};

// This maps from 'op_type' + DeviceType to the set of KernelDefs and
// factory functions for instantiating the OpKernel that matches the
// KernelDef.
struct KernelRegistry {
  mutex mu;
  std::unordered_multimap<string, KernelRegistration> registry
      TF_GUARDED_BY(mu);
};

#if defined(_WIN32)
static const char kKernelLibPattern[] = "libtfkernel*.dll";
#elif defined(__APPLE__)
static const char kKernelLibPattern[] = "libtfkernel*.dylib";
#else
static const char kKernelLibPattern[] = "libtfkernel*.so";
#endif

#define FEATURE(x) \
  { x, #x }

// Returns Status::OK if the dynamic library at the given path is safe to
// load with some level of confidence.
static absl::Status IsProbablySafeToLoad(const string& path) {
  // A map of platform string to required CPU feature.
  using port::CPUFeature;
  static const auto* feature_map =
      new std::map<string, std::pair<CPUFeature, string>>{
          {"__AVX512VL__=1", FEATURE(CPUFeature::AVX512VL)},
      };

  std::vector<std::string> platform_strings;
  int result = GetPlatformStrings(path, &platform_strings);
  if (result) {
    return absl::Status(absl::StatusCode::kUnknown, strerror(result));
  }
  if (platform_strings.empty()) {
    return absl::Status(absl::StatusCode::kFailedPrecondition,
                        "Didn't find any platform strings");
  }
  std::vector<std::string> missing_features;
  for (const auto& platform_string : platform_strings) {
    const auto& entry = feature_map->find(platform_string);
    if (entry != feature_map->end() &&
        !port::TestCPUFeature(entry->second.first)) {
      missing_features.emplace_back(entry->second.second);
    }
  }
  if (!missing_features.empty()) {
    string errmsg = "Missing CPU features: ";
    errmsg.append(absl::StrJoin(missing_features, ", "));
    return errors::FailedPrecondition(errmsg);
  }
  return absl::OkStatus();
}

void LoadDynamicKernelsInternal() {
  Env* env = Env::Default();

  // Override to allow loading unsafe packages for development.
  // DO NOT USE UNLESS YOU KNOW WHAT ABI ISSUES YOU CAN ENCOUNTER.
  char* _abi_check_env_var = getenv("TF_REALLY_LOAD_UNSAFE_PACKAGES");
  bool override_abi_check = false;
  if (_abi_check_env_var != nullptr) {
    override_abi_check = strcmp(_abi_check_env_var, "1") == 0;
  }

  string bazel_kernel_dir =
      io::JoinPath(env->GetRunfilesDir(), "tensorflow", "core", "kernels");
  std::vector<string> files;
  absl::Status s_kernel_dir = env->GetChildren(bazel_kernel_dir, &files);
  if (s_kernel_dir.ok()) {
    string dll_spec = io::JoinPath(bazel_kernel_dir, kKernelLibPattern);
    for (const auto& file : files) {
      string fullpath = io::JoinPath(bazel_kernel_dir, file);
      if (env->MatchPath(fullpath, dll_spec)) {
        absl::Status s = IsProbablySafeToLoad(fullpath);
        if (!s.ok() && override_abi_check) {
          LOG(WARNING) << "Loading UNSAFE library " << fullpath
                       << " because ABI check override is set: " << s.message();
        }
        if (s.ok() || override_abi_check) {
          // TODO(gunan): Store the handles to the opened files.
          void* unused_filehandle;
          TF_CHECK_OK(
              env->LoadDynamicLibrary(fullpath.c_str(), &unused_filehandle));
        } else {
          LOG(WARNING) << "Not loading plugin library " << fullpath << ": "
                       << s.message();
        }
      }
    }
  }
}

// Mechanism for loading existing kernel libraries.
void LoadDynamicKernels() {
  // TODO(gunan): As more features are available, add intelligent kernel
  // selection, and dropping unsuitable kernel logic here.
  static absl::once_flag dll_loader_flag;
  absl::call_once(dll_loader_flag, LoadDynamicKernelsInternal);
}

static string Key(absl::string_view op_type, const DeviceType& device_type,
                  absl::string_view label) {
  return strings::StrCat(op_type, ":", DeviceTypeString(device_type), ":",
                         label);
}

// Provide a way for users to disable JIT kernels for a transitional period.
// Until this is removed, this function also removes the JIT label that is added
// to JIT kernels during the static registration, to allow them to be found
// during lookup as normal kernels.
void SetupOrDisableJit(KernelRegistry* registry) {
  std::unordered_multimap<string, KernelRegistration> jit_kernels;
  bool remove_jit_kernels = absl::StrContains(
      absl::NullSafeStringView(getenv(kDisableJitKernelsEnvVar)), "1");

  mutex_lock l(registry->mu);
  std::unordered_multimap<string, KernelRegistration>& all_kernels =
      registry->registry;
  auto it = all_kernels.begin();
  while (it != all_kernels.end()) {
    if (absl::StrContains(it->second.def.label(), kJitKernelLabel)) {
      // Remove all kernels that have the jit label. They will be added back
      // without the label if they are not to be disabled.
      KernelDef def_without_label = it->second.def;
      def_without_label.set_label("");

      if (!remove_jit_kernels) {
        jit_kernels.emplace(
            Key(def_without_label.op(),
                DeviceType(def_without_label.device_type()),
                def_without_label.label()),
            KernelRegistration(def_without_label, it->second.kernel_class_name,
                               std::move(it->second.factory)));
      }

      it = all_kernels.erase(it);
    } else {
      it++;
    }
  }

  // Add back kernels if they are not disabled. This new key-value pair have all
  // references to the label removed.
  for (auto& jit_kernel : jit_kernels) {
    all_kernels.insert(std::move(jit_kernel));
  }
}

namespace register_kernel {

// Defined out of line to save code space
Name::Name(const char* op) : KernelDefBuilder(op) {}

}  // namespace register_kernel

void* GlobalKernelRegistry() {
  static KernelRegistry* global_kernel_registry = []() {
    KernelRegistry* registry = new KernelRegistry;
    OpRegistry::Global()->RegisterValidator(ValidateKernelRegistrations);
    return registry;
  }();
  return global_kernel_registry;
}

static KernelRegistry* GlobalKernelRegistryTyped() {
#ifdef AUTOLOAD_DYNAMIC_KERNELS
  LoadDynamicKernels();
#endif  // AUTOLOAD_DYNAMIC_KERNELS
  auto* registry = reinterpret_cast<KernelRegistry*>(GlobalKernelRegistry());
  // Update or disable JIT kernels based on user configuration. This is a
  // temporary fallback as part of the initial release of JIT kernels.
  static absl::once_flag setup_or_disable_jit;
  absl::call_once(setup_or_disable_jit, SetupOrDisableJit, registry);
  return registry;
}

namespace kernel_factory {

void OpKernelRegistrar::InitInternal(const KernelDef* kernel_def,
                                     absl::string_view kernel_class_name,
                                     std::unique_ptr<OpKernelFactory> factory) {
  const string key =
      Key(kernel_def->op(), DeviceType(kernel_def->device_type()),
          kernel_def->label());

  // To avoid calling LoadDynamicKernels DO NOT CALL GlobalKernelRegistryTyped
  // here.
  // InitInternal gets called by static initializers, so it ends up executing
  // before main. This causes LoadKernelLibraries function to get called
  // before some file libraries can initialize, which in turn crashes the
  // program flakily. Until we get rid of static initializers in kernel
  // registration mechanism, we have this workaround here.
  auto global_registry =
      reinterpret_cast<KernelRegistry*>(GlobalKernelRegistry());
  mutex_lock l(global_registry->mu);
  global_registry->registry.emplace(
      key,
      KernelRegistration(*kernel_def, kernel_class_name, std::move(factory)));
  delete kernel_def;
}

OpKernel* OpKernelRegistrar::PtrOpKernelFactory::Create(
    OpKernelConstruction* context) {
  return (*create_func_)(context);
}

}  // namespace kernel_factory

namespace {

// Label defaults to empty if not found in NodeDef.
const string& GetKernelLabelAttr(const AttrSlice& node_attrs) {
  static const string& kKernelAttr = *new string("_kernel");
  static const string& kEmptyString = *new string("");

  // NOTE: We inline the implementation of `GetNodeAttrString()` here in order
  // to use the `AttrSlice::FindByString()` overload, which does a more
  // efficient map lookup (instead of a linear scan) when the attribute name is
  // already a `const string&`.
  const AttrValue* attr_value = node_attrs.FindByString(kKernelAttr);
  if (attr_value == nullptr || attr_value->value_case() != AttrValue::kS)
    return kEmptyString;
  else
    return attr_value->s();
}

// TODO(irving): Replace with const Node& version below.
absl::Status FindKernelRegistration(
    const DeviceType& device_type, absl::string_view node_name,
    bool has_experimental_debug_info,
    const NodeDef_ExperimentalDebugInfo& experimental_debug_info,
    absl::string_view node_op, AttrSlice node_attrs,
    const KernelRegistration** reg, bool* was_attr_mismatch) {
  *reg = nullptr;
  *was_attr_mismatch = false;

  const string& label = GetKernelLabelAttr(node_attrs);

  const string key = Key(node_op, device_type, label);
  auto typed_registry = GlobalKernelRegistryTyped();
  tf_shared_lock lock(typed_registry->mu);
  auto regs = typed_registry->registry.equal_range(key);
  for (auto iter = regs.first; iter != regs.second; ++iter) {
    // If there is a kernel registered for the op and device_type,
    // check that the attrs match.
    bool match;
    TF_RETURN_IF_ERROR(KernelAttrsMatch(iter->second.def, node_attrs, &match));
    if (match) {
      if (*reg != nullptr) {
        if ((*reg)->def.priority() == iter->second.def.priority()) {
          return errors::InvalidArgument(
              "Multiple OpKernel registrations match NodeDef at the same "
              "priority '",
              FormatNodeDefForError(node_name, has_experimental_debug_info,
                                    experimental_debug_info),
              "': '", (*reg)->def.ShortDebugString(), "' and '",
              iter->second.def.ShortDebugString(), "'");
        } else if ((*reg)->def.priority() > iter->second.def.priority()) {
          continue;
        }
        // iter->second's priority is higher than *reg.
      }
      *reg = &iter->second;
    } else {
      *was_attr_mismatch = true;
    }
  }
  // Check if no device specific registrations found. If not, try finding a
  // default kernel.
  if (*reg == nullptr &&
      !IsSymbolicExecutionDevice(device_type.type_string())) {
    const string default_key = Key(node_op, DEVICE_DEFAULT, label);
    auto regs = typed_registry->registry.equal_range(default_key);
    for (auto iter = regs.first; iter != regs.second; ++iter) {
      // If there is a kernel registered for the op and device_type,
      // check that the attrs match.
      bool match;
      TF_RETURN_IF_ERROR(
          KernelAttrsMatch(iter->second.def, node_attrs, &match));
      if (match) {
        if (*reg != nullptr) {
          return errors::InvalidArgument(
              "Multiple Default OpKernel registrations match NodeDef '",
              FormatNodeDefForError(node_name, has_experimental_debug_info,
                                    experimental_debug_info),
              "': '", (*reg)->def.ShortDebugString(), "' and '",
              iter->second.def.ShortDebugString(), "'");
        }
        *reg = &iter->second;
      } else {
        *was_attr_mismatch = true;
      }
    }

    if (*reg != nullptr) {
      VLOG(1) << "No device-specific kernels found for NodeDef '"
              << FormatNodeDefForError(node_name, has_experimental_debug_info,
                                       experimental_debug_info)
              << "'"
              << "Will fall back to a default kernel." << std::endl;
    }
  }

  return absl::OkStatus();
}

absl::Status FindKernelRegistration(const DeviceType& device_type,
                                    const NodeDef& node_def,
                                    const KernelRegistration** reg,
                                    bool* was_attr_mismatch) {
  return FindKernelRegistration(
      device_type, node_def.name(), node_def.has_experimental_debug_info(),
      node_def.experimental_debug_info(), node_def.op(),
      AttrSlice(&node_def.attr()), reg, was_attr_mismatch);
}

}  // namespace

bool KernelDefAvailable(const DeviceType& device_type,
                        const NodeDef& node_def) {
  const KernelRegistration* reg = nullptr;
  bool was_attr_mismatch;
  absl::Status result =
      FindKernelRegistration(device_type, node_def, &reg, &was_attr_mismatch);
  return result.ok() && reg != nullptr;
}

// TODO(irving): Change const NodeDef& to const Node&
absl::Status FindKernelDef(
    const DeviceType& device_type, absl::string_view node_name,
    bool has_experimental_debug_info,
    const NodeDef_ExperimentalDebugInfo& experimental_debug_info,
    absl::string_view node_op, absl::string_view node_device,
    AttrSlice node_attrs, const KernelDef** def, string* kernel_class_name) {
  const KernelRegistration* reg = nullptr;
  bool was_attr_mismatch;
  TF_RETURN_IF_ERROR(FindKernelRegistration(
      device_type, node_name, has_experimental_debug_info,
      experimental_debug_info, node_op, node_attrs, &reg, &was_attr_mismatch));
  if (reg == nullptr) {
    const std::string device_str = DeviceTypeString(device_type);
    absl::Status s = errors::NotFound(
        "No registered '", node_op, "' OpKernel for ", device_str,
        " devices compatible with node ",
        FormatNodeDefForError(node_name, has_experimental_debug_info,
                              experimental_debug_info));
    if (was_attr_mismatch) {
      errors::AppendToMessage(
          &s, " (OpKernel was found, but attributes didn't match) ",
          "Requested Attributes: ",
          SummarizeAttrsHelper(node_attrs, node_device));
    }

    // Do not print kernel registrations for other devices when using _JIT
    // devices for compilation or for MKL ops.
    // TODO (intel-tf) : Remove the check for MKL ops when support for
    // block format is removed.
    if (!absl::StrContains(device_str, "JIT") &&
        !absl::StartsWith(node_name, "_Mkl")) {
      errors::AppendToMessage(
          &s, ".  Registered:", KernelsRegisteredForOp(node_op));
    }

    return s;
  }
  if (def != nullptr) *def = &reg->def;
  if (kernel_class_name != nullptr) *kernel_class_name = reg->kernel_class_name;
  return absl::OkStatus();
}

absl::Status FindKernelDef(const DeviceType& device_type,
                           const NodeDef& node_def, const KernelDef** def,
                           string* kernel_class_name) {
  return FindKernelDef(
      device_type, node_def.name(), node_def.has_experimental_debug_info(),
      node_def.experimental_debug_info(), node_def.op(), node_def.device(),
      AttrSlice(&node_def.attr()), def, kernel_class_name);
}

absl::Status SupportedDeviceTypesForNode(
    const std::vector<DeviceType>& prioritized_types, const NodeDef& def,
    PrioritizedDeviceTypeVector* prioritized_device_types,
    const DeviceNameUtils::ParsedName* local_address_spec) {
  // TODO(zhifengc): Changes the callers (SimplePlacer and
  // DynamicPlacer) to consider the possibility that 'def' is call to
  // a user-defined function and only calls this
  // SupportedDeviceTypesForNode for primitive ops.
  const OpRegistrationData* op_reg_data;
  const absl::Status s = OpRegistry::Global()->LookUp(def.op(), &op_reg_data);
  if (s.ok()) {
    bool exists_attr_mismatch = false;
    for (const DeviceType& device_type : prioritized_types) {
      const KernelRegistration* reg = nullptr;
      bool was_attr_mismatch = false;
      TF_RETURN_IF_ERROR(
          FindKernelRegistration(device_type, def, &reg, &was_attr_mismatch));
      exists_attr_mismatch = exists_attr_mismatch || was_attr_mismatch;
      if (reg != nullptr) {
        int32_t priority = reg->def.priority();
        prioritized_device_types->emplace_back(device_type, priority);
      }
    }
    // Add extra supported device types if the following conditions are
    // satisfied:
    // 1) No kernel is defined for the given op (e.g. PyFunc on worker process)
    // 2) A device is requested for this node which specifies job/replica/task
    // 3) A local device is provided which specifies job/replica/task
    // 4) The local device does not have the same (job, replica, task) as the
    //    requested device
    //
    // The goal is to address the issue where a graph includes op (e.g. PyFunc)
    // whose kernel is known to a remote process but not to the current process.
    if (prioritized_device_types->empty() && !exists_attr_mismatch &&
        local_address_spec != nullptr) {
      DeviceNameUtils::ParsedName requested_device_name;
      DeviceNameUtils::ParseFullName(def.device(), &requested_device_name);
      if (DeviceNameUtils::IsDifferentAddressSpace(*local_address_spec,
                                                   requested_device_name)) {
        if (requested_device_name.has_type) {
          prioritized_device_types->push_back(
              std::make_pair(DeviceType(requested_device_name.type), 0));
        } else {
          for (const DeviceType& device_type : prioritized_types) {
            prioritized_device_types->push_back(std::make_pair(device_type, 0));
          }
        }
      }
    }

    // If we were unable to find any valid devices let's validate if the node is
    // even valid.
    if (prioritized_device_types->empty()) {
      TF_RETURN_IF_ERROR(ValidateNodeDef(def, op_reg_data->op_def));
    }

    std::stable_sort(prioritized_device_types->begin(),
                     prioritized_device_types->end(),
                     [](const std::pair<DeviceType, int32>& a,
                        const std::pair<DeviceType, int32>& b) {
                       return a.second > b.second;
                     });
  } else {
    // Assumes that all device types support this node.
    for (const DeviceType& device_type : prioritized_types) {
      prioritized_device_types->push_back(std::make_pair(device_type, 0));
    }
  }
  return absl::OkStatus();
}

void LogAllRegisteredKernels() {
  KernelList kernel_list = GetAllRegisteredKernels();
  for (const auto& kernel_def : kernel_list.kernel()) {
    LOG(INFO) << "OpKernel ('" << kernel_def.ShortDebugString() << "')";
  }
}

KernelList GetAllRegisteredKernels() {
  return GetFilteredRegisteredKernels([](const KernelDef& k) { return true; });
}

KernelList GetFilteredRegisteredKernels(
    const std::function<bool(const KernelDef&)>& predicate) {
  KernelRegistry* const typed_registry = GlobalKernelRegistryTyped();
  KernelList kernel_list;
  tf_shared_lock lock(typed_registry->mu);
  kernel_list.mutable_kernel()->Reserve(typed_registry->registry.size());
  for (const auto& p : typed_registry->registry) {
    const KernelDef& kernel_def = p.second.def;
    if (predicate(kernel_def)) {
      *kernel_list.add_kernel() = kernel_def;
    }
  }
  return kernel_list;
}

KernelList GetRegisteredKernelsForOp(absl::string_view op_name) {
  auto op_pred = [op_name](const KernelDef& k) { return k.op() == op_name; };
  return GetFilteredRegisteredKernels(op_pred);
}

string KernelsRegisteredForOp(absl::string_view op_name) {
  KernelList kernel_list = GetRegisteredKernelsForOp(op_name);
  if (kernel_list.kernel_size() == 0) return "  <no registered kernels>\n";
  string ret;
  for (const auto& kernel_def : kernel_list.kernel()) {
    strings::StrAppend(&ret, "  device='", kernel_def.device_type(), "'");
    if (!kernel_def.label().empty()) {
      strings::StrAppend(&ret, "; label='", kernel_def.label(), "'");
    }
    for (int i = 0; i < kernel_def.constraint_size(); ++i) {
      strings::StrAppend(
          &ret, "; ", kernel_def.constraint(i).name(), " in ",
          SummarizeAttrValue(kernel_def.constraint(i).allowed_values()));
    }
    strings::StrAppend(&ret, "\n");
  }
  return ret;
}

/* TODO(rmlarsen): This API is deprecated. Remove it if possible to avoid
 * copying the NodeDef. */
std::unique_ptr<OpKernel> CreateOpKernel(
    DeviceType device_type, DeviceBase* device, Allocator* allocator,
    const NodeDef& node_def, int graph_def_version, absl::Status* status) {
  // Look up the Op registered for this op name.
  std::shared_ptr<const NodeProperties> props;
  status->Update(NodeProperties::CreateFromNodeDef(
      node_def, OpRegistry::Global(), &props));
  if (!status->ok()) {
    errors::AppendToMessage(status,
                            " for node: ", FormatNodeDefForError(node_def));
    return nullptr;
  }
  return CreateOpKernel(device_type, device, allocator, props,
                        graph_def_version, status);
}

std::unique_ptr<OpKernel> CreateOpKernel(
    DeviceType device_type, DeviceBase* device, Allocator* allocator,
    const std::shared_ptr<const NodeProperties>& props, int graph_def_version,
    absl::Status* status) {
  OpKernel* kernel = nullptr;
  *status = CreateOpKernel(std::move(device_type), device, allocator,
                           /*flib=*/nullptr, props, graph_def_version, &kernel);
  return std::unique_ptr<OpKernel>(kernel);
}

absl::Status CreateOpKernel(DeviceType device_type, DeviceBase* device,
                            Allocator* allocator, FunctionLibraryRuntime* flib,
                            const std::shared_ptr<const NodeProperties>& props,
                            int graph_def_version, OpKernel** kernel) {
  return CreateOpKernel(std::move(device_type), device, allocator, flib,
                        /* resource_mgr= */ nullptr, props, graph_def_version,
                        kernel);
}

absl::Status CreateOpKernel(DeviceType device_type, DeviceBase* device,
                            Allocator* allocator, FunctionLibraryRuntime* flib,
                            ResourceMgr* resource_mgr,
                            const std::shared_ptr<const NodeProperties>& props,
                            int graph_def_version, OpKernel** kernel) {
  const NodeDef& node_def = props->node_def;
  bool was_attr_mismatch;
  const KernelRegistration* registration = nullptr;
  absl::Status s;
  if (props != nullptr) {
    VLOG(1) << "Instantiating kernel for node: " << SummarizeNodeDef(node_def);

    // Validate node_def against OpDef.
    TF_RETURN_IF_ERROR(ValidateNodeDef(node_def, *props->op_def));

    // Look up kernel registration.
    s = FindKernelRegistration(device_type, node_def, &registration,
                               &was_attr_mismatch);
    if (!s.ok()) {
      errors::AppendToMessage(&s, " when instantiating ", node_def.op());
      return s;
    }
  }
  if (registration == nullptr) {
    s.Update(errors::NotFound("No registered '", node_def.op(),
                              "' OpKernel for '", DeviceTypeString(device_type),
                              "' devices compatible with node ",
                              FormatNodeDefForError(node_def)));
    if (was_attr_mismatch) {
      errors::AppendToMessage(
          &s, " (OpKernel was found, but attributes didn't match) ",
          "Requested Attributes: ", SummarizeAttrs(node_def));
    }
    errors::AppendToMessage(
        &s, ".  Registered:", KernelsRegisteredForOp(node_def.op()));
    return s;
  }

  // We are creating a kernel for an op registered in
  // OpRegistry::Global(), we consult the kernel registry to decide
  // the kernel's input and output memory types.
  MemoryTypeVector input_memory_types;
  MemoryTypeVector output_memory_types;
  TF_RETURN_IF_ERROR(MemoryTypesForNode(OpRegistry::Global(), device_type,
                                        node_def, &input_memory_types,
                                        &output_memory_types));

  // Everything needed for OpKernel construction.
  OpKernelConstruction context(std::move(device_type), device, allocator, flib,
                               resource_mgr, props, input_memory_types,
                               output_memory_types, graph_def_version, &s);
  *kernel = registration->factory->Create(&context);
  if (!s.ok()) {
    delete *kernel;
    *kernel = nullptr;
  }
  return s;
}

namespace {

bool FindArgInOp(absl::string_view arg_name,
                 const protobuf::RepeatedPtrField<OpDef::ArgDef>& args) {
  for (const auto& arg : args) {
    if (arg_name == arg.name()) {
      return true;
    }
  }
  return false;
}

}  // namespace

absl::Status ValidateKernelRegistrations(
    const OpRegistryInterface& op_registry) {
  auto typed_registry = GlobalKernelRegistryTyped();
  tf_shared_lock lock(typed_registry->mu);
  for (const auto& key_registration : typed_registry->registry) {
    const KernelDef& kernel_def(key_registration.second.def);
    const OpRegistrationData* op_reg_data;
    const absl::Status status =
        op_registry.LookUp(kernel_def.op(), &op_reg_data);
    if (!status.ok()) {
      LOG(WARNING) << "OpKernel ('" << kernel_def.ShortDebugString()
                   << "') for unknown op: " << kernel_def.op();
      continue;
    }
    const OpDef& op_def = op_reg_data->op_def;
    for (const auto& host_memory_arg : kernel_def.host_memory_arg()) {
      if (!FindArgInOp(host_memory_arg, op_def.input_arg()) &&
          !FindArgInOp(host_memory_arg, op_def.output_arg())) {
        return errors::InvalidArgument(
            "HostMemory arg '", host_memory_arg,
            "' not found in OpDef: ", SummarizeOpDef(op_def));
      }
    }
  }
  return absl::OkStatus();
}

template <>
const Eigen::ThreadPoolDevice& OpKernelContext::eigen_device() const {
  return eigen_cpu_device();
}

template <>
const Eigen::GpuDevice& OpKernelContext::eigen_device() const {
  return eigen_gpu_device();
}

void OpKernelConstruction::CtxFailure(const absl::Status& s) {
  VLOG(1) << s;
  SetStatus(s);
}

void OpKernelConstruction::CtxFailureWithWarning(const absl::Status& s) {
  LOG(WARNING) << s;
  SetStatus(s);
}

void OpKernelConstruction::CtxFailure(const char* file, int line,
                                      const absl::Status& s) {
  VLOG(1) << "OP_REQUIRES failed at " << io::Basename(file) << ":" << line
          << " : " << s;
  SetStatus(s);
}

void OpKernelConstruction::CtxFailureWithWarning(const char* file, int line,
                                                 const absl::Status& s) {
  LOG(WARNING) << "OP_REQUIRES failed at " << io::Basename(file) << ":" << line
               << " : " << s;
  SetStatus(s);
}

void OpKernelContext::CtxFailure(const absl::Status& s) {
  VLOG(1) << s;
  SetStatus(s);
}

void OpKernelContext::CtxFailureWithWarning(const absl::Status& s) {
  LOG(WARNING) << s;
  SetStatus(s);
}

void OpKernelContext::CtxFailure(const char* file, int line,
                                 const absl::Status& s) {
  VLOG(1) << "OP_REQUIRES failed at " << io::Basename(file) << ":" << line
          << " : " << s;
  SetStatus(s);
}

void OpKernelContext::CtxFailureWithWarning(const char* file, int line,
                                            const absl::Status& s) {
  LOG(WARNING) << "OP_REQUIRES failed at " << io::Basename(file) << ":" << line
               << " : " << s;
  SetStatus(s);
}

void CheckNotInComputeAsync(OpKernelContext* ctx,
                            const char* correct_macro_name) {
  CHECK_EQ(nullptr, ctx->params_->op_kernel->AsAsync())
      << "Use " << correct_macro_name << " in AsyncOpKernel implementations.";
}

}  // namespace tensorflow
