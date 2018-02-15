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

#include <unordered_map>
#include <utility>
#include <vector>

#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/framework/device_attributes.pb.h"
#include "tensorflow/core/framework/graph.pb_text.h"
#include "tensorflow/core/framework/kernel_def.pb_text.h"
#include "tensorflow/core/framework/log_memory.h"
#include "tensorflow/core/framework/memory_types.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op_def_util.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

namespace {

Status MatchSignatureHelper(const DataTypeSlice expected_inputs,
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
  return Status::OK();
}

}  // namespace

// OpKernel ------------------------------------------------------------------

// TODO(mrry): Convert to std::make_unique when available.
OpKernel::OpKernel(OpKernelConstruction* context)
    : OpKernel(context,
               std::unique_ptr<const NodeDef>(new NodeDef(context->def()))) {}

OpKernel::OpKernel(OpKernelConstruction* context,
                   std::unique_ptr<const NodeDef> node_def)
    : def_(std::move(node_def)),
      input_types_(context->input_types().begin(),
                   context->input_types().end()),
      input_memory_types_(context->input_memory_types().begin(),
                          context->input_memory_types().end()),
      output_types_(context->output_types().begin(),
                    context->output_types().end()),
      output_memory_types_(context->output_memory_types().begin(),
                           context->output_memory_types().end()),
      graph_def_version_(context->graph_def_version()),
      is_internal_(StringPiece(type_string()).starts_with("_")),
      input_name_map_(context->num_inputs()),
      output_name_map_(context->num_outputs()) {
  OP_REQUIRES_OK(context,
                 NameRangesForNode(*def_, *context->op_def_, &input_name_map_,
                                   &output_name_map_));
  OP_REQUIRES_OK(context, CheckOpDeprecation(*context->op_def_,
                                             context->graph_def_version()));

  // Kernels executing on GPU/SYCL tie very few resources on the CPU where the
  // scheduler runs: we consider them as inexpensive.
  expensive_ = context->device_type() != DeviceType(DEVICE_GPU) &&
               context->device_type() != DeviceType(DEVICE_SYCL);
}

OpKernel::~OpKernel() {}

const string& OpKernel::name() const { return def_->name(); }
const string& OpKernel::type_string() const { return def_->op(); }
const string& OpKernel::requested_device() const { return def_->device(); }
const string& OpKernel::requested_input(int i) const { return def_->input(i); }

Status OpKernel::InputRange(StringPiece input_name, int* start,
                            int* stop) const {
  const auto result = input_name_map_.find(input_name);
  if (result == input_name_map_.end()) {
    return errors::InvalidArgument("Unknown input name: ", input_name);
  } else {
    *start = result->second.first;
    *stop = result->second.second;
    return Status::OK();
  }
}

Status OpKernel::OutputRange(StringPiece output_name, int* start,
                             int* stop) const {
  const auto result = output_name_map_.find(output_name);
  if (result == output_name_map_.end()) {
    return errors::InvalidArgument("Unknown output name: ", output_name);
  } else {
    *start = result->second.first;
    *stop = result->second.second;
    return Status::OK();
  }
}

Status OpKernel::MakeShape(const Tensor& shape, TensorShape* out) const {
  if (!IsLegacyVector(shape.shape())) {
    return errors::InvalidArgument(
        "shape must be a vector of {int32,int64}, got shape ",
        shape.shape().DebugString());
  }
  if (shape.dtype() == DataType::DT_INT32) {
    auto vec = shape.flat<int32>();
    return TensorShapeUtils::MakeShape(vec.data(), vec.size(), out);
  } else if (shape.dtype() == DataType::DT_INT64) {
    auto vec = shape.flat<int64>();
    return TensorShapeUtils::MakeShape(vec.data(), vec.size(), out);
  } else {
    return errors::InvalidArgument("shape must be a vector of {int32,int64}.");
  }
}

void AsyncOpKernel::Compute(OpKernelContext* context) {
  Notification n;
  ComputeAsync(context, [&n]() { n.Notify(); });
  n.WaitForNotification();
}

// PersistentTensor ----------------------------------------------------------

Tensor* PersistentTensor::AccessTensor(OpKernelConstruction* context) {
  // the caller has to have a valid context
  CHECK(context);
  return &tensor_;
}

Tensor* PersistentTensor::AccessTensor(OpKernelContext* context) {
  context->NotifyUseOfPersistentTensor(tensor_);
  return &tensor_;
}

// OpKernelConstruction ------------------------------------------------------

OpKernelConstruction::OpKernelConstruction(
    DeviceType device_type, DeviceBase* device, Allocator* allocator,
    const NodeDef* node_def, const OpDef* op_def, FunctionLibraryRuntime* flib,
    const DataTypeSlice& input_types, const MemoryTypeSlice& input_memory_types,
    const DataTypeSlice& output_types,
    const MemoryTypeSlice& output_memory_types, int graph_def_version,
    Status* status)
    : device_type_(std::move(device_type)),
      device_(device),
      allocator_(allocator),
      def_(node_def),
      op_def_(op_def),
      flib_(flib),
      input_types_(input_types),
      input_memory_types_(input_memory_types),
      output_types_(output_types),
      output_memory_types_(output_memory_types),
      graph_def_version_(graph_def_version),
      status_(status) {}

bool OpKernelConstruction::HasAttr(StringPiece attr_name) const {
  return HasNodeAttr(def(), attr_name);
}

void OpKernelConstruction::SetStatus(const Status& status) {
  status_->Update(status);
}

Status OpKernelConstruction::MatchSignature(
    const DataTypeSlice expected_inputs, const DataTypeSlice expected_outputs) {
  return MatchSignatureHelper(expected_inputs, expected_outputs, input_types_,
                              output_types_);
}

Status OpKernelConstruction::allocate_temp(DataType type,
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
        def_->name(), LogMemory::OP_KERNEL_CONSTRUCTION_STEP_ID, new_temp);
  }
  *out_temp = new_temp;
  return Status::OK();
}

Status OpKernelConstruction::allocate_persistent(
    DataType type, const TensorShape& shape, PersistentTensor* out_persistent,
    Tensor** out_tensor) {
  // for now just do the same thing as allocate_temp
  // TODO(misard) add specific memory tracking for persistent tensors
  Tensor persistent;
  Status s = allocate_temp(type, shape, &persistent);
  if (!s.ok()) {
    return s;
  }
  *out_persistent = PersistentTensor(persistent);
  Tensor* allocated = out_persistent->AccessTensor(this);
  if (out_tensor) {
    *out_tensor = allocated;
  }
  return s;
}

// OpKernelContext -----------------------------------------------------------

OpKernelContext::OpKernelContext(Params* params)
    : OpKernelContext(
          params, static_cast<int>(params->op_kernel->output_types().size())) {}

OpKernelContext::OpKernelContext(Params* params, int num_outputs)
    : params_(params),
      outputs_(num_outputs),
      temp_memory_allocated_(0),
      persistent_memory_allocated_(0) {
  Allocator* eigen_gpu_allocator = get_allocator(AllocatorAttributes());
  params_->ensure_eigen_gpu_device();
  params_->device->ReinitializeGpuDevice(this, params_->eigen_gpu_device,
                                         params_->op_device_context,
                                         eigen_gpu_allocator);
  if (params_->record_tensor_accesses) {
    referenced_tensors_.Init();
  }
}

OpKernelContext::~OpKernelContext() {
  for (TensorValue& value : outputs_) {
    if (!value.is_ref()) {
      delete value.tensor;
    }
  }
  if (params_->record_tensor_accesses) referenced_tensors_.Destroy();
}

Allocator* OpKernelContext::get_allocator(AllocatorAttributes attr) {
  Allocator* allocator =
      params_->device->GetStepAllocator(attr, resource_manager());
  if (track_allocations()) {
    mutex_lock lock(mu_);
    for (const auto& wrapped : wrapped_allocators_) {
      if (wrapped.first == allocator) {
        return wrapped.second;
      }
    }
    TrackingAllocator* wrapped_allocator =
        new TrackingAllocator(allocator, params_->track_allocations);
    wrapped_allocators_.push_back(std::make_pair(allocator, wrapped_allocator));
    return wrapped_allocator;
  } else {
    return allocator;
  }
}

void OpKernelContext::SetStatus(const Status& status) {
  status_.Update(status);
}

void OpKernelContext::really_record_tensor_reference(const Tensor& tensor) {
  mutex_lock l(mu_);
  // Keep a reference to the underlying memory around.
  referenced_tensors_->Add(tensor);
}

Status OpKernelContext::input(StringPiece name, const Tensor** tensor) {
  int start, stop;
  TF_RETURN_IF_ERROR(params_->op_kernel->InputRange(name, &start, &stop));
  if (stop != start + 1) {
    return errors::InvalidArgument("OpKernel used list-valued input name '",
                                   name,
                                   "' when single-valued input was "
                                   "expected");
  }
  if (input_is_ref(start)) {
    return errors::InvalidArgument("OpKernel used ref input name '", name,
                                   "' when non-ref input was expected");
  }
  *tensor = (*params_->inputs)[start].tensor;
  record_tensor_reference(**tensor);
  return Status::OK();
}

Status OpKernelContext::input_dtype(StringPiece name, DataType* dtype) const {
  int start, stop;
  TF_RETURN_IF_ERROR(params_->op_kernel->InputRange(name, &start, &stop));
  if (stop != start + 1) {
    return errors::InvalidArgument("OpKernel used list-valued input name '",
                                   name,
                                   "' when single-valued input was "
                                   "expected");
  }
  const TensorValue& value((*params_->inputs)[start]);
  if (value.is_ref()) {
    *dtype = MakeRefType(value->dtype());
  } else {
    *dtype = value->dtype();
  }
  return Status::OK();
}

Status OpKernelContext::input_ref_mutex(StringPiece name, mutex** out_mutex) {
  int start, stop;
  TF_RETURN_IF_ERROR(params_->op_kernel->InputRange(name, &start, &stop));
  if (stop != start + 1) {
    return errors::InvalidArgument("OpKernel used list-valued input name '",
                                   name,
                                   "' when single-valued input was expected");
  }
  *out_mutex = input_ref_mutex(start);
  return Status::OK();
}

const Tensor& OpKernelContext::input(int index) {
  DCHECK_GE(index, 0);
  DCHECK_LT(index, num_inputs());
  DCHECK(!input_is_ref(index));
  const Tensor& tensor = *((*params_->inputs)[index].tensor);
  record_tensor_reference(tensor);
  return tensor;
}

Tensor OpKernelContext::mutable_input(int index, bool lock_held) {
  DCHECK_GE(index, 0);
  DCHECK_LT(index, num_inputs());
  DCHECK(input_is_ref(index));
  // return a copy of the Ref acquired while holding the mutex
  if (lock_held) {
    Tensor& tensor = *((*params_->inputs)[index].tensor);
    record_tensor_reference(tensor);
    return tensor;
  } else {
    mutex_lock l(*input_ref_mutex(index));
    Tensor& tensor = *((*params_->inputs)[index].tensor);
    record_tensor_reference(tensor);
    return tensor;
  }
}

void OpKernelContext::replace_ref_input(int index, const Tensor& tensor,
                                        bool lock_held) {
  DCHECK_GE(index, 0);
  DCHECK_LT(index, num_inputs());
  DCHECK(input_is_ref(index));
  // should only modify the tensor while holding the mutex
  if (lock_held) {
    *(*params_->inputs)[index].tensor = tensor;
  } else {
    mutex_lock l(*input_ref_mutex(index));
    *(*params_->inputs)[index].tensor = tensor;
  }
  record_tensor_reference(tensor);
}

void OpKernelContext::forward_ref_input_to_ref_output(int input_index,
                                                      int output_index) {
  DCHECK_GE(input_index, 0);
  DCHECK_LT(input_index, num_inputs());
  DCHECK(input_is_ref(input_index));
  set_output_ref(output_index, (*params_->inputs)[input_index].mutex_if_ref,
                 (*params_->inputs)[input_index].tensor);
}

bool OpKernelContext::forward_input_to_output_with_shape(
    int input_index, int output_index, const TensorShape& output_shape,
    Tensor** output) {
  const auto output_attr = params_->output_attr_array == nullptr
                               ? AllocatorAttributes()
                               : output_alloc_attr(output_index);
  std::unique_ptr<Tensor> new_tensor = forward_input(
      input_index, expected_output_dtype(output_index), output_shape,
      output_memory_type(output_index), output_attr);
  if (new_tensor != nullptr) {
    // Transfer ownership to the output slot in OpKernelContext.
    outputs_[output_index] = TensorValue(new_tensor.release());
    *output = outputs_[output_index].tensor;
    return true;
  } else {
    return false;
  }
}

Status OpKernelContext::forward_input_to_output_with_shape(
    StringPiece input_name, StringPiece output_name,
    const TensorShape& output_shape, Tensor** output) {
  int input_index, output_index, stop;
  TF_RETURN_IF_ERROR(
      params_->op_kernel->InputRange(input_name, &input_index, &stop));
  if (stop != input_index + 1) {
    return errors::InvalidArgument("OpKernel used list-valued input name '",
                                   input_name,
                                   "' when single-valued input was "
                                   "expected");
  }
  TF_RETURN_IF_ERROR(
      params_->op_kernel->OutputRange(output_name, &output_index, &stop));
  if (stop != output_index + 1) {
    return errors::InvalidArgument("OpKernel used list-valued output name '",
                                   output_name,
                                   "' when single-valued output was "
                                   "expected");
  }
  if (!forward_input_to_output_with_shape(input_index, output_index,
                                          output_shape, output)) {
    return errors::FailedPrecondition("OpKernel could not forward input '",
                                      input_name, "' to output '", output_name);
  }
  return Status::OK();
}

std::unique_ptr<Tensor> OpKernelContext::forward_input(
    int input_index, DataType output_dtype, const TensorShape& output_shape,
    MemoryType output_memory_type, const AllocatorAttributes& output_attr) {
  DCHECK_GE(input_index, 0);
  DCHECK_LT(input_index, num_inputs());
  const TensorValue& input = (*params_->inputs)[input_index];
  // Check that input tensor exists, is not a ref, and has no other consumers.
  if (input.tensor == nullptr || input.is_ref() || !input->RefCountIsOne()) {
    return nullptr;
  }
  // Check that input type matches.
  if (input_dtype(input_index) != output_dtype) {
    return nullptr;
  }
  // Check that the input and output sizes are compatible.
  if (input.tensor->shape().num_elements() != output_shape.num_elements()) {
    return nullptr;
  }
  // Check that input and output memory types match, i.e.
  // that they either both live in host or both live in device memory.
  if (input_memory_type(input_index) != output_memory_type) {
    return nullptr;
  }
  // Check that output allocator attributes are not more restrictive than
  // input allocator attributes.
  const auto input_attr = params_->input_alloc_attrs == nullptr
                              ? AllocatorAttributes()
                              : input_alloc_attr(input_index);
  if (!output_attr.IsEqualOrLessRestrictiveThan(input_attr)) {
    return nullptr;
  }
  // TODO(rmlarsen): Use MakeUnique here. There is already a copy in
  // tensorflow/compiler/xla/ptr_util.h. Perhaps this should be part of
  // general cleanup of ownership in this code.
  std::unique_ptr<Tensor> output_tensor(new Tensor());
  CHECK(output_tensor->CopyFrom(*input.tensor, output_shape));
  return output_tensor;
}

Status OpKernelContext::forward_input_or_allocate_temp(
    gtl::ArraySlice<int> candidate_input_indices, DataType type,
    const TensorShape& shape, const AllocatorAttributes& allocator_attr,
    Tensor* out_temp) {
  for (int input_index : candidate_input_indices) {
    std::unique_ptr<Tensor> new_tensor =
        forward_input(input_index, type, shape, DEVICE_MEMORY, allocator_attr);
    if (new_tensor != nullptr) {
      *out_temp = std::move(*new_tensor);
      return Status::OK();
    }
  }
  return allocate_temp(type, shape, out_temp, allocator_attr);
}

void OpKernelContext::delete_ref_input(int index, bool lock_held) {
  DCHECK_GE(index, 0);
  DCHECK_LT(index, num_inputs());
  DCHECK(input_is_ref(index));
  // should only modify the tensor while holding the mutex
  if (lock_held) {
    delete (*params_->inputs)[index].tensor;
  } else {
    mutex_lock l(*input_ref_mutex(index));
    delete (*params_->inputs)[index].tensor;
  }
}

Status OpKernelContext::mutable_input(StringPiece name, Tensor* tensor,
                                      bool lock_held) {
  int start, stop;
  TF_RETURN_IF_ERROR(params_->op_kernel->InputRange(name, &start, &stop));
  if (stop != start + 1) {
    return errors::InvalidArgument("OpKernel used list-valued input name '",
                                   name,
                                   "' when single-valued input was expected");
  }
  if (!input_is_ref(start)) {
    return errors::InvalidArgument("OpKernel used non-ref input name '", name,
                                   "' when ref input was expected");
  }
  // return a copy of the Ref acquired while holding the mutex
  if (lock_held) {
    *tensor = *(*params_->inputs)[start].tensor;
  } else {
    mutex_lock l(*input_ref_mutex(start));
    *tensor = *(*params_->inputs)[start].tensor;
  }
  record_tensor_reference(*tensor);
  return Status::OK();
}

Status OpKernelContext::replace_ref_input(StringPiece name,
                                          const Tensor& tensor,
                                          bool lock_held) {
  int start, stop;
  TF_RETURN_IF_ERROR(params_->op_kernel->InputRange(name, &start, &stop));
  if (stop != start + 1) {
    return errors::InvalidArgument("OpKernel used list-valued input name '",
                                   name,
                                   "' when single-valued input was expected");
  }
  if (!input_is_ref(start)) {
    return errors::InvalidArgument("OpKernel used immutable input name '", name,
                                   "' when ref input was expected");
  }
  replace_ref_input(start, tensor, lock_held);
  return Status::OK();
}

Status OpKernelContext::input_list(StringPiece name, OpInputList* list) {
  int start, stop;
  TF_RETURN_IF_ERROR(params_->op_kernel->InputRange(name, &start, &stop));
  *list = OpInputList(this, start, stop);
  return Status::OK();
}

Status OpKernelContext::mutable_input_list(StringPiece name,
                                           OpMutableInputList* list) {
  int start, stop;
  TF_RETURN_IF_ERROR(params_->op_kernel->InputRange(name, &start, &stop));
  *list = OpMutableInputList(this, start, stop);
  return Status::OK();
}

Status OpKernelContext::output_list(StringPiece name, OpOutputList* list) {
  int start, stop;
  TF_RETURN_IF_ERROR(params_->op_kernel->OutputRange(name, &start, &stop));
  *list = OpOutputList(this, start, stop);
  return Status::OK();
}

Status OpKernelContext::allocate_output(int index, const TensorShape& shape,
                                        Tensor** output) {
  DCHECK_GE(index, 0);
  DCHECK_LT(index, num_outputs());
  AllocatorAttributes attr = output_alloc_attr(index);
  return allocate_output(index, shape, output, attr);
}

Status OpKernelContext::allocate_output(StringPiece name,
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

Status OpKernelContext::allocate_output(StringPiece name,
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

Status OpKernelContext::allocate_tensor(
    DataType type, const TensorShape& shape, Tensor* out_tensor,
    AllocatorAttributes attr, const AllocationAttributes& allocation_attr) {
  Allocator* a = get_allocator(attr);
  AllocationAttributes logged_attr(allocation_attr);
  logged_attr.allocation_will_be_logged = true;
  Tensor new_tensor(a, type, shape, logged_attr);

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
  record_tensor_reference(new_tensor);
  *out_tensor = std::move(new_tensor);
  return Status::OK();
}

Status OpKernelContext::allocate_output(int index, const TensorShape& shape,
                                        Tensor** output,
                                        AllocatorAttributes attr) {
  DCHECK_GE(index, 0);
  DCHECK_LT(index, outputs_.size());
  const DataType type = params_->op_kernel->output_type(index);
  DCHECK(!IsRefType(type));
  DCHECK(mutable_output(index) == nullptr);
  Tensor* output_tensor = new Tensor();
  Status s = allocate_tensor(type, shape, output_tensor, attr);
  if (s.ok()) {
    outputs_[index] = TensorValue(output_tensor);
    *output = outputs_[index].tensor;
  }
  return s;
}

Status OpKernelContext::allocate_temp(
    DataType type, const TensorShape& shape, Tensor* out_temp,
    AllocatorAttributes allocator_attr,
    const AllocationAttributes& allocation_attr) {
  Status s =
      allocate_tensor(type, shape, out_temp, allocator_attr, allocation_attr);
  if (track_allocations() && s.ok() && out_temp->TotalBytes() > 0) {
    Allocator* a = get_allocator(allocator_attr);
    if (a->TracksAllocationSizes()) {
      int64 alloc_size = a->AllocatedSize(out_temp->tensor_data().data());
      record_temp_memory_allocation(alloc_size, *out_temp);
    }
  }
  return s;
}

Status OpKernelContext::allocate_persistent(DataType type,
                                            const TensorShape& shape,
                                            PersistentTensor* out_persistent,
                                            Tensor** out_tensor,
                                            AllocatorAttributes attr) {
  Tensor persistent;
  Status s = allocate_tensor(type, shape, &persistent, attr);
  if (s.ok()) {
    *out_persistent = PersistentTensor(persistent);
    if (out_tensor) {
      *out_tensor = out_persistent->AccessTensor(this);
    }
    if (track_allocations()) {
      Tensor* t = out_persistent->AccessTensor(this);
      Allocator* a = get_allocator(attr);
      if (a->TracksAllocationSizes()) {
        int64 alloc_size = a->AllocatedSize(t->tensor_data().data());
        int64 alloc_id = a->AllocationId(t->tensor_data().data());
        record_persistent_memory_allocation(alloc_size, alloc_id);
      }
    }
  }
  return s;
}

Status OpKernelContext::set_output(StringPiece name, const Tensor& tensor) {
  int start, stop;
  TF_RETURN_IF_ERROR(params_->op_kernel->OutputRange(name, &start, &stop));
  if (stop != start + 1) {
    return errors::InvalidArgument("OpKernel used list-valued output name '",
                                   name,
                                   "' when single-valued output was "
                                   "expected");
  }
  set_output(start, tensor);
  return Status::OK();
}

void OpKernelContext::set_output(int index, const Tensor& tensor) {
  DCHECK_GE(index, 0);
  DCHECK_LT(index, outputs_.size());
  DCHECK(!IsRefType(params_->op_kernel->output_type(index)));
  DCHECK_EQ(mutable_output(index), nullptr);
  record_tensor_reference(tensor);
  outputs_[index] = TensorValue(new Tensor(tensor));
  if (track_allocations() && tensor.TotalBytes() > 0) {
    mutex_lock l(stats_mu_);
    if (!temp_tensor_buffer_and_size_) {
      return;
    }
    auto it = std::find_if(temp_tensor_buffer_and_size_->begin(),
                           temp_tensor_buffer_and_size_->end(),
                           [&tensor](const std::pair<const void*, int64>& e) {
                             return e.first == static_cast<const void*>(
                                                   tensor.tensor_data().data());
                           });
    if (it != temp_tensor_buffer_and_size_->end()) {
      temp_memory_allocated_ -= it->second;
      temp_tensor_buffer_and_size_->erase(it);
    }
  }
}

void OpKernelContext::set_output_ref(int index, mutex* mu,
                                     Tensor* tensor_for_ref) {
  DCHECK_GE(index, 0);
  DCHECK_LT(index, outputs_.size());
  DCHECK(IsRefType(params_->op_kernel->output_type(index)));
  record_tensor_reference(*tensor_for_ref);
  outputs_[index] = TensorValue(mu, tensor_for_ref);
}

Status OpKernelContext::set_output_ref(StringPiece name, mutex* mu,
                                       Tensor* tensor_for_ref) {
  int start, stop;
  TF_RETURN_IF_ERROR(params_->op_kernel->OutputRange(name, &start, &stop));
  if (stop != start + 1) {
    return errors::InvalidArgument("OpKernel used list-valued output name '",
                                   name,
                                   "' when single-valued output was "
                                   "expected");
  }
  set_output_ref(start, mu, tensor_for_ref);
  return Status::OK();
}

Status OpKernelContext::mutable_output(StringPiece name, Tensor** tensor) {
  int start, stop;
  TF_RETURN_IF_ERROR(params_->op_kernel->OutputRange(name, &start, &stop));
  if (stop != start + 1) {
    return errors::InvalidArgument("OpKernel used list-valued output name '",
                                   name,
                                   "' when single-valued output was "
                                   "expected");
  }
  *tensor = mutable_output(start);
  return Status::OK();
}

Status OpKernelContext::release_output(StringPiece name, TensorValue* value) {
  int start, stop;
  TF_RETURN_IF_ERROR(params_->op_kernel->OutputRange(name, &start, &stop));
  if (stop != start + 1) {
    return errors::InvalidArgument("OpKernel used list-valued output name '",
                                   name,
                                   "' when single-valued output was "
                                   "expected");
  }
  *value = release_output(start);
  return Status::OK();
}

bool OpKernelContext::ValidateInputsAreSameShape(OpKernel* op) {
  const auto& inputs = *params_->inputs;
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

Status OpKernelContext::MatchSignature(const DataTypeSlice expected_inputs,
                                       const DataTypeSlice expected_outputs) {
  DataTypeVector inputs;
  for (const TensorValue& t : *params_->inputs) {
    inputs.push_back(t.is_ref() ? MakeRefType(t->dtype()) : t->dtype());
  }
  DataTypeVector outputs = params_->op_kernel->output_types();
  return MatchSignatureHelper(expected_inputs, expected_outputs, inputs,
                              outputs);
}

void OpKernelContext::record_temp_memory_allocation(int64 size,
                                                    const Tensor& t) {
  mutex_lock l(stats_mu_);
  temp_memory_allocated_ += size;
  if (!temp_tensor_buffer_and_size_) {
    temp_tensor_buffer_and_size_.reset(
        new gtl::InlinedVector<std::pair<const void*, int64>, 2>());
  }
  temp_tensor_buffer_and_size_->emplace_back(
      static_cast<const void*>(t.tensor_data().data()), size);
}

int64 OpKernelContext::temp_memory_allocated() const {
  mutex_lock l(stats_mu_);
  return temp_memory_allocated_;
}

void OpKernelContext::record_persistent_memory_allocation(int64 size,
                                                          int64 alloc_id) {
  mutex_lock l(stats_mu_);
  persistent_memory_allocated_ += size;
  if (alloc_id >= 0) {
    if (!persistent_alloc_ids_) {
      persistent_alloc_ids_.reset(new gtl::InlinedVector<int64, 2>());
    }
    persistent_alloc_ids_->push_back(alloc_id);
  }
}

int64 OpKernelContext::persistent_memory_allocated() const {
  mutex_lock l(stats_mu_);
  return persistent_memory_allocated_;
}

std::vector<int64> OpKernelContext::persistent_alloc_ids() const {
  mutex_lock l(stats_mu_);
  if (persistent_alloc_ids_) {
    return std::vector<int64>(persistent_alloc_ids_->begin(),
                              persistent_alloc_ids_->end());
  } else {
    return std::vector<int64>();
  }
}

void OpKernelContext::clear_recorded_memory() {
  mutex_lock l(stats_mu_);
  temp_memory_allocated_ = 0;
  persistent_memory_allocated_ = 0;
  if (temp_tensor_buffer_and_size_) {
    temp_tensor_buffer_and_size_->clear();
  }
  if (persistent_alloc_ids_) {
    persistent_alloc_ids_->clear();
  }
}

// OpKernel registration ------------------------------------------------------

struct KernelRegistration {
  KernelRegistration(const KernelDef& d, StringPiece c,
                     kernel_factory::OpKernelRegistrar::Factory f)
      : def(d), kernel_class_name(c.ToString()), factory(f) {}
  const KernelDef def;
  const string kernel_class_name;
  const kernel_factory::OpKernelRegistrar::Factory factory;
};

// This maps from 'op_type' + DeviceType to the set of KernelDefs and
// factory functions for instantiating the OpKernel that matches the
// KernelDef.
typedef std::unordered_multimap<string, KernelRegistration> KernelRegistry;

void* GlobalKernelRegistry() {
  static KernelRegistry* global_kernel_registry = new KernelRegistry;
  return global_kernel_registry;
}

static KernelRegistry* GlobalKernelRegistryTyped() {
  return reinterpret_cast<KernelRegistry*>(GlobalKernelRegistry());
}

static string Key(StringPiece op_type, const DeviceType& device_type,
                  StringPiece label) {
  return strings::StrCat(op_type, ":", DeviceTypeString(device_type), ":",
                         label);
}

namespace kernel_factory {

void OpKernelRegistrar::InitInternal(const KernelDef* kernel_def,
                                     StringPiece kernel_class_name,
                                     Factory factory) {
  // See comments in register_kernel::Name in header for info on _no_register.
  if (kernel_def->op() != "_no_register") {
    const string key =
        Key(kernel_def->op(), DeviceType(kernel_def->device_type()),
            kernel_def->label());
    GlobalKernelRegistryTyped()->insert(std::make_pair(
        key, KernelRegistration(*kernel_def, kernel_class_name, factory)));
  }
  delete kernel_def;
}

}  // namespace kernel_factory

namespace {

// Helper for AttrsMatch().
bool InTypeList(DataType dt, const AttrValue& type_list) {
  for (int in_list : type_list.list().type()) {
    if (dt == in_list) return true;
  }
  return false;
}

// Returns whether the attrs satisfy the constraints in the kernel_def.  Returns
// an error if attrs in kernel_def are not found, or have a mismatching type.
Status AttrsMatch(AttrSlice attrs, const KernelDef& kernel_def, bool* match) {
  *match = false;
  for (const auto& constraint : kernel_def.constraint()) {
    if (constraint.allowed_values().list().type_size() == 0) {
      return errors::Unimplemented(
          "KernelDef '", ProtoShortDebugString(kernel_def),
          " has constraint on attr '", constraint.name(),
          "' with unsupported type: ",
          SummarizeAttrValue(constraint.allowed_values()));
    }

    const AttrValue* found = attrs.Find(constraint.name());
    if (found) {
      if (found->type() != DT_INVALID) {
        if (!InTypeList(found->type(), constraint.allowed_values())) {
          return Status::OK();
        }
      } else {
        if (!AttrValueHasType(*found, "list(type)").ok()) {
          return errors::InvalidArgument(
              "KernelDef '", ProtoShortDebugString(kernel_def),
              "' has constraint on attr '", constraint.name(),
              "' that has value '", SummarizeAttrValue(*found),
              "' that does not have type 'type' or 'list(type)' in NodeDef "
              "'",
              attrs.SummarizeNode(), "'");
        }

        for (int t : found->list().type()) {
          if (!InTypeList(static_cast<DataType>(t),
                          constraint.allowed_values())) {
            return Status::OK();
          }
        }
      }
    } else {
      return errors::InvalidArgument(
          "OpKernel '", kernel_def.op(), "' has constraint on attr '",
          constraint.name(), "' not in NodeDef '", attrs.SummarizeNode(),
          "', KernelDef: '", ProtoShortDebugString(kernel_def), "'");
    }
  }
  *match = true;
  return Status::OK();
}

static const StringPiece kKernelAttr("_kernel");

// TODO(irving): Replace with const Node& version below.
Status FindKernelRegistration(const DeviceType& device_type,
                              const NodeDef& node_def,
                              const KernelRegistration** reg,
                              bool* was_attr_mismatch) {
  *reg = nullptr;
  *was_attr_mismatch = false;
  // Label defaults to empty if not found in NodeDef.
  const string& label = GetNodeAttrString(node_def, kKernelAttr);

  const string key = Key(node_def.op(), device_type, label);
  auto regs = GlobalKernelRegistryTyped()->equal_range(key);
  for (auto iter = regs.first; iter != regs.second; ++iter) {
    // If there is a kernel registered for the op and device_type,
    // check that the attrs match.
    bool match;
    TF_RETURN_IF_ERROR(AttrsMatch(node_def, iter->second.def, &match));
    if (match) {
      if (*reg != nullptr) {
        return errors::InvalidArgument(
            "Multiple OpKernel registrations match NodeDef '",
            SummarizeNodeDef(node_def), "': '",
            ProtoShortDebugString((*reg)->def), "' and '",
            ProtoShortDebugString(iter->second.def), "'");
      }
      *reg = &iter->second;
    } else {
      *was_attr_mismatch = true;
    }
  }
  return Status::OK();
}

}  // namespace

// TODO(irving): Change const NodeDef& to const Node&
Status FindKernelDef(const DeviceType& device_type, const NodeDef& node_def,
                     const KernelDef** def, string* kernel_class_name) {
  const KernelRegistration* reg = nullptr;
  bool was_attr_mismatch;
  TF_RETURN_IF_ERROR(
      FindKernelRegistration(device_type, node_def, &reg, &was_attr_mismatch));
  if (reg == nullptr) {
    Status s = errors::NotFound(
        "No registered '", node_def.op(), "' OpKernel for ",
        DeviceTypeString(device_type), " devices compatible with node ",
        SummarizeNodeDef(node_def));
    if (was_attr_mismatch) {
      errors::AppendToMessage(
          &s, " (OpKernel was found, but attributes didn't match)");
    }
    errors::AppendToMessage(
        &s, ".  Registered:", KernelsRegisteredForOp(node_def.op()));
    return s;
  }
  if (def != nullptr) *def = &reg->def;
  if (kernel_class_name != nullptr) *kernel_class_name = reg->kernel_class_name;
  return Status::OK();
}

Status SupportedDeviceTypesForNode(
    const std::vector<DeviceType>& prioritized_types, const NodeDef& def,
    DeviceTypeVector* device_types) {
  // TODO(zhifengc): Changes the callers (SimplePlacer and
  // DynamicPlacer) to consider the possibility that 'def' is call to
  // a user-defined function and only calls this
  // SupportedDeviceTypesForNode for primitive ops.
  const OpRegistrationData* op_reg_data;
  const Status s = OpRegistry::Global()->LookUp(def.op(), &op_reg_data);
  if (s.ok()) {
    for (const DeviceType& device_type : prioritized_types) {
      const KernelRegistration* reg = nullptr;
      bool was_attr_mismatch;
      TF_RETURN_IF_ERROR(
          FindKernelRegistration(device_type, def, &reg, &was_attr_mismatch));
      if (reg != nullptr) device_types->push_back(device_type);
    }
  } else {
    // Assumes that all device types support this node.
    for (const DeviceType& device_type : prioritized_types) {
      device_types->push_back(device_type);
    }
  }
  return Status::OK();
}

void LogAllRegisteredKernels() {
  for (const auto& key_registration : *GlobalKernelRegistryTyped()) {
    const KernelDef& kernel_def(key_registration.second.def);
    LOG(INFO) << "OpKernel ('" << ProtoShortDebugString(kernel_def) << "')";
  }
}

string KernelsRegisteredForOp(StringPiece op_name) {
  string ret;
  for (const auto& key_registration : *GlobalKernelRegistryTyped()) {
    const KernelDef& kernel_def(key_registration.second.def);
    if (kernel_def.op() == op_name) {
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
  }
  if (ret.empty()) return "  <no registered kernels>\n";
  return ret;
}

std::unique_ptr<OpKernel> CreateOpKernel(
    DeviceType device_type, DeviceBase* device, Allocator* allocator,
    const NodeDef& node_def, int graph_def_version, Status* status) {
  OpKernel* kernel = nullptr;
  *status = CreateOpKernel(std::move(device_type), device, allocator, nullptr,
                           node_def, graph_def_version, &kernel);
  return std::unique_ptr<OpKernel>(kernel);
}

Status CreateOpKernel(DeviceType device_type, DeviceBase* device,
                      Allocator* allocator, FunctionLibraryRuntime* flib,
                      const NodeDef& node_def, int graph_def_version,
                      OpKernel** kernel) {
  VLOG(1) << "Instantiating kernel for node: " << SummarizeNodeDef(node_def);

  // Look up the Op registered for this op name.
  const OpDef* op_def = nullptr;
  Status s = OpRegistry::Global()->LookUpOpDef(node_def.op(), &op_def);
  if (!s.ok()) return s;

  // Validate node_def against OpDef.
  s = ValidateNodeDef(node_def, *op_def);
  if (!s.ok()) return s;

  // Look up kernel registration.
  const KernelRegistration* registration;
  bool was_attr_mismatch;
  s = FindKernelRegistration(device_type, node_def, &registration,
                             &was_attr_mismatch);
  if (!s.ok()) {
    errors::AppendToMessage(&s, " when instantiating ", node_def.op());
    return s;
  }
  if (registration == nullptr) {
    s.Update(errors::NotFound("No registered '", node_def.op(),
                              "' OpKernel for ", DeviceTypeString(device_type),
                              " devices compatible with node ",
                              SummarizeNodeDef(node_def)));
    if (was_attr_mismatch) {
      errors::AppendToMessage(
          &s, " (OpKernel was found, but attributes didn't match)");
    }
    errors::AppendToMessage(
        &s, ".  Registered:", KernelsRegisteredForOp(node_def.op()));
    return s;
  }

  // Get signature from the OpDef & NodeDef
  DataTypeVector inputs;
  DataTypeVector outputs;
  s.Update(InOutTypesForNode(node_def, *op_def, &inputs, &outputs));
  if (!s.ok()) {
    errors::AppendToMessage(&s, " for node: ", SummarizeNodeDef(node_def));
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
  OpKernelConstruction context(
      device_type, device, allocator, &node_def, op_def, flib, inputs,
      input_memory_types, outputs, output_memory_types, graph_def_version, &s);
  *kernel = (*registration->factory)(&context);
  if (!s.ok()) {
    delete *kernel;
    *kernel = nullptr;
  }
  return s;
}

namespace {

bool FindArgInOp(StringPiece arg_name,
                 const protobuf::RepeatedPtrField<OpDef::ArgDef>& args) {
  for (const auto& arg : args) {
    if (arg_name == arg.name()) {
      return true;
    }
  }
  return false;
}

}  // namespace

Status ValidateKernelRegistrations(const OpRegistryInterface& op_registry) {
  for (const auto& key_registration : *GlobalKernelRegistryTyped()) {
    const KernelDef& kernel_def(key_registration.second.def);
    const OpRegistrationData* op_reg_data;
    const Status status = op_registry.LookUp(kernel_def.op(), &op_reg_data);
    if (!status.ok()) {
      // TODO(josh11b): Make this a hard error.
      LOG(ERROR) << "OpKernel ('" << ProtoShortDebugString(kernel_def)
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
  return Status::OK();
}

template <>
const Eigen::ThreadPoolDevice& OpKernelContext::eigen_device() const {
  return eigen_cpu_device();
}

template <>
const Eigen::GpuDevice& OpKernelContext::eigen_device() const {
  return eigen_gpu_device();
}

#ifdef TENSORFLOW_USE_SYCL
template <>
const Eigen::SyclDevice& OpKernelContext::eigen_device() const {
  return eigen_sycl_device();
}
#endif

void OpKernelConstruction::CtxFailure(const Status& s) {
  VLOG(1) << s;
  SetStatus(s);
}

void OpKernelConstruction::CtxFailureWithWarning(const Status& s) {
  LOG(WARNING) << s;
  SetStatus(s);
}

void OpKernelConstruction::CtxFailure(const char* file, int line,
                                      const Status& s) {
  VLOG(1) << "OP_REQUIRES failed at " << io::Basename(file) << ":" << line
          << " : " << s;
  SetStatus(s);
}

void OpKernelConstruction::CtxFailureWithWarning(const char* file, int line,
                                                 const Status& s) {
  LOG(WARNING) << "OP_REQUIRES failed at " << io::Basename(file) << ":" << line
               << " : " << s;
  SetStatus(s);
}

void OpKernelContext::CtxFailure(const Status& s) {
  VLOG(1) << s;
  SetStatus(s);
}

void OpKernelContext::CtxFailureWithWarning(const Status& s) {
  LOG(WARNING) << s;
  SetStatus(s);
}

void OpKernelContext::CtxFailure(const char* file, int line, const Status& s) {
  VLOG(1) << "OP_REQUIRES failed at " << io::Basename(file) << ":" << line
          << " : " << s;
  SetStatus(s);
}

void OpKernelContext::CtxFailureWithWarning(const char* file, int line,
                                            const Status& s) {
  LOG(WARNING) << "OP_REQUIRES failed at " << io::Basename(file) << ":" << line
               << " : " << s;
  SetStatus(s);
}

}  // namespace tensorflow
