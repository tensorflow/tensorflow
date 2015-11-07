#include "tensorflow/core/framework/op_kernel.h"

#include <unordered_map>

#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op_def_util.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/port.h"

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
    return errors::InvalidArgument("Signature mismatch, have: ",
                                   DataTypeSliceString(inputs), "->",
                                   DataTypeSliceString(outputs), " expected: ",
                                   DataTypeSliceString(expected_inputs), "->",
                                   DataTypeSliceString(expected_outputs));
  }
  return Status::OK();
}

// Check HostMemory backward compatibility.
bool CheckHostMemoryCompatibility(const DeviceType device_type,
                                  const OpKernel* kernel) {
  if (device_type == DEVICE_GPU) {
    for (int i = 0; i < kernel->num_inputs(); ++i) {
      if (kernel->input_type(i) == DT_INT32 &&
          kernel->input_memory_types()[i] != HOST_MEMORY) {
        return false;
      }
    }
    for (int i = 0; i < kernel->num_outputs(); ++i) {
      if (kernel->output_type(i) == DT_INT32 &&
          kernel->output_memory_types()[i] != HOST_MEMORY) {
        return false;
      }
    }
  }
  return true;
}

}  // namespace

// OpKernel ------------------------------------------------------------------

OpKernel::OpKernel(OpKernelConstruction* context)
    : def_(context->def()),
      input_types_(context->input_types().begin(),
                   context->input_types().end()),
      output_types_(context->output_types().begin(),
                    context->output_types().end()),
      input_name_map_(context->num_inputs()),
      output_name_map_(context->num_outputs()) {
  OP_REQUIRES_OK(context,
                 NameRangesForNode(def_, context->op_def(), &input_name_map_,
                                   &output_name_map_));

  // By default, the input and output memory types are always in device memory,
  // but can be overridden by individual implementations of OpKernels in their
  // constructor.
  input_memory_types_ = MemoryTypeVector(input_types_.size(), DEVICE_MEMORY);
  output_memory_types_ = MemoryTypeVector(output_types_.size(), DEVICE_MEMORY);
  // TODO(yuanbyu): For now we assume the memory types of function
  // inputs/outputs to be DEVICE_MEMORY.
  auto lib = context->function_library();
  if (lib == nullptr || !lib->IsDefined(def_.op())) {
    OP_REQUIRES_OK(context, MemoryTypesForNode(
                                context->device_type(), def_, context->op_def(),
                                input_name_map_, output_name_map_,
                                &input_memory_types_, &output_memory_types_));
    // Log all the uses of int32 on GPU.
    // TODO(yunabyu): Remove once everyone transitions to HostMemory.
    if (VLOG_IS_ON(2)) {
      if (!CheckHostMemoryCompatibility(context->device_type(), this)) {
        VLOG(2) << "Using int32 on GPU at node: " << SummarizeNodeDef(def());
      }
    }
  }
}

Status OpKernel::InputRange(const string& input_name, int* start,
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

Status OpKernel::OutputRange(const string& output_name, int* start,
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

Status OpKernelConstruction::MatchSignature(
    const DataTypeSlice expected_inputs, const DataTypeSlice expected_outputs) {
  return MatchSignatureHelper(expected_inputs, expected_outputs, input_types_,
                              output_types_);
}

Status OpKernelConstruction::allocate_temp(DataType type,
                                           const TensorShape& shape,
                                           Tensor* out_temp) {
  Tensor new_temp(allocator_, type, shape);

  if (!new_temp.IsInitialized() && shape.num_elements() > 0) {
    return errors::ResourceExhausted(
        "OOM when allocating temporary tensor with shape", shape.DebugString());
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

OpKernelContext::OpKernelContext(const Params& params)
    : params_(params),
      outputs_(params.op_kernel->output_types().size()),
      output_allocation_types_(params.op_kernel->output_types().size()) {
  Allocator* eigen_gpu_allocator = get_allocator(AllocatorAttributes());
  eigen_gpu_device_ = params_.device->MakeGpuDevice(params_.op_device_context,
                                                    eigen_gpu_allocator);
}

OpKernelContext::~OpKernelContext() {
  for (TensorValue& value : outputs_) {
    if (!value.is_ref()) {
      delete value.tensor;
    }
  }
  for (Tensor* t : temp_tensors_) delete t;
  delete eigen_gpu_device_;
}

Status OpKernelContext::input(const string& name, const Tensor** tensor) const {
  int start, stop;
  TF_RETURN_IF_ERROR(params_.op_kernel->InputRange(name, &start, &stop));
  if (stop != start + 1) {
    return errors::InvalidArgument("OpKernel used list-valued input name '",
                                   name,
                                   "' when single-valued input was "
                                   "expected");
  }
  if ((*params_.inputs)[start].is_ref()) {
    return errors::InvalidArgument("OpKernel used ref input name '", name,
                                   "' when immutable input was expected");
  }
  *tensor = (*params_.inputs)[start].tensor;
  return Status::OK();
}

Status OpKernelContext::input_ref_mutex(const string& name, mutex** out_mutex) {
  int start, stop;
  TF_RETURN_IF_ERROR(params_.op_kernel->InputRange(name, &start, &stop));
  if (stop != start + 1) {
    return errors::InvalidArgument("OpKernel used list-valued input name '",
                                   name,
                                   "' when single-valued input was expected");
  }
  *out_mutex = input_ref_mutex(start);
  return Status::OK();
}

Status OpKernelContext::mutable_input(const string& name, Tensor* tensor,
                                      bool lock_held) {
  int start, stop;
  TF_RETURN_IF_ERROR(params_.op_kernel->InputRange(name, &start, &stop));
  if (stop != start + 1) {
    return errors::InvalidArgument("OpKernel used list-valued input name '",
                                   name,
                                   "' when single-valued input was expected");
  }
  if (!(*params_.inputs)[start].is_ref()) {
    return errors::InvalidArgument("OpKernel used immutable input name '", name,
                                   "' when ref input was expected");
  }
  // return a copy of the Ref acquired while holding the mutex
  if (lock_held) {
    *tensor = *(*params_.inputs)[start].tensor;
  } else {
    mutex_lock l(*input_ref_mutex(start));
    *tensor = *(*params_.inputs)[start].tensor;
  }
  return Status::OK();
}

Status OpKernelContext::replace_ref_input(const string& name,
                                          const Tensor& tensor,
                                          bool lock_held) {
  int start, stop;
  TF_RETURN_IF_ERROR(params_.op_kernel->InputRange(name, &start, &stop));
  if (stop != start + 1) {
    return errors::InvalidArgument("OpKernel used list-valued input name '",
                                   name,
                                   "' when single-valued input was expected");
  }
  if (!(*params_.inputs)[start].is_ref()) {
    return errors::InvalidArgument("OpKernel used immutable input name '", name,
                                   "' when ref input was expected");
  }
  replace_ref_input(start, tensor, lock_held);
  return Status::OK();
}

Status OpKernelContext::input_list(const string& name,
                                   OpInputList* list) const {
  int start, stop;
  TF_RETURN_IF_ERROR(params_.op_kernel->InputRange(name, &start, &stop));
  *list = OpInputList(this, start, stop);
  return Status::OK();
}

Status OpKernelContext::mutable_input_list(const string& name,
                                           OpMutableInputList* list) {
  int start, stop;
  TF_RETURN_IF_ERROR(params_.op_kernel->InputRange(name, &start, &stop));
  *list = OpMutableInputList(this, start, stop);
  return Status::OK();
}

Status OpKernelContext::output_list(const string& name, OpOutputList* list) {
  int start, stop;
  TF_RETURN_IF_ERROR(params_.op_kernel->OutputRange(name, &start, &stop));
  *list = OpOutputList(this, start, stop);
  return Status::OK();
}

Status OpKernelContext::allocate_output(const string& name,
                                        const TensorShape& shape,
                                        Tensor** tensor) {
  int start, stop;
  TF_RETURN_IF_ERROR(params_.op_kernel->OutputRange(name, &start, &stop));
  if (stop != start + 1) {
    return errors::InvalidArgument("OpKernel used list-valued output name '",
                                   name,
                                   "' when single-valued output was "
                                   "expected");
  }
  return allocate_output(start, shape, tensor);
}

Status OpKernelContext::allocate_output(const string& name,
                                        const TensorShape& shape,
                                        Tensor** tensor,
                                        AllocatorAttributes attr) {
  int start, stop;
  TF_RETURN_IF_ERROR(params_.op_kernel->OutputRange(name, &start, &stop));
  if (stop != start + 1) {
    return errors::InvalidArgument("OpKernel used list-valued output name '",
                                   name,
                                   "' when single-valued output was "
                                   "expected");
  }
  return allocate_output(start, shape, tensor, attr);
}

Status OpKernelContext::set_output(const string& name, const Tensor& tensor) {
  int start, stop;
  TF_RETURN_IF_ERROR(params_.op_kernel->OutputRange(name, &start, &stop));
  if (stop != start + 1) {
    return errors::InvalidArgument("OpKernel used list-valued output name '",
                                   name,
                                   "' when single-valued output was "
                                   "expected");
  }
  set_output(start, tensor);
  return Status::OK();
}

Status OpKernelContext::set_output_ref(const string& name, mutex* mu,
                                       Tensor* tensor_for_ref) {
  int start, stop;
  TF_RETURN_IF_ERROR(params_.op_kernel->OutputRange(name, &start, &stop));
  if (stop != start + 1) {
    return errors::InvalidArgument("OpKernel used list-valued output name '",
                                   name,
                                   "' when single-valued output was "
                                   "expected");
  }
  set_output_ref(start, mu, tensor_for_ref);
  return Status::OK();
}

Status OpKernelContext::mutable_output(const string& name, Tensor** tensor) {
  int start, stop;
  TF_RETURN_IF_ERROR(params_.op_kernel->OutputRange(name, &start, &stop));
  if (stop != start + 1) {
    return errors::InvalidArgument("OpKernel used list-valued output name '",
                                   name,
                                   "' when single-valued output was "
                                   "expected");
  }
  *tensor = mutable_output(start);
  return Status::OK();
}

Status OpKernelContext::release_output(const string& name, TensorValue* value) {
  int start, stop;
  TF_RETURN_IF_ERROR(params_.op_kernel->OutputRange(name, &start, &stop));
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
  const auto& inputs = *params_.inputs;
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
  for (const TensorValue& t : *params_.inputs) {
    inputs.push_back(t.is_ref() ? MakeRefType(t->dtype()) : t->dtype());
  }
  DataTypeVector outputs = params_.op_kernel->output_types();
  return MatchSignatureHelper(expected_inputs, expected_outputs, inputs,
                              outputs);
}

// OpKernel registration ------------------------------------------------------

struct KernelRegistration {
  KernelRegistration(const KernelDef& d,
                     kernel_factory::OpKernelRegistrar::Factory f)
      : def(d), factory(f) {}
  const KernelDef def;
  const kernel_factory::OpKernelRegistrar::Factory factory;
};

// This maps from 'op_type' + DeviceType to the set of KernelDefs and
// factory functions for instantiating the OpKernel that matches the
// KernelDef.
typedef std::unordered_multimap<string, KernelRegistration> KernelRegistry;

static KernelRegistry* GlobalKernelRegistry() {
  static KernelRegistry* global_kernel_registry = new KernelRegistry;
  return global_kernel_registry;
}

static string Key(const string& op_type, DeviceType device_type,
                  const string& label) {
  return strings::StrCat(op_type, ":", DeviceTypeString(device_type), ":",
                         label);
}

namespace kernel_factory {

OpKernelRegistrar::OpKernelRegistrar(const KernelDef* kernel_def,
                                     Factory factory) {
  const string key =
      Key(kernel_def->op(), DeviceType(kernel_def->device_type()),
          kernel_def->label());
  GlobalKernelRegistry()->insert(
      std::make_pair(key, KernelRegistration(*kernel_def, factory)));
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

// Returns whether the attrs in the NodeDef satisfy the constraints in
// the kernel_def.  Returns an error if attrs in kernel_def are not
// found, or have a mismatching type.
Status AttrsMatch(const NodeDef& node_def, const KernelDef& kernel_def,
                  bool* match) {
  *match = false;
  AttrSlice attrs(node_def);
  for (const auto& constraint : kernel_def.constraint()) {
    if (constraint.allowed_values().list().type_size() == 0) {
      return errors::Unimplemented(
          "KernelDef '", kernel_def.ShortDebugString(),
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
              "KernelDef '", kernel_def.ShortDebugString(),
              "' has constraint on attr '", constraint.name(),
              "' that has value '", SummarizeAttrValue(*found),
              "' that does not have type 'type' or 'list(type)' in NodeDef '",
              SummarizeNodeDef(node_def), "'");
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
          constraint.name(), "' not in NodeDef '", SummarizeNodeDef(node_def),
          "', KernelDef: '", kernel_def.ShortDebugString(), "'");
    }
  }
  *match = true;
  return Status::OK();
}

Status FindKernelRegistration(DeviceType device_type, const NodeDef& node_def,
                              const KernelRegistration** reg) {
  *reg = nullptr;
  string label;  // Label defaults to empty if not found in NodeDef.
  GetNodeAttr(node_def, "_kernel", &label);
  const string key = Key(node_def.op(), device_type, label);
  auto regs = GlobalKernelRegistry()->equal_range(key);
  for (auto iter = regs.first; iter != regs.second; ++iter) {
    // If there is a kernel registered for the op and device_type,
    // check that the attrs match.
    bool match;
    TF_RETURN_IF_ERROR(AttrsMatch(node_def, iter->second.def, &match));
    if (match) {
      if (*reg != nullptr) {
        return errors::InvalidArgument(
            "Multiple OpKernel registrations match NodeDef '",
            SummarizeNodeDef(node_def), "': '", (*reg)->def.ShortDebugString(),
            "' and '", iter->second.def.ShortDebugString(), "'");
      }
      *reg = &iter->second;
    }
  }
  return Status::OK();
}

}  // namespace

Status SupportedDeviceTypesForNode(
    const std::vector<DeviceType>& prioritized_types, const NodeDef& def,
    DeviceTypeVector* device_types) {
  // TODO(zhifengc): Changes the callers (SimplePlacer and
  // DynamicPlacer) to consider the possibility that 'def' is call to
  // a user-defined function and only calls this
  // SupportedDeviceTypesForNode for primitive ops.
  Status s;
  const OpDef* op_def = OpRegistry::Global()->LookUp(def.op(), &s);
  if (op_def) {
    for (const DeviceType& device_type : prioritized_types) {
      const KernelRegistration* reg = nullptr;
      TF_RETURN_IF_ERROR(FindKernelRegistration(device_type, def, &reg));
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

std::unique_ptr<OpKernel> CreateOpKernel(DeviceType device_type,
                                         DeviceBase* device,
                                         Allocator* allocator,
                                         const NodeDef& node_def,
                                         Status* status) {
  OpKernel* kernel = nullptr;
  *status = CreateOpKernel(device_type, device, allocator, nullptr, node_def,
                           &kernel);
  return std::unique_ptr<OpKernel>(kernel);
}

Status CreateOpKernel(DeviceType device_type, DeviceBase* device,
                      Allocator* allocator, FunctionLibraryRuntime* flib,
                      const NodeDef& node_def, OpKernel** kernel) {
  VLOG(1) << "Instantiating kernel for node: " << SummarizeNodeDef(node_def);

  // Look up the Op registered for this op name.
  Status s;
  const OpDef* op_def = OpRegistry::Global()->LookUp(node_def.op(), &s);
  if (op_def == nullptr) return s;

  // Validate node_def against OpDef.
  s = ValidateNodeDef(node_def, *op_def);
  if (!s.ok()) return s;

  // Look up kernel registration.
  const KernelRegistration* registration;
  s = FindKernelRegistration(device_type, node_def, &registration);
  if (!s.ok()) {
    errors::AppendToMessage(&s, " when instantiating ", node_def.op());
    return s;
  }
  if (registration == nullptr) {
    s.Update(errors::NotFound("No registered '", node_def.op(),
                              "' OpKernel for ", DeviceTypeString(device_type),
                              " devices compatible with node ",
                              SummarizeNodeDef(node_def)));
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

  // Everything needed for OpKernel construction.
  OpKernelConstruction context(device_type, device, allocator, &node_def,
                               op_def, flib, inputs, outputs, &s);
  *kernel = (*registration->factory)(&context);
  if (!s.ok()) {
    delete *kernel;
    *kernel = nullptr;
  }
  return s;
}

namespace {  // Helper for MemoryTypesForNode.
// Fills memory_types for either input or output, setting everything
// to DEVICE_MEMORY except those args in host_memory_args.  Removes
// elements of host_memory_args that were used.
void MemoryTypesHelper(const NameRangeMap& name_map,
                       std::vector<string>* host_memory_args,
                       MemoryTypeVector* memory_types) {
  // Set total to the largest endpoint of anything in the name_map.
  int total = 0;
  for (const auto& item : name_map) {
    total = std::max(total, item.second.second);
  }

  // Now that we know the size, fill with the default 'DEVICE_MEMORY'.
  memory_types->clear();
  memory_types->resize(total, DEVICE_MEMORY);

  // Update args that have been marked as in "HOST_MEMORY".
  size_t keep = 0;
  for (size_t i = 0; i < host_memory_args->size(); ++i) {
    auto iter = name_map.find((*host_memory_args)[i]);
    if (iter != name_map.end()) {
      for (int j = iter->second.first; j < iter->second.second; ++j) {
        (*memory_types)[j] = HOST_MEMORY;
      }
    } else {
      // (*host_memory_args)[i] not found, save it for the next pass.
      if (i > keep) (*host_memory_args)[keep] = (*host_memory_args)[i];
      ++keep;
    }
  }
  host_memory_args->resize(keep);
}
}  // namespace

Status MemoryTypesForNode(DeviceType device_type, const NodeDef& ndef,
                          const OpDef& op_def,
                          const NameRangeMap& input_name_map,
                          const NameRangeMap& output_name_map,
                          MemoryTypeVector* input_memory_types,
                          MemoryTypeVector* output_memory_types) {
  Status status;
  const KernelRegistration* registration;
  TF_RETURN_IF_ERROR(FindKernelRegistration(device_type, ndef, &registration));

  if (registration != nullptr) {
    const auto& from_proto = registration->def.host_memory_arg();
    std::vector<string> host_memory_args(from_proto.begin(), from_proto.end());
    MemoryTypesHelper(input_name_map, &host_memory_args, input_memory_types);
    MemoryTypesHelper(output_name_map, &host_memory_args, output_memory_types);
    if (!host_memory_args.empty()) {
      return errors::InvalidArgument(
          "HostMemory args '", str_util::Join(host_memory_args, "', '"),
          "' not found in OpDef: ", SummarizeOpDef(op_def));
    }
  }
  return status;
}

Status MemoryTypesForNode(const OpRegistryInterface* op_registry,
                          DeviceType device_type, const NodeDef& ndef,
                          MemoryTypeVector* input_memory_types,
                          MemoryTypeVector* output_memory_types) {
  // Look up the Op registered for this op name.
  Status status;
  const OpDef* op_def = op_registry->LookUp(ndef.op(), &status);
  if (op_def == nullptr) return status;

  NameRangeMap inputs, outputs;
  status = NameRangesForNode(ndef, *op_def, &inputs, &outputs);
  if (!status.ok()) return status;

  return MemoryTypesForNode(device_type, ndef, *op_def, inputs, outputs,
                            input_memory_types, output_memory_types);
}

namespace {

bool FindArgInOp(const string& arg_name,
                 const protobuf::RepeatedPtrField<OpDef::ArgDef>& args) {
  for (const auto& arg : args) {
    if (arg_name == arg.name()) {
      return true;
    }
  }
  return false;
}

}  // namespace

Status ValidateKernelRegistrations(const OpRegistryInterface* op_registry) {
  Status unused_status;
  for (const auto& key_registration : *GlobalKernelRegistry()) {
    const KernelDef& kernel_def(key_registration.second.def);
    const OpDef* op_def = op_registry->LookUp(kernel_def.op(), &unused_status);
    if (op_def == nullptr) {
      // TODO(josh11b): Make this a hard error.
      LOG(ERROR) << "OpKernel ('" << kernel_def.ShortDebugString()
                 << "') for unknown op: " << kernel_def.op();
      continue;
    }
    for (const auto& host_memory_arg : kernel_def.host_memory_arg()) {
      if (!FindArgInOp(host_memory_arg, op_def->input_arg()) &&
          !FindArgInOp(host_memory_arg, op_def->output_arg())) {
        return errors::InvalidArgument("HostMemory arg '", host_memory_arg,
                                       "' not found in OpDef: ",
                                       SummarizeOpDef(*op_def));
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

}  // namespace tensorflow
