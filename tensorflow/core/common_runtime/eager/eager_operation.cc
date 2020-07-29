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
#include "tensorflow/core/common_runtime/eager/eager_operation.h"

#include "absl/types/span.h"
#include "tensorflow/c/eager/abstract_operation.h"
#include "tensorflow/c/eager/abstract_tensor_handle.h"
#include "tensorflow/c/eager/immediate_execution_tensor_handle.h"
#include "tensorflow/c/tf_tensor_internal.h"
#include "tensorflow/core/common_runtime/eager/attr_builder.h"
#include "tensorflow/core/common_runtime/input_colocation_exemption_registry.h"
#include "tensorflow/core/platform/casts.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/host_info.h"

namespace tensorflow {

// An EagerOperation object can be reused for a different op by calling
// Clear(), and then Reset(...) with the same arguments that would have
// been provided to the constructor.
void EagerOperation::Clear() {
  for (TensorHandle* h : inputs_) {
    h->Unref();
  }
  inputs_.clear();
  ClearInferenceState();
}

Status EagerOperation::SetAttrValue(const char* attr_name,
                                    const AttrValue& value) {
  MutableAttrs()->Set(attr_name, value);
  return Status::OK();
}

Status EagerOperation::SetAttrString(const char* attr_name, const char* data,
                                     size_t length) {
  MutableAttrs()->Set(attr_name, StringPiece(data, length));
  return Status::OK();
}

Status EagerOperation::SetAttrInt(const char* attr_name, int64_t value) {
  MutableAttrs()->Set(attr_name, static_cast<int64>(value));
  return Status::OK();
}

Status EagerOperation::SetAttrFloat(const char* attr_name, float value) {
  MutableAttrs()->Set(attr_name, value);
  return Status::OK();
}

Status EagerOperation::SetAttrBool(const char* attr_name, bool value) {
  MutableAttrs()->Set(attr_name, value);
  return Status::OK();
}

Status EagerOperation::SetAttrType(const char* attr_name, DataType value) {
  MutableAttrs()->Set(attr_name, value);
  return Status::OK();
}

Status EagerOperation::SetAttrShape(const char* attr_name, const int64_t* dims,
                                    const int num_dims) {
  if (num_dims > TensorShape::MaxDimensions()) {
    return errors::InvalidArgument("Value specified for `", attr_name, "` has ",
                                   num_dims,
                                   " dimensions which is over the limit of ",
                                   TensorShape::MaxDimensions(), ".");
  }

  TensorShapeProto proto;
  if (num_dims < 0) {
    proto.set_unknown_rank(true);
  } else {
    for (int d = 0; d < num_dims; ++d) {
      proto.add_dim()->set_size(dims[d]);
    }
  }

  MutableAttrs()->Set(attr_name, proto);

  return Status::OK();
}

Status EagerOperation::SetAttrFunction(const char* attr_name,
                                       const AbstractOperation* value) {
  AttrValue attr_value;
  NameAttrList* func = attr_value.mutable_func();
  func->set_name(value->Name());
  auto* value_operation = down_cast<const EagerOperation*>(value);
  value_operation->Attrs().FillAttrValueMap(func->mutable_attr());
  MutableAttrs()->Set(attr_name, attr_value);
  return Status::OK();
}

Status EagerOperation::SetAttrFunctionName(const char* attr_name,
                                           const char* data, size_t length) {
  AttrValue attr_value;
  NameAttrList* func = attr_value.mutable_func();
  func->set_name(data, length);
  MutableAttrs()->Set(attr_name, attr_value);
  return Status::OK();
}

Status EagerOperation::SetAttrTensor(const char* attr_name,
                                     AbstractTensorInterface* tensor) {
  Tensor t = TensorFromInterface(tensor);
  MutableAttrs()->Set(attr_name, t);
  return Status::OK();
}

Status EagerOperation::SetAttrStringList(const char* attr_name,
                                         const void* const* values,
                                         const size_t* lengths,
                                         int num_values) {
  std::vector<StringPiece> v(num_values);
  for (int i = 0; i < num_values; ++i) {
    v[i] = StringPiece(static_cast<const char*>(values[i]), lengths[i]);
  }
  MutableAttrs()->Set(attr_name, v);

  return Status::OK();
}

Status EagerOperation::SetAttrFloatList(const char* attr_name,
                                        const float* values, int num_values) {
  MutableAttrs()->Set(attr_name,
                      gtl::ArraySlice<const float>(values, num_values));
  return Status::OK();
}

Status EagerOperation::SetAttrIntList(const char* attr_name,
                                      const int64_t* values, int num_values) {
  MutableAttrs()->Set(attr_name,
                      gtl::ArraySlice<const int64>(
                          reinterpret_cast<const int64*>(values), num_values));
  return Status::OK();
}

Status EagerOperation::SetAttrTypeList(const char* attr_name,
                                       const DataType* values, int num_values) {
  MutableAttrs()->Set(attr_name,
                      gtl::ArraySlice<const DataType>(values, num_values));
  return Status::OK();
}

Status EagerOperation::SetAttrBoolList(const char* attr_name,
                                       const unsigned char* values,
                                       int num_values) {
  std::unique_ptr<bool[]> b(new bool[num_values]);
  for (int i = 0; i < num_values; ++i) {
    b[i] = values[i];
  }
  MutableAttrs()->Set(attr_name,
                      gtl::ArraySlice<const bool>(b.get(), num_values));
  return Status::OK();
}

Status EagerOperation::SetAttrShapeList(const char* attr_name,
                                        const int64_t** dims,
                                        const int* num_dims, int num_values) {
  std::unique_ptr<TensorShapeProto[]> proto(new TensorShapeProto[num_values]);
  for (int i = 0; i < num_values; ++i) {
    const auto num_dims_i = num_dims[i];

    if (num_dims_i > TensorShape::MaxDimensions()) {
      return errors::InvalidArgument(
          strings::StrCat("Value specified for `", attr_name, "` has ",
                          num_dims_i, " dimensions which is over the limit of ",
                          TensorShape::MaxDimensions(), "."));
    }
    if (num_dims_i < 0) {
      proto[i].set_unknown_rank(true);
    } else {
      const int64_t* dims_i = dims[i];
      auto proto_i = &proto[i];
      for (int d = 0; d < num_dims_i; ++d) {
        proto_i->add_dim()->set_size(dims_i[d]);
      }
    }
  }
  MutableAttrs()->Set(
      attr_name, gtl::ArraySlice<TensorShapeProto>(proto.get(), num_values));
  return Status::OK();
}

Status EagerOperation::SetAttrFunctionList(
    const char* attr_name, absl::Span<const AbstractOperation*> values) {
  size_t num_values = values.size();
  std::unique_ptr<NameAttrList[]> funcs(new NameAttrList[num_values]);
  for (int i = 0; i < num_values; i++) {
    auto* value_operation = down_cast<const EagerOperation*>(values[i]);
    funcs[i].set_name(value_operation->Name());
    value_operation->Attrs().FillAttrValueMap(funcs[i].mutable_attr());
  }
  MutableAttrs()->Set(
      attr_name, gtl::ArraySlice<const NameAttrList>(funcs.get(), num_values));
  return Status::OK();
}

const OpDef* EagerOperation::GetOpDef(Status* status) {
  const tensorflow::OpDef* op_def = OpDef();
  if (op_def) return op_def;
  *status = OpDefForOp(Name(), &op_def);
  return op_def;
}

Status EagerOperation::InputLength(const char* input_name, int* length) {
  Status status;
  const tensorflow::OpDef* op_def = GetOpDef(&status);
  if (!status.ok()) {
    return status;
  }
  AttrValueMap attrs;
  Attrs().FillAttrValueMap(&attrs);
  NameRangeMap name_ranges;
  TF_RETURN_IF_ERROR(
      NameRangesForNode(AttrSlice(&attrs), *op_def, &name_ranges, nullptr));
  auto iter = name_ranges.find(input_name);
  if (iter == name_ranges.end()) {
    return errors::InvalidArgument("Input '", input_name, "' not found");
  }
  *length = iter->second.second - iter->second.first;
  return Status::OK();
}

Status EagerOperation::OutputLength(const char* output_name, int* length) {
  Status status;
  const tensorflow::OpDef* op_def = GetOpDef(&status);
  if (!status.ok()) {
    return status;
  }
  AttrValueMap attrs;
  Attrs().FillAttrValueMap(&attrs);
  NameRangeMap name_ranges;
  TF_RETURN_IF_ERROR(
      NameRangesForNode(AttrSlice(&attrs), *op_def, nullptr, &name_ranges));
  auto iter = name_ranges.find(output_name);
  if (iter == name_ranges.end()) {
    return errors::InvalidArgument("Output '", output_name, "' not found");
  }
  *length = iter->second.second - iter->second.first;
  return Status::OK();
}

Status EagerOperation::AddInput(AbstractTensorHandle* input) {
  TensorHandle* h = TensorHandleFromInterface(input);
  AddTensorHandle(h);
  return MaybeInferSingleInputAttrs(h);
}

Status EagerOperation::AddInputList(absl::Span<AbstractTensorHandle*> inputs) {
  for (auto& input : inputs) {
    TensorHandle* h = TensorHandleFromInterface(input);
    AddTensorHandle(h);
  }
  return InferInputListAttrs(inputs.size());
}

Status EagerOperation::SetUseXla(bool enable) {
  use_xla_ = enable;
  return Status::OK();
}

Status EagerOperation::Reset(
    const char* op, const char* device_name, bool remote,
    EagerExecutor* executor,
    const absl::optional<EagerRemoteFunctionParams> remote_func_params) {
  DCHECK(inputs_.empty());
  ClearInferenceState();
  bool is_function = false;
  TF_RETURN_IF_ERROR(AttrTypeMapForOp(op, &attr_types_, &is_function));

  // Don't update the device of direct function calls.
  // Particularly, if the user did not explicitly request any device for this
  // function, picking a device would result in this device being the default
  // for nodes inside the function. This is undesirable for multi-device
  // functions since the not-explicitly-placed nodes inside the body will all
  // end up on this default device.
  colocation_exempt_ = is_function;
  if (!is_function) {
    const auto& exempt_ops = InputColocationExemptionRegistry::Global()->Get();
    colocation_exempt_ = exempt_ops.find(op) != exempt_ops.end();

    TF_RETURN_IF_ERROR(OpDefForOp(op, &op_def_));
  } else if (!remote && !ctx_.FindFunctionByName(op)) {
    return errors::NotFound(
        "'", op,
        "' is neither a type of a primitive operation nor a name "
        "of a function registered in binary running on ",
        port::Hostname(),
        ". Make sure the operation or function is "
        "registered in the binary running in this process.");
  }
  attrs_.Reset(op);
  use_xla_ = false;
  is_function_ = is_function;
  cancellation_manager_ = nullptr;
  executor_ = executor ? executor : &ctx_.Executor();
  remote_func_params_ = remote_func_params;
  op_name_ = op;
  return SetDeviceName(device_name);
}

Status EagerOperation::MaybeInferSingleInputAttrs(TensorHandle* handle) {
  if (!op_def_) return Status::OK();

  const auto& input_def = op_def_->input_arg(inference_arg_idx_++);
  if (!input_def.number_attr().empty() || !input_def.type_list_attr().empty()) {
    // Some clients that are still setting their input attributes manually are
    // adding input list to their op by calling `TFE_OpAddInput` for each of
    // its elements instead of calling `TFE_OpAddInputList`. When this happens,
    // we cannot detect the end of such list, thus lose track of the input
    // arguments in the op definition. To guarantee backward compatibility with
    // those clients, disable automatic inference in this case.
    ClearInferenceState();
    return Status::OK();
  }
  const std::string& type_attr = input_def.type_attr();
  if (!type_attr.empty() &&
      inference_attrs_.find(type_attr) == inference_attrs_.end()) {
    MutableAttrs()->Set(type_attr, handle->dtype);
    inference_attrs_.insert(type_attr);
  }
  return Status::OK();
}

void EagerOperation::InferSingleTypeInputListAttrs(
    const OpDef::ArgDef& input_def, const DataType dtype, int num_inputs) {
  if (inference_attrs_.find(input_def.number_attr()) ==
      inference_attrs_.end()) {
    MutableAttrs()->Set(input_def.number_attr(), num_inputs);
    inference_attrs_.insert(input_def.number_attr());
  }
  if (inference_attrs_.find(input_def.type_attr()) == inference_attrs_.end()) {
    MutableAttrs()->Set(input_def.type_attr(), dtype);
    inference_attrs_.insert(input_def.type_attr());
  }
}

void EagerOperation::InferMixedTypeInputListAttrs(
    const OpDef::ArgDef& input_def, const std::vector<DataType>& dtypes) {
  if (inference_attrs_.find(input_def.type_list_attr()) ==
      inference_attrs_.end()) {
    MutableAttrs()->Set(
        input_def.type_list_attr(),
        gtl::ArraySlice<const DataType>(dtypes.data(), dtypes.size()));
    inference_attrs_.insert(input_def.type_list_attr());
  }
}

Status EagerOperation::InferInputListAttrs(int num_inputs) {
  if (!op_def_) return Status::OK();

  int start = inference_arg_idx_;
  const auto& input_def = op_def_->input_arg(inference_arg_idx_++);
  if (!input_def.type_list_attr().empty()) {
    std::vector<DataType> dtypes(num_inputs);
    for (int i = 0; i < num_inputs; ++i) {
      dtypes[i] = inputs_[start + i]->dtype;
    }
    InferMixedTypeInputListAttrs(input_def, dtypes);
  } else if (!input_def.type_attr().empty() &&
             !input_def.number_attr().empty()) {
    InferSingleTypeInputListAttrs(input_def, inputs_[start]->dtype, num_inputs);
  } else {
    return errors::InvalidArgument("Invalid input list definition");
  }
  return Status::OK();
}

Status EagerOperation::SetDeviceName(const char* c_name) {
  string name(c_name != nullptr ? c_name : "");
  if (name != last_set_device_name_) {
    if (!DeviceNameUtils::ParseFullName(name, &device_parsed_name_)) {
      return errors::InvalidArgument("Malformed device specification '", name,
                                     "' in eager op: ", DebugString());
    }
    last_set_device_name_ = name;
    device_name_ = DeviceNameUtils::ParsedNameToString(device_parsed_name_);
    CustomDevice* custom_device;
    if (ctx_.FindCustomDeviceFromName(device_name_, &custom_device).ok()) {
      device_ = custom_device;
    } else {
      // Device placement for physical devices happens lazily in
      // EagerExecute/EagerRemoteExecute, and can depend on the inputs.
      device_ = kVariantDeviceNull;
    }
  }
  return Status::OK();
}

bool EagerOperation::IsLocal() const {
  if (ctx_.remote_device_mgr() == nullptr) return true;

  if (!device_parsed_name_.has_job && !device_parsed_name_.has_replica &&
      !device_parsed_name_.has_task)
    return true;
  auto& host_cpu_name = ctx_.HostCPU()->parsed_name();
  return device_parsed_name_.job == host_cpu_name.job &&
         device_parsed_name_.replica == host_cpu_name.replica &&
         device_parsed_name_.task == host_cpu_name.task;
}

string EagerOperation::DebugString() const {
  string out;
  VLOG(1) << "EagerOperation::DebugString() over " << this;

  strings::StrAppend(&out, "Name: ", Name(), "\n");
  strings::StrAppend(&out, "Device Name: [", device_name_, "]\n");
  strings::StrAppend(&out, "Device: ", VariantDeviceDebugString(Device()),
                     "\n");
  for (const auto& input : inputs_) {
    VLOG(1) << "Input ptr: " << input;
    strings::StrAppend(&out, "Input: ", input->DebugString(), "\n");
  }

  NodeDef ndef;
  Attrs().FillAttrValueMap(ndef.mutable_attr());
  strings::StrAppend(&out, "Attrs: ", ndef.DebugString(), "\n");
  return out;
}

void EagerOperation::AddTensorHandle(TensorHandle* h) {
  h->Ref();
  inputs_.push_back(h);
  attrs_.NumInputs(static_cast<int>(inputs_.size()));
}

}  // namespace tensorflow
