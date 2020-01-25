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

#include "tensorflow/core/common_runtime/eager/attr_builder.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/host_info.h"

namespace tensorflow {

Status EagerOperation::Reset(
    const char* op, const char* raw_device_name, bool remote,
    EagerExecutor* executor,
    const absl::optional<EagerRemoteFunctionParams> remote_func_params) {
  DCHECK(inputs_.empty());
  ClearInferenceState();
  bool is_function = false;
  TF_RETURN_IF_ERROR(AttrTypeMapForOp(op, &attr_types_, &is_function));

  if (!is_function) {
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
  device_ = nullptr;
  use_xla_ = false;
  is_function_ = is_function;
  cancellation_manager_ = nullptr;
  executor_ = executor ? executor : &ctx_.Executor();
  remote_func_params_ = remote_func_params;
#ifdef TENSORFLOW_MEM_DEBUG
  op_name_ = op;
#endif
  return SetDeviceName(raw_device_name, true);
}

tensorflow::Status EagerOperation::MaybeInferSingleInputAttrs(
    TensorHandle* handle) {
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
    const tensorflow::OpDef::ArgDef& input_def,
    const tensorflow::DataType dtype, int num_inputs) {
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
    const tensorflow::OpDef::ArgDef& input_def,
    const std::vector<tensorflow::DataType>& dtypes) {
  if (inference_attrs_.find(input_def.type_list_attr()) ==
      inference_attrs_.end()) {
    MutableAttrs()->Set(input_def.type_list_attr(),
                        tensorflow::gtl::ArraySlice<const tensorflow::DataType>(
                            dtypes.data(), dtypes.size()));
    inference_attrs_.insert(input_def.type_list_attr());
  }
}

tensorflow::Status EagerOperation::InferInputListAttrs(int num_inputs) {
  if (!op_def_) return Status::OK();

  int start = inference_arg_idx_;
  const auto& input_def = op_def_->input_arg(inference_arg_idx_++);
  if (!input_def.type_list_attr().empty()) {
    std::vector<tensorflow::DataType> dtypes(num_inputs);
    for (int i = 0; i < num_inputs; ++i) {
      dtypes[i] = inputs_[start + i]->dtype;
    }
    InferMixedTypeInputListAttrs(input_def, dtypes);
  } else if (!input_def.type_attr().empty() &&
             !input_def.number_attr().empty()) {
    InferSingleTypeInputListAttrs(input_def, inputs_[start]->dtype, num_inputs);
  } else {
    return tensorflow::errors::InvalidArgument("Invalid input list definition");
  }
  return tensorflow::Status::OK();
}

tensorflow::Status EagerOperation::SetDeviceName(const char* device,
                                                 const bool reset) {
  if (device != nullptr && strlen(device) > 0) {
    if (device != raw_device_name_) {
      if (!DeviceNameUtils::ParseFullName(device, &device_parsed_name_)) {
        return errors::InvalidArgument("Malformed device specification '",
                                       device,
                                       "' in eager op: ", DebugString());
      }
      raw_device_name_ = device;
      device_name_ =
          DeviceNameUtils::HasSomeDetails(device_parsed_name_)
              ? DeviceNameUtils::ParsedNameToString(device_parsed_name_)
              : "";
    }
  } else if (reset) {
    raw_device_name_.clear();
    device_name_.clear();
    device_parsed_name_.Clear();
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
  strings::StrAppend(
      &out, "Device: ", Device() ? Device()->DebugString() : "[]", "\n");
  for (const auto& input : inputs_) {
    VLOG(1) << "Input ptr: " << input;
    strings::StrAppend(&out, "Input: ", input->DebugString(), "\n");
  }

  NodeDef ndef;
  Attrs().FillAttrValueMap(ndef.mutable_attr());
  strings::StrAppend(&out, "Attrs: ", ndef.DebugString(), "\n");
  return out;
}

}  // namespace tensorflow
