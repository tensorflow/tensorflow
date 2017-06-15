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

#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/kernel_def.pb_text.h"
#include "tensorflow/core/framework/kernel_def_builder.h"

namespace tensorflow {

KernelDefBuilder::KernelDefBuilder(const char* op_name) {
  kernel_def_ = new KernelDef;
  kernel_def_->set_op(op_name);
}

KernelDefBuilder& KernelDefBuilder::Device(const char* device_type) {
  kernel_def_->set_device_type(device_type);
  return *this;
}

KernelDefBuilder& KernelDefBuilder::TypeConstraint(
    const char* attr_name, gtl::ArraySlice<DataType> allowed) {
  auto* constraint = kernel_def_->add_constraint();
  constraint->set_name(attr_name);
  auto* allowed_values = constraint->mutable_allowed_values()->mutable_list();
  for (DataType dt : allowed) {
    allowed_values->add_type(dt);
  }
  return *this;
}

KernelDefBuilder& KernelDefBuilder::TypeConstraint(const char* attr_name,
                                                   DataType allowed) {
  auto* constraint = kernel_def_->add_constraint();
  constraint->set_name(attr_name);
  constraint->mutable_allowed_values()->mutable_list()->add_type(allowed);
  return *this;
}

KernelDefBuilder& KernelDefBuilder::HostMemory(const char* arg_name) {
  kernel_def_->add_host_memory_arg(arg_name);
  return *this;
}

KernelDefBuilder& KernelDefBuilder::Label(const char* label) {
  CHECK_EQ(kernel_def_->label(), "")
      << "Trying to set a kernel's label a second time: '" << label
      << "' in: " << ProtoShortDebugString(*kernel_def_);
  kernel_def_->set_label(label);
  return *this;
}

}  // namespace tensorflow
