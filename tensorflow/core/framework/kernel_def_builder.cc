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

#include "tensorflow/core/framework/kernel_def_builder.h"

#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/kernel_def.pb.h"

namespace tensorflow {

KernelDefBuilder::KernelDefBuilder(const char* op_name) {
  kernel_def_ = new KernelDef;
  kernel_def_->set_op(op_name);
}

KernelDefBuilder::~KernelDefBuilder() {
  DCHECK(kernel_def_ == nullptr) << "Did not call Build()";
}

KernelDefBuilder& KernelDefBuilder::Device(const char* device_type) {
  kernel_def_->set_device_type(device_type);
  return *this;
}

template <>
KernelDefBuilder& KernelDefBuilder::AttrConstraint<int64>(
    const char* attr_name, gtl::ArraySlice<int64> allowed) {
  auto* constraint = kernel_def_->add_constraint();
  constraint->set_name(attr_name);
  auto* allowed_values = constraint->mutable_allowed_values()->mutable_list();
  for (const int64 integer : allowed) {
    allowed_values->add_i(integer);
  }
  return *this;
}

template <>
KernelDefBuilder& KernelDefBuilder::AttrConstraint<int64>(const char* attr_name,
                                                          int64 allowed) {
  return AttrConstraint(
      attr_name,
      gtl::ArraySlice<int64>(std::initializer_list<int64>({allowed})));
}

template <>
KernelDefBuilder& KernelDefBuilder::AttrConstraint<string>(
    const char* attr_name, gtl::ArraySlice<string> allowed) {
  auto* constraint = kernel_def_->add_constraint();
  constraint->set_name(attr_name);
  auto* allowed_values = constraint->mutable_allowed_values()->mutable_list();
  for (const auto& str : allowed) {
    allowed_values->add_s(str);
  }
  return *this;
}

template <>
KernelDefBuilder& KernelDefBuilder::AttrConstraint<string>(
    const char* attr_name, string allowed) {
  return AttrConstraint(
      attr_name,
      gtl::ArraySlice<string>(std::initializer_list<string>({allowed})));
}

template <>
KernelDefBuilder& KernelDefBuilder::AttrConstraint<const char*>(
    const char* attr_name, gtl::ArraySlice<const char*> allowed) {
  auto* constraint = kernel_def_->add_constraint();
  constraint->set_name(attr_name);
  auto* allowed_values = constraint->mutable_allowed_values()->mutable_list();
  for (const auto& str : allowed) {
    allowed_values->add_s(str);
  }
  return *this;
}

template <>
KernelDefBuilder& KernelDefBuilder::AttrConstraint<const char*>(
    const char* attr_name, const char* allowed) {
  return AttrConstraint(attr_name,
                        gtl::ArraySlice<const char*>(
                            std::initializer_list<const char*>({allowed})));
}

template <>
KernelDefBuilder& KernelDefBuilder::AttrConstraint<bool>(const char* attr_name,
                                                         bool allowed) {
  auto* constraint = kernel_def_->add_constraint();
  constraint->set_name(attr_name);
  auto* allowed_values = constraint->mutable_allowed_values()->mutable_list();
  allowed_values->add_b(allowed);
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
      << "' in: " << kernel_def_->DebugString();
  kernel_def_->set_label(label);
  return *this;
}

KernelDefBuilder& KernelDefBuilder::Priority(int32 priority) {
  kernel_def_->set_priority(priority);
  return *this;
}

const KernelDef* KernelDefBuilder::Build() {
  KernelDef* r = kernel_def_;
  kernel_def_ = nullptr;
  return r;
}

}  // namespace tensorflow
