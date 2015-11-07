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
      << "' in: " << kernel_def_->ShortDebugString();
  kernel_def_->set_label(label);
  return *this;
}

}  // namespace tensorflow
