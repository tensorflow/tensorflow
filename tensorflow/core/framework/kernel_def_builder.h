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

#ifndef TENSORFLOW_FRAMEWORK_KERNEL_DEF_BUILDER_H_
#define TENSORFLOW_FRAMEWORK_KERNEL_DEF_BUILDER_H_

#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

// Forward declare proto so that kernels don't need to depend on it
class KernelDef;

// Builder class passed to the REGISTER_KERNEL_BUILDER() macro.
class KernelDefBuilder {
 public:
  // Starts with just the name field set.
  // Caller MUST call Build() and take ownership of the result.
  explicit KernelDefBuilder(const char* op_name);
  ~KernelDefBuilder();

  // Required: specify the type of device this kernel supports.
  // Returns *this.
  KernelDefBuilder& Device(const char* device_type);
  //  KernelDefBuilder& Device(DeviceType device_type);

  // Specify that this kernel supports a limited set of values for a
  // particular type or list(type) attr (a further restriction than
  // what the Op allows).
  // Returns *this.
  KernelDefBuilder& TypeConstraint(const char* attr_name,
                                   gtl::ArraySlice<DataType> allowed);

  // Like TypeConstraint but supports just a single type.
  KernelDefBuilder& TypeConstraint(const char* attr_name, DataType allowed);

  // Like TypeConstraint, but (a) gets the type from a template parameter
  // and (b) only supports a constraint to a single type.
  template <class T>
  KernelDefBuilder& TypeConstraint(const char* attr_name);
  // TODO(josh11b): Support other types of attr constraints as needed.

  // Specify that this kernel requires/provides an input/output arg
  // in host memory (instead of the default, device memory).
  // Returns *this.
  KernelDefBuilder& HostMemory(const char* arg_name);

  // Specify that this kernel requires a particular value for the
  // "_kernel" attr.  May only be specified once.  Returns *this.
  KernelDefBuilder& Label(const char* label);

  // Returns a pointer to a KernelDef with fields set based on the
  // above calls to this instance.
  // Caller takes ownership of the result.
  const KernelDef* Build();

 private:
  KernelDef* kernel_def_;

  TF_DISALLOW_COPY_AND_ASSIGN(KernelDefBuilder);
};

// IMPLEMENTATION

template <class T>
KernelDefBuilder& KernelDefBuilder::TypeConstraint(const char* attr_name) {
  return this->TypeConstraint(attr_name, DataTypeToEnum<T>::v());
}

}  // namespace tensorflow

#endif  // TENSORFLOW_FRAMEWORK_KERNEL_DEF_BUILDER_H_
