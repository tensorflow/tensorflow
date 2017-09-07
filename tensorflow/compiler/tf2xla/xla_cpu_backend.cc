/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/core/framework/kernel_def.pb.h"

namespace tensorflow {

bool CpuOpFilter(KernelDef* kdef) {
  // TODO(b/34339814): implement inverse erf for double types and remove this
  // workaround.
  if (kdef->op() == "RandomStandardNormal") {
    kdef->clear_constraint();
    // Change the type constraint to permit only DTD_FLOAT.
    KernelDef::AttrConstraint* attr_constraint = kdef->add_constraint();
    attr_constraint->set_name("dtype");
    attr_constraint->mutable_allowed_values()->mutable_list()->add_type(
        DT_FLOAT);
    return true;
  }
  return true;
}

REGISTER_XLA_BACKEND(DEVICE_CPU_XLA_JIT, kCpuAllTypes, CpuOpFilter);

}  // namespace tensorflow
