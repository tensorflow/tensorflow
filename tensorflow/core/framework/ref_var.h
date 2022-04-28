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

#ifndef TENSORFLOW_CORE_FRAMEWORK_REF_VAR_H_
#define TENSORFLOW_CORE_FRAMEWORK_REF_VAR_H_

#include <functional>

namespace tensorflow {
class OpKernelContext;

void AssignRefVariable(
    OpKernelContext* context, int input_ref_index, int output_ref_index,
    int value_index, bool use_locking, bool validate_shape,
    bool relax_constraints,
    std::function<void(OpKernelContext*, Tensor*, const Tensor&)> copy);
}  //  end namespace tensorflow

#endif  // TENSORFLOW_CORE_FRAMEWORK_REF_VAR_H_
