/* Copyright 2015-2017 The TensorFlow Authors. All Rights Reserved.

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

// See docs in ../ops/array_ops.cc.
#include "tensorflow/core/kernels/identity_n_op.h"

#include "tensorflow/core/common_runtime/input_colocation_exemption_registry.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"

namespace tensorflow {

REGISTER_KERNEL_BUILDER(Name("IdentityN").Device(DEVICE_DEFAULT), IdentityNOp);
// Do not worry about colocating IdentityN op with its resource inputs since
// it just forwards it's inputs anyway. This is needed because we create
// IdentityN nodes to club "all" outputs of functional ops while lowering to
// make the original functional op fetchable.
REGISTER_INPUT_COLOCATION_EXEMPTION("IdentityN");

}  // namespace tensorflow
