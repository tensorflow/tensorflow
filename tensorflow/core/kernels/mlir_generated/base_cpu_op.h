/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_KERNELS_MLIR_GENERATED_BASE_CPU_OP_H_
#define TENSORFLOW_CORE_KERNELS_MLIR_GENERATED_BASE_CPU_OP_H_

#include "tensorflow/core/kernels/mlir_generated/base_op.h"

namespace tensorflow {

#define GENERATE_AND_REGISTER_UNARY_CPU_KERNEL(tf_op, input_type) \
  GENERATE_AND_REGISTER_UNARY_KERNEL(tf_op, CPU, input_type)

#define GENERATE_UNARY_CPU_KERNEL(tf_op, input_type) \
  GENERATE_UNARY_KERNEL(tf_op, CPU, input_type)

#define GENERATE_UNARY_CPU_KERNEL2(tf_op, input_type, output_type) \
  GENERATE_UNARY_KERNEL2(tf_op, CPU, input_type, output_type)

#define REGISTER_ALIASED_CPU_KERNEL(tf_op, mlir_op, input_type, output_type) \
  REGISTER_ALIASED_KERNEL(tf_op, mlir_op, CPU, input_type, output_type)

#define REGISTER_CPU_KERNEL(tf_op, input_type, output_type) \
  REGISTER_KERNEL(tf_op, CPU, input_type, output_type)

#define REGISTER_COMPLEX_CPU_KERNEL(tf_op, input_type, output_type) \
  REGISTER_COMPLEX_KERNEL(tf_op, CPU, input_type, output_type)

#define REGISTER_CPU_KERNEL_NO_TYPE_CONSTRAINT(tf_op, input_type) \
  REGISTER_KERNEL_NO_TYPE_CONSTRAINT(tf_op, CPU, input_type)

#define GENERATE_AND_REGISTER_BINARY_CPU_KERNEL(tf_op, input_type) \
  GENERATE_AND_REGISTER_BINARY_KERNEL(tf_op, CPU, input_type)

#define GENERATE_AND_REGISTER_BINARY_CPU_KERNEL2(tf_op, input_type, \
                                                 output_type)       \
  GENERATE_AND_REGISTER_BINARY_KERNEL2(tf_op, CPU, input_type, output_type)

#define GENERATE_BINARY_CPU_KERNEL(tf_op, input_type) \
  GENERATE_BINARY_KERNEL(tf_op, CPU, input_type)

#define GENERATE_BINARY_CPU_KERNEL2(tf_op, input_type, output_type) \
  GENERATE_BINARY_KERNEL2(tf_op, CPU, input_type, output_type)

#define GENERATE_AND_REGISTER_SELECT_CPU_KERNEL(tf_op, input_type) \
  GENERATE_AND_REGISTER_SELECT_KERNEL(tf_op, CPU, input_type)

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_MLIR_GENERATED_BASE_CPU_OP_H_
