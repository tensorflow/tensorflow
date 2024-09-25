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

#ifndef TENSORFLOW_COMPILER_TF2XLA_KERNELS_IF_OP_H_
#define TENSORFLOW_COMPILER_TF2XLA_KERNELS_IF_OP_H_

#include <vector>

#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

// This TensorFlow op provides a functional conditional primitive.
//
// The outputs of the then/else branches must agree on the number, types, and
// shapes of the Tensors carried around the two bodies.
//
// Computations in then/else bodies may read from and write to resource
// variables.
// Resource variables may be passed as arguments to the then/else function's
// bodies. The XlaCompiler converts resource variable arguments
// into parameters to the XLA computation and moves them to the end of the
// parameter list, and by using the `return_updated_values_for_all_variables`
// we ensure that all variables that appear in the input also appear at the
// end of the then/else bodies output. This ensures the then/else bodies output
// signatures match.
//
// It is the user's responsibility to ensure that each non-variable _Arg matches
// the corresponding _Retval.
class XlaIfOp : public XlaOpKernel {
 public:
  explicit XlaIfOp(OpKernelConstruction* ctx);

  void Compile(XlaOpKernelContext* ctx) override;

 private:
  XlaIfOp(const XlaIfOp&) = delete;
  void operator=(const XlaIfOp&) = delete;

  NameAttrList then_branch_;
  NameAttrList else_branch_;
  DataType cond_type_;
  DataTypeVector input_types_;
  DataTypeVector output_types_;
  std::vector<PartialTensorShape> output_shapes_;
  bool has_token_input_output_;
  std::vector<string> token_input_nodes_;
  string original_node_name_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_TF2XLA_KERNELS_IF_OP_H_
