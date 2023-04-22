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

#ifndef TENSORFLOW_COMPILER_TF2XLA_KERNELS_WHILE_OP_H_
#define TENSORFLOW_COMPILER_TF2XLA_KERNELS_WHILE_OP_H_

#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/core/framework/attr_value.pb.h"

namespace tensorflow {

// This TensorFlow op provides a functional iteration primitive.
//
// The inputs and outputs of the loop body must agree on the number, types, and
// shapes of the Tensors carried around the loop body.
//
// Computations in while loops may read from and write to resource variables.
// Resource variables may be passed as arguments to a function's body and
// condition functions. The XlaCompiler converts resource variable arguments
// into parameters to the XLA computation and moves them to the end of the
// parameter list, and by using the `return_updated_values_for_all_variables`
// we ensure that all variables that appear in the input also appear at the
// end of the body's output. This ensures the loop body's input and output
// signatures match.
//
// It is the user's responsibility to ensure that each non-variable _Arg matches
// the corresponding _Retval.
//
// For example, suppose we have a loop body with arguments:
// DT_INT32, DT_RESOURCE (pointing to a DT_BOOL var), DT_FLOAT
// and return values
// DT_INT32, DT_FLOAT
// It is an error for the body to return DT_RESOURCE values.
//
// The body will be lowered into an XLA computation that takes and returns a
// tuple with XLA type (I32, F32, PRED). Note the resource variable appears at
// the end of both the loop body's input and output argument lists.
class XlaWhileOp : public XlaOpKernel {
 public:
  explicit XlaWhileOp(OpKernelConstruction* ctx);

  void Compile(XlaOpKernelContext* ctx) override;

 private:
  NameAttrList cond_name_attr_;
  NameAttrList body_name_attr_;
  bool has_token_input_output_;
  std::vector<string> token_input_nodes_;
  string original_node_name_;
  // Whether to propagate compile time consts into the loop body.
  // This is not supported by default now since it may cause HBM memory
  // overheads.
  bool propagate_compile_time_consts_ = false;

  TF_DISALLOW_COPY_AND_ASSIGN(XlaWhileOp);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_TF2XLA_KERNELS_WHILE_OP_H_
