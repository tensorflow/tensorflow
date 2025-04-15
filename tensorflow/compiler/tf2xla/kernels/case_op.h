/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_TF2XLA_KERNELS_CASE_OP_H_
#define TENSORFLOW_COMPILER_TF2XLA_KERNELS_CASE_OP_H_

#include <string>
#include <utility>
#include <vector>

#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/types.h"

namespace tensorflow {

// This TensorFlow op provides a functional switch/case primitive.
//
// The outputs of the branches must agree on the number, types, and
// shapes of the Tensors carried around the two bodies.
//
// Computations in branch bodies may read from and write to resource variables.
// Resource variables may be passed as arguments to the branch function's
// bodies. The XlaCompiler converts resource variable arguments
// into parameters to the XLA computation and moves them to the end of the
// parameter list, and by using the `return_updated_values_for_all_variables`
// we ensure that all variables that appear in the input also appear at the
// end of the branch bodies output. This ensures the branch bodies output
// signatures match.
//
// It is the user's responsibility to ensure that each non-variable _Arg matches
// the corresponding _Retval.
class XlaCaseOp : public XlaOpKernel {
 public:
  explicit XlaCaseOp(OpKernelConstruction* ctx);

  void Compile(XlaOpKernelContext* ctx) override;

 private:
  XlaCaseOp(const XlaCaseOp&) = delete;
  void operator=(const XlaCaseOp&) = delete;

  // If the branch_index input is a constant: prunes out all but the branch
  // corrresponding to that constant branch index, and returns that branch and
  // the literal 0 (as the first and second component of the pair).
  //
  // If the branch_index input is not a constant: returns unpruned_branches_ and
  // the branch_index input.
  std::pair<std::vector<NameAttrList>, xla::XlaOp> GetPrunedBranchesAndIndex(
      XlaOpKernelContext* ctx);

  std::vector<NameAttrList> unpruned_branches_;
  DataTypeVector input_types_;
  DataTypeVector output_types_;
  bool has_token_input_output_;
  std::vector<string> token_input_nodes_;
  string original_node_name_;
  // Whether to propagate compile time consts into the cond branches.
  // This is not supported by default now since it may cause HBM memory
  // overheads.
  bool propagate_compile_time_consts_ = false;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_TF2XLA_KERNELS_CASE_OP_H_
