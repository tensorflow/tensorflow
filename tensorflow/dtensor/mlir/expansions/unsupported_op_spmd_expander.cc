/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/dtensor/mlir/expansions/unsupported_op_spmd_expander.h"

namespace tensorflow {
namespace dtensor {

UnsupportedOpSPMDExpander::UnsupportedOpSPMDExpander(
    const absl::string_view error_message) {
  error_message_ = error_message;
}

StatusOr<mlir::Operation*> UnsupportedOpSPMDExpander::ExpandOp(
    mlir::Operation* op) {
  return errors::Unimplemented(error_message_);
}

StatusOr<llvm::DenseMap<int, Layout>>
UnsupportedOpSPMDExpander::ComputeLayoutForward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& input_layouts) {
  return errors::Unimplemented(error_message_);
}

StatusOr<llvm::DenseMap<int, Layout>>
UnsupportedOpSPMDExpander::ComputeLayoutBackward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& output_layouts) {
  return errors::Unimplemented(error_message_);
}

}  // namespace dtensor
}  // namespace tensorflow
