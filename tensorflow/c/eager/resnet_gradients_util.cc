/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/c/eager/resnet_gradients_util.h"
#include "tensorflow/c/eager/mnist_gradients_util.h"

#include <memory>

#include "absl/container/flat_hash_set.h"
#include "absl/types/span.h"
#include "tensorflow/c/eager/abstract_tensor_handle.h"
#include "tensorflow/c/eager/c_api_experimental.h"
#include "tensorflow/c/eager/c_api_unified_experimental.h"
#include "tensorflow/c/eager/c_api_unified_experimental_internal.h"
#include "tensorflow/c/eager/gradients.h"
#include "tensorflow/c/eager/gradients_internal.h"
#include "tensorflow/c/experimental/ops/array_ops.h"
#include "tensorflow/c/experimental/ops/math_ops.h"
#include "tensorflow/c/experimental/ops/nn_ops.h"
#include "tensorflow/c/tf_status_helper.h"
#include "tensorflow/c/tf_tensor.h"
#include "tensorflow/core/lib/llvm_rtti/llvm_rtti.h"

using std::vector;
using tracing::TracingOperation;

// ========================== Tape Ops ==============================

// Computes Sum = A + bias
Status BiasAdd(AbstractContext* ctx, Tape* tape,
   absl::Span<AbstractTensorHandle* const> inputs,
   absl::Span<AbstractTensorHandle*> outputs, const char* name,
   const GradientRegistry& registry) {
       
 AbstractTensorHandle* value = inputs[0];
 AbstractTensorHandle* bias = inputs[1];
 
 AbstractOperationPtr biasAdd_op(ctx->CreateOperation());
 ForwardOperation forward_op;
 forward_op.ctx = ctx;
 TF_RETURN_IF_ERROR(Reset(biasAdd_op.get(), "BiasAdd",
                          /*raw_device_name=*/nullptr, &forward_op));
 if (isa<tracing::TracingOperation>(biasAdd_op.get())) {
   TF_RETURN_IF_ERROR(
       dyn_cast<tracing::TracingOperation>(biasAdd_op.get())->SetOpName(name));
 }
 
 TF_RETURN_IF_ERROR(AddInput(biasAdd_op.get(), value, &forward_op));
 TF_RETURN_IF_ERROR(AddInput(biasAdd_op.get(), bias, &forward_op));
 
 int num_retvals = 1;
 return Execute(biasAdd_op.get(), ctx, outputs, &num_retvals, &forward_op, tape,
                registry);
}


//===================== Test Models to run =========================

// Status MatMulGradModel(AbstractContext* ctx,
//                        absl::Span<AbstractTensorHandle* const> inputs,
//                        absl::Span<AbstractTensorHandle*> outputs,
//                        const GradientRegistry& registry) {
//   TapeVSpace vspace(ctx);
//   auto tape = new Tape(/*persistent=*/false);
//   tape->Watch(ToId(inputs[0]));  // Watch x.
//   tape->Watch(ToId(inputs[1]));  // Watch y.
//   vector<AbstractTensorHandle*> mm_outputs(1);
//   TF_RETURN_IF_ERROR(MatMul(ctx, tape, inputs, absl::MakeSpan(mm_outputs),
//                             "matmul0", /*transpose_a=*/false,
//                             /*transpose_b=*/false, registry));  // Compute x*y.

//   std::unordered_map<tensorflow::int64, TapeTensor>
//       source_tensors_that_are_targets;

//   vector<AbstractTensorHandle*> out_grads;
//   TF_RETURN_IF_ERROR(tape->ComputeGradient(
//       vspace, /*target_tensor_ids=*/{ToId(mm_outputs[0])},
//       /*source_tensor_ids=*/{ToId(inputs[0]), ToId(inputs[1])},
//       source_tensors_that_are_targets,
//       /*output_gradients=*/{}, &out_grads,
//       /*build_default_zeros_grads=*/false));
//   for (auto mm_output : mm_outputs) {
//     mm_output->Unref();
//   }
//   outputs[0] = out_grads[0];
//   outputs[1] = out_grads[1];
//   delete tape;
//   return Status::OK();
// }

Status BiasAddGradModel(AbstractContext* ctx,
                    absl::Span<AbstractTensorHandle* const> inputs,
                    absl::Span<AbstractTensorHandle*> outputs,
                    const GradientRegistry& registry) {
 TapeVSpace vspace(ctx);
 auto tape = new Tape(/*persistent=*/false);
 tape->Watch(ToId(inputs[0]));  // Watch A
 tape->Watch(ToId(inputs[1]));  // Watch bias
 std::vector<AbstractTensorHandle*> ba_outputs(1);
 TF_RETURN_IF_ERROR(BiasAdd(ctx, tape, inputs, absl::MakeSpan(ba_outputs),
                         "bias_add_test0", registry)); 
 
 std::unordered_map<tensorflow::int64, TapeTensor>
     source_tensors_that_are_targets;
 
 std::vector<AbstractTensorHandle*> out_grads;
 TF_RETURN_IF_ERROR(tape->ComputeGradient(
      vspace, /*target_tensor_ids=*/{ToId(ba_outputs[0])},
      /*source_tensor_ids=*/{ToId(inputs[0]), ToId(inputs[1])},
      source_tensors_that_are_targets,
      /*output_gradients=*/{}, &out_grads,
      /*build_default_zeros_grads=*/false));
 
 for (auto ba_output : ba_outputs) {
   ba_output->Unref();
 }
 
 outputs[0] = out_grads[0];
 outputs[1] = out_grads[1];
 delete tape;
 return Status::OK();
}



// ====================== End Models ================================


