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
#ifndef TENSORFLOW_C_EXPERIMENTAL_OPS_MATH_OPS_H_
#define TENSORFLOW_C_EXPERIMENTAL_OPS_MATH_OPS_H_

#include "tensorflow/c/eager/abstract_context.h"
#include "tensorflow/c/eager/abstract_tensor_handle.h"

namespace tensorflow {
namespace ops {
Status Mul(AbstractContext* ctx, absl::Span<AbstractTensorHandle* const> inputs,
           absl::Span<AbstractTensorHandle*> outputs, const char* name);

Status Conj(AbstractContext* ctx,
            absl::Span<AbstractTensorHandle* const> inputs,
            absl::Span<AbstractTensorHandle*> outputs, const char* name);

Status Add(AbstractContext* ctx, absl::Span<AbstractTensorHandle* const> inputs,
           absl::Span<AbstractTensorHandle*> outputs, const char* name);

Status MatMul(AbstractContext* ctx,
              absl::Span<AbstractTensorHandle* const> inputs,
              absl::Span<AbstractTensorHandle*> outputs, const char* name,
              bool transpose_a, bool transpose_b);

Status Neg(AbstractContext* ctx, absl::Span<AbstractTensorHandle* const> inputs,
           absl::Span<AbstractTensorHandle*> outputs, const char* name);

Status Sum(AbstractContext* ctx, absl::Span<AbstractTensorHandle* const> inputs,
           absl::Span<AbstractTensorHandle*> outputs, const char* name);

Status Sub(AbstractContext* ctx, absl::Span<AbstractTensorHandle* const> inputs,
           absl::Span<AbstractTensorHandle*> outputs, const char* name);

Status Div(AbstractContext* ctx, absl::Span<AbstractTensorHandle* const> inputs,
           absl::Span<AbstractTensorHandle*> outputs, const char* name);

Status DivNoNan(AbstractContext* ctx,
                absl::Span<AbstractTensorHandle* const> inputs,
                absl::Span<AbstractTensorHandle*> outputs, const char* name);

Status Exp(AbstractContext* ctx, absl::Span<AbstractTensorHandle* const> inputs,
           absl::Span<AbstractTensorHandle*> outputs, const char* name);

Status Sqrt(AbstractContext* ctx,
            absl::Span<AbstractTensorHandle* const> inputs,
            absl::Span<AbstractTensorHandle*> outputs, const char* name);

Status SqrtGrad(AbstractContext* ctx,
                absl::Span<AbstractTensorHandle* const> inputs,
                absl::Span<AbstractTensorHandle*> outputs, const char* name);

Status Log1p(AbstractContext* ctx,
             absl::Span<AbstractTensorHandle* const> inputs,
             absl::Span<AbstractTensorHandle*> outputs, const char* name);

Status ArgMax(AbstractContext* ctx, AbstractTensorHandle* const input,
              AbstractTensorHandle* const dimension,
              absl::Span<AbstractTensorHandle*> outputs,
              DataType output_type = DT_INT64, const char* name = nullptr);

Status Equal(AbstractContext* ctx, AbstractTensorHandle* const x,
             AbstractTensorHandle* const y,
             absl::Span<AbstractTensorHandle*> outputs,
             bool incompatible_shape_error = true, const char* name = nullptr);

Status AddN(AbstractContext* ctx,
            absl::Span<AbstractTensorHandle* const> inputs,
            absl::Span<AbstractTensorHandle*> outputs,
            const char* name = nullptr);

Status Cast(AbstractContext* ctx, AbstractTensorHandle* const x, DataType DstT,
            absl::Span<AbstractTensorHandle*> outputs, bool Truncate = false,
            const char* name = nullptr);

Status Mean(AbstractContext* ctx, AbstractTensorHandle* const input,
            AbstractTensorHandle* const axis,
            absl::Span<AbstractTensorHandle*> outputs, bool keep_dims = false,
            const char* name = nullptr);

}  // namespace ops
}  // namespace tensorflow

#endif  // TENSORFLOW_C_EXPERIMENTAL_OPS_MATH_OPS_H_
