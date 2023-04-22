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

// This file is MACHINE GENERATED! Do not edit.

#ifndef TENSORFLOW_C_EXPERIMENTAL_OPS_MATH_OPS_H_
#define TENSORFLOW_C_EXPERIMENTAL_OPS_MATH_OPS_H_

#include "tensorflow/c/eager/abstract_context.h"
#include "tensorflow/c/eager/abstract_tensor_handle.h"

namespace tensorflow {
namespace ops {

Status Mul(AbstractContext* ctx, AbstractTensorHandle* const x,
           AbstractTensorHandle* const y, AbstractTensorHandle** z,
           const char* name = nullptr);

Status Conj(AbstractContext* ctx, AbstractTensorHandle* const input,
            AbstractTensorHandle** output, const char* name = nullptr);

Status AddV2(AbstractContext* ctx, AbstractTensorHandle* const x,
             AbstractTensorHandle* const y, AbstractTensorHandle** z,
             const char* name = nullptr);

Status MatMul(AbstractContext* ctx, AbstractTensorHandle* const a,
              AbstractTensorHandle* const b, AbstractTensorHandle** product,
              bool transpose_a = false, bool transpose_b = false,
              const char* name = nullptr);

Status Neg(AbstractContext* ctx, AbstractTensorHandle* const x,
           AbstractTensorHandle** y, const char* name = nullptr);

Status Sum(AbstractContext* ctx, AbstractTensorHandle* const input,
           AbstractTensorHandle* const reduction_indices,
           AbstractTensorHandle** output, bool keep_dims = false,
           const char* name = nullptr);

Status Sub(AbstractContext* ctx, AbstractTensorHandle* const x,
           AbstractTensorHandle* const y, AbstractTensorHandle** z,
           const char* name = nullptr);

Status Div(AbstractContext* ctx, AbstractTensorHandle* const x,
           AbstractTensorHandle* const y, AbstractTensorHandle** z,
           const char* name = nullptr);

Status DivNoNan(AbstractContext* ctx, AbstractTensorHandle* const x,
                AbstractTensorHandle* const y, AbstractTensorHandle** z,
                const char* name = nullptr);

Status Exp(AbstractContext* ctx, AbstractTensorHandle* const x,
           AbstractTensorHandle** y, const char* name = nullptr);

Status Sqrt(AbstractContext* ctx, AbstractTensorHandle* const x,
            AbstractTensorHandle** y, const char* name = nullptr);

Status SqrtGrad(AbstractContext* ctx, AbstractTensorHandle* const y,
                AbstractTensorHandle* const dy, AbstractTensorHandle** z,
                const char* name = nullptr);

Status Log1p(AbstractContext* ctx, AbstractTensorHandle* const x,
             AbstractTensorHandle** y, const char* name = nullptr);

}  // namespace ops
}  // namespace tensorflow

#endif  // TENSORFLOW_C_EXPERIMENTAL_OPS_MATH_OPS_H_
