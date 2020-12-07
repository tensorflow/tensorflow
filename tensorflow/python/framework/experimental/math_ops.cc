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

#include "tensorflow/c/experimental/ops/math_ops.h"

#include <pybind11/stl.h>

#include <memory>

#include "absl/types/span.h"
#include "pybind11/pybind11.h"
#include "tensorflow/c/eager/abstract_context.h"
#include "tensorflow/c/eager/abstract_tensor_handle.h"
#include "tensorflow/python/lib/core/pybind11_status.h"

using tensorflow::AbstractContext;
using tensorflow::AbstractTensorHandle;

namespace tensorflow {
PYBIND11_MODULE(_math_ops, m) {
  m.def("add", [](AbstractContext* ctx, AbstractTensorHandle* a,
                  AbstractTensorHandle* b, const char* name) {
    int num_outputs = 1;
    std::vector<AbstractTensorHandle*> outputs(1);
    if (!name) {
      name = "Add";
    }
    MaybeRaiseRegisteredFromStatus(
        ops::Add(ctx, {a, b}, absl::MakeSpan(outputs), name));
    return outputs[0];
  });
  m.def("mat_mul", [](AbstractContext* ctx, AbstractTensorHandle* a,
                      AbstractTensorHandle* b, const char* name) {
    int num_outputs = 1;
    std::vector<AbstractTensorHandle*> outputs(1);
    if (!name) {
      name = "MatMul";
    }
    MaybeRaiseRegisteredFromStatus(
        ops::MatMul(ctx, {a, b}, absl::MakeSpan(outputs), name,
                    /*transpose_a=*/false, /*transpose_b=*/false));
    return outputs[0];
  });
  m.def("neg",
        [](AbstractContext* ctx, AbstractTensorHandle* a, const char* name) {
          int num_outputs = 1;
          std::vector<AbstractTensorHandle*> outputs(1);
          if (!name) {
            name = "Neg";
          }
          MaybeRaiseRegisteredFromStatus(
              ops::Neg(ctx, {a}, absl::MakeSpan(outputs), name));
          return outputs[0];
        });
  m.def("sub", [](AbstractContext* ctx, AbstractTensorHandle* a,
                  AbstractTensorHandle* b, const char* name) {
    int num_outputs = 1;
    std::vector<AbstractTensorHandle*> outputs(1);
    if (!name) {
      name = "Sub";
    }
    MaybeRaiseRegisteredFromStatus(
        ops::Sub(ctx, {a, b}, absl::MakeSpan(outputs), name));
    return outputs[0];
  });
  m.def("mul", [](AbstractContext* ctx, AbstractTensorHandle* a,
                  AbstractTensorHandle* b, const char* name) {
    int num_outputs = 1;
    std::vector<AbstractTensorHandle*> outputs(1);
    if (!name) {
      name = "Mul";
    }
    MaybeRaiseRegisteredFromStatus(
        ops::Mul(ctx, {a, b}, absl::MakeSpan(outputs), name));
    return outputs[0];
  });
  m.def("log1p",
        [](AbstractContext* ctx, AbstractTensorHandle* a, const char* name) {
          int num_outputs = 1;
          std::vector<AbstractTensorHandle*> outputs(1);
          if (!name) {
            name = "Log1p";
          }
          MaybeRaiseRegisteredFromStatus(
              ops::Log1p(ctx, {a}, absl::MakeSpan(outputs), name));
          return outputs[0];
        });
  m.def("div_no_nan", [](AbstractContext* ctx, AbstractTensorHandle* a,
                         AbstractTensorHandle* b, const char* name) {
    int num_outputs = 1;
    std::vector<AbstractTensorHandle*> outputs(1);
    if (!name) {
      name = "DivNoNan";
    }
    MaybeRaiseRegisteredFromStatus(
        ops::DivNoNan(ctx, {a, b}, absl::MakeSpan(outputs), name));
    return outputs[0];
  });
}
}  // namespace tensorflow
