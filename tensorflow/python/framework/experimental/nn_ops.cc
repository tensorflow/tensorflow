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

#include "tensorflow/c/experimental/ops/nn_ops.h"

#include <pybind11/stl.h>

#include <memory>

#include "absl/types/span.h"
#include "pybind11/pybind11.h"
#include "tensorflow/c/eager/abstract_context.h"
#include "tensorflow/c/eager/abstract_tensor_handle.h"
#include "tensorflow/c/eager/gradients.h"
#include "tensorflow/python/lib/core/pybind11_status.h"

// This library provides helpers for running ops and recording them on a
// GradientTape. This is currently needed because the tape does not provide
// an implementation of the abstract execution APIs but that will change.
// TODO(b/168209775): Remove this and its imported symbols once the tape
// execution context is ready.
#include "tensorflow/c/eager/mnist_gradients_testutil.h"

using tensorflow::AbstractContext;
using tensorflow::AbstractTensorHandle;
using tensorflow::gradients::GradientRegistry;
using tensorflow::gradients::Tape;

namespace tensorflow {
PYBIND11_MODULE(_nn_ops, m) {
  m.def("relu", [](AbstractContext* ctx, AbstractTensorHandle* a,
                   const char* name, Tape* tape, GradientRegistry* registry) {
    int num_outputs = 1;
    std::vector<AbstractTensorHandle*> outputs(1);
    if (!name) {
      name = "Relu";
    }
    if (!tape) {
      MaybeRaiseRegisteredFromStatus(
          ops::Relu(ctx, {a}, absl::MakeSpan(outputs), name));
    } else {
      MaybeRaiseRegisteredFromStatus(gradients::internal::Relu(
          ctx, tape, {a}, absl::MakeSpan(outputs), name, *registry));
    }
    return outputs[0];
  });

  m.def("sparse_softmax_cross_entropy_with_logits",
        [](AbstractContext* ctx, AbstractTensorHandle* features,
           AbstractTensorHandle* labels, const char* name, Tape* tape,
           GradientRegistry* registry) {
          int num_outputs = 2;
          std::vector<AbstractTensorHandle*> outputs(2);
          if (!name) {
            name = "SparseSoftmaxCrossEntropyWithLogits";
          }
          if (!tape) {
            MaybeRaiseRegisteredFromStatus(
                ops::SparseSoftmaxCrossEntropyWithLogits(
                    ctx, {features, labels}, absl::MakeSpan(outputs), name));
          } else {
            MaybeRaiseRegisteredFromStatus(
                gradients::internal::SparseSoftmaxCrossEntropyWithLogits(
                    ctx, tape, {features, labels}, absl::MakeSpan(outputs),
                    name, *registry));
          }
          return outputs[0];  // Only return the loss vals, not the backprop.
        });
}
}  // namespace tensorflow
