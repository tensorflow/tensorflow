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
#include <pybind11/stl.h>

#include "pybind11/pybind11.h"
#include "tensorflow/c/eager/gradients.h"
#include "tensorflow/c/experimental/gradients/math_grad.h"
#include "tensorflow/c/experimental/gradients/nn_grad.h"
#include "tensorflow/c/experimental/gradients/tape/tape_context.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/python/lib/core/pybind11_status.h"

namespace py = pybind11;

namespace tensorflow {
namespace gradients {

Status RegisterGradients(GradientRegistry* registry) {
  // TODO(srbs): Rename ops::Add and AddRegisterer to AddV2.
  TF_RETURN_IF_ERROR(registry->Register("AddV2", AddRegisterer));
  TF_RETURN_IF_ERROR(registry->Register("Exp", ExpRegisterer));
  TF_RETURN_IF_ERROR(registry->Register("MatMul", MatMulRegisterer));
  TF_RETURN_IF_ERROR(registry->Register("Relu", ReluRegisterer));
  TF_RETURN_IF_ERROR(
      registry->Register("SparseSoftmaxCrossEntropyWithLogits",
                         SparseSoftmaxCrossEntropyWithLogitsRegisterer));
  TF_RETURN_IF_ERROR(registry->Register("Neg", NegRegisterer));
  TF_RETURN_IF_ERROR(registry->Register("Sub", SubRegisterer));
  TF_RETURN_IF_ERROR(registry->Register("Mul", MulRegisterer));
  TF_RETURN_IF_ERROR(registry->Register("Log1p", Log1pRegisterer));
  TF_RETURN_IF_ERROR(registry->Register("DivNoNan", DivNoNanRegisterer));
  return Status::OK();
}

PYBIND11_MODULE(_tape, m) {
  py::class_<Tape>(m, "Tape")
      .def(py::init([](bool persistent) { return new Tape(persistent); }))
      .def("Watch",
           [](Tape* self, AbstractTensorHandle* t) { self->Watch(ToId(t)); })
      .def("ComputeGradient",
           [](Tape* self, TapeVSpace* vspace,
              std::vector<AbstractTensorHandle*> target_tensors,
              std::vector<AbstractTensorHandle*> source_tensors,
              std::vector<AbstractTensorHandle*> output_gradients) {
             std::vector<int64> target_tensor_ids;
             std::vector<int64> source_tensor_ids;
             target_tensor_ids.reserve(target_tensors.size());
             source_tensor_ids.reserve(source_tensors.size());
             for (auto t : target_tensors) {
               target_tensor_ids.emplace_back(ToId(t));
             }
             for (auto t : source_tensors) {
               source_tensor_ids.emplace_back(ToId(t));
             }
             std::unordered_map<tensorflow::int64, TapeTensor>
                 source_tensors_that_are_targets;
             std::vector<AbstractTensorHandle*> results;
             Status s = self->ComputeGradient(
                 *vspace, target_tensor_ids, source_tensor_ids,
                 source_tensors_that_are_targets, output_gradients, &results,
                 /*build_default_zeros_grads=*/false);
             MaybeRaiseRegisteredFromStatus(s);
             return results;
           });
  py::class_<TapeVSpace>(m, "TapeVSpace")
      .def(py::init([](AbstractContext* ctx) { return new TapeVSpace(ctx); }));
  py::class_<GradientRegistry>(m, "GradientRegistry").def(py::init([]() {
    auto registry = new GradientRegistry();
    MaybeRaiseRegisteredFromStatus(RegisterGradients(registry));
    return registry;
  }));
  py::class_<TapeContext, AbstractContext>(m, "TapeContext")
      .def(py::init(
          [](AbstractContext* ctx, Tape* tape, GradientRegistry* registry) {
            return new TapeContext(ctx, tape, *registry);
          }));
}
}  // namespace gradients
}  // namespace tensorflow
