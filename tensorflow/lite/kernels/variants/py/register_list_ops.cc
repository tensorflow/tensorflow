/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
#include "pybind11/pybind11.h"  // from @pybind11
#include "pybind11/pytypes.h"  // from @pybind11
#include "tensorflow/lite/kernels/variants/list_ops_lib.h"
#include "tensorflow/lite/mutable_op_resolver.h"

namespace tflite {

void RegisterListOps(tflite::MutableOpResolver* resolver) {
  resolver->AddCustom("TensorListReserve",
                      ::tflite::variants::ops::Register_LIST_RESERVE());
  resolver->AddCustom("TensorListStack",
                      ::tflite::variants::ops::Register_LIST_STACK());
  resolver->AddCustom("TensorListSetItem",
                      ::tflite::variants::ops::Register_LIST_SET_ITEM());
  resolver->AddCustom("TensorListFromTensor",
                      ::tflite::variants::ops::Register_LIST_FROM_TENSOR());
}

}  // namespace tflite

PYBIND11_MODULE(register_list_ops, m) {
  m.doc() = R"pbdoc(
    Bindings to register list ops with python interpreter.
  )pbdoc";
  m.def(
      "TFLRegisterListOps",
      [](uintptr_t resolver) {
        ::tflite::RegisterListOps(
            reinterpret_cast<::tflite::MutableOpResolver*>(resolver));
      },
      R"pbdoc(
        Register all custom list ops.
      )pbdoc");
}
