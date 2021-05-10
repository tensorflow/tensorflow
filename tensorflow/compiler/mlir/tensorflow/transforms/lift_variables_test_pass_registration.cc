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

#include "tensorflow/compiler/mlir/tensorflow/transforms/lift_variables_test_pass.h"

namespace mlir {
namespace tf_saved_model {

static PassRegistration<LiftVariablesTestPass> lift_variables_test_pass(
    "tf-saved-model-lift-variables-test",
    "Lift variables and save them as global tensors");

static PassRegistration<LiftVariablesInvalidSessionTestPass>
    lift_variables_invalid_session_test_pass(
        "tf-saved-model-lift-variables-invalid-session-test",
        "Lift variables and save them as global tensors with an invalid "
        "session");

}  // namespace tf_saved_model
}  // namespace mlir
