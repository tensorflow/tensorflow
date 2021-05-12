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
#include "tensorflow/compiler/mlir/tensorflow/transforms/tf_saved_model_passes.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/fake_session.h"

namespace mlir {
namespace tf_saved_model {
namespace {
class InitializeVariablesInSessionInitializerPass;

static PassRegistration<InitializeVariablesInSessionInitializerPass>
    initialize_variables_in_session_init_test_pass(
        "tf-saved-model-initialize-variables-in-session-init-test",
        "Initialize variables in session initializer function.", [] {
          static tensorflow::Session* session =
              new TF::test_util::FakeSession();
          return CreateInitializeVariablesInSessionInitializerPass(session);
        });
}  // namespace
}  // namespace tf_saved_model
}  // namespace mlir
