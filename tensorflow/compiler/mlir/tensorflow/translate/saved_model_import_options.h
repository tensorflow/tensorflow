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

#ifndef TENSORFLOW_COMPILER_MLIR_TENSORFLOW_TRANSLATE_SAVED_MODEL_IMPORT_OPTIONS_H_
#define TENSORFLOW_COMPILER_MLIR_TENSORFLOW_TRANSLATE_SAVED_MODEL_IMPORT_OPTIONS_H_

namespace tensorflow {

struct SavedModelImportOptions {
  // If true, functionalize the input graph before importing it into MLIR.
  bool upgrade_legacy = false;

  // Whether to unconditionally use the shape set via _output_shapes on import.
  bool unconditionally_use_set_output_shapes = false;

  // Apply default attributes from the op definition to the loaded op.
  bool add_default_attributes = true;

  // If set, promote tf.VarHandleOp to resource arguments for all functions.
  bool lift_variables = true;

  // Keeps the variables in initializers before lifting variables (when
  // `lift_variables == true`) or newly adding variable initialization patterns
  // in the initializer functions. One might want to set this to `true` because
  // the `RemoveVariablesInSessionInitializerPass` pass, which runs otherwise,
  // may unexpectedly also remove the initialization patterns for non-variable
  // resources (like hash tables) if they involve variables. Such a case is
  // illustrated in the test file
  // "../tests/tf_saved_model_remove_vars_in_session_initializer.mlir".
  // This defaults to `false` to avoid breaking existing uses.
  bool include_variables_in_initializers = false;

  // Load the model without restoring associated variables from disk. Enables
  // loading raw programs without checkpoints.
  bool allow_uninitialized_variables = false;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_TENSORFLOW_TRANSLATE_SAVED_MODEL_IMPORT_OPTIONS_H_
