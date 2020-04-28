/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_C_EAGER_C_API_INTERNAL_H_
#define TENSORFLOW_C_EAGER_C_API_INTERNAL_H_

#include "tensorflow/c/c_api_internal.h"
#include "tensorflow/c/eager/c_api.h"
#include "tensorflow/c/eager/c_api_experimental.h"
#include "tensorflow/c/eager/tfe_cancellation_manager_internal.h"  // IWYU pragma: export
#include "tensorflow/c/eager/tfe_context_internal.h"  // IWYU pragma: export
#include "tensorflow/c/eager/tfe_executor_internal.h"  // IWYU pragma: export
#include "tensorflow/c/eager/tfe_monitoring_internal.h"  // IWYU pragma: export
#include "tensorflow/c/eager/tfe_op_attrs_internal.h"  // IWYU pragma: export
#include "tensorflow/c/eager/tfe_op_internal.h"  // IWYU pragma: export
#include "tensorflow/c/eager/tfe_tensor_debug_info_internal.h"  // IWYU pragma: export
#include "tensorflow/c/eager/tfe_tensorhandle_internal.h"  // IWYU pragma: export

// TODO(b/154564140): Move this to its own header. This requires splitting
// c_api_experimental.h
struct TFE_ContextOptions {
  TF_SessionOptions session_options;
  // true if async execution is enabled.
  bool async = false;
  TFE_ContextDevicePlacementPolicy device_placement_policy{
      TFE_DEVICE_PLACEMENT_SILENT};
  TFE_ContextMirroringPolicy mirroring_policy{TFE_MIRRORING_NONE};
  // If true, lazily copy the remote inputs of a function to the target devices.
  bool lazy_remote_inputs_copy = true;
  // If true, use TFRT backend
  bool use_tfrt = false;
};

#endif  // TENSORFLOW_C_EAGER_C_API_INTERNAL_H_
