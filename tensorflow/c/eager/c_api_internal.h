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
#include "tensorflow/c/eager/tfe_executor_internal.h"  // IWYU pragma: export
#include "tensorflow/c/eager/tfe_monitoring_internal.h"  // IWYU pragma: export
#include "tensorflow/c/eager/tfe_op_attrs_internal.h"  // IWYU pragma: export
#include "tensorflow/c/eager/tfe_tensor_debug_info_internal.h"  // IWYU pragma: export

// TODO(b/154564140): Move this to its own header. This requires splitting
// c_api_experimental.h
struct TFE_ContextOptions {
  TF_SessionOptions session_options;
  // true if async execution is enabled.
  bool async = false;
  TFE_ContextDevicePlacementPolicy device_placement_policy{
      TFE_DEVICE_PLACEMENT_SILENT};
  // If true, use TFRT backend
  bool use_tfrt = false;
  // This option is effective only when use_tfrt is true. If true, TFRT will use
  // native TFRT distributed runtime. Otherwise, TFRT will use current runtime's
  // distributed runtime. Note that TFRT distributed runtime is in development
  // and not functionally complete.
  bool use_tfrt_distributed_runtime = false;
  // Whether to run elementary eager ops wrapped in a call op.
  bool run_eager_op_as_function = false;
  // Whether to rewrite jit_compile functions.
  bool jit_compile_rewrite = false;
};

#endif  // TENSORFLOW_C_EAGER_C_API_INTERNAL_H_
