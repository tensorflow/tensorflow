// Copyright 2025 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tensorflow/lite/experimental/litert/c/litert_accelerator_options.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_compiled_model.h"
#include "tensorflow/lite/experimental/litert/c/litert_compiled_model_options.h"
#include "tensorflow/lite/experimental/litert/c/vendors/qualcomm/accelerator_options.h"
#include "tensorflow/lite/experimental/litert/cc/litert_macros.h"

LiteRtStatus example() {
  // This is the object that is passed to LiteRtCreateCompitedModel.
  LiteRtCompilationOptions compilation_options;
  LITERT_RETURN_IF_ERROR(LiteRtCreateCompilationOptions(&compilation_options));

  // Create and populate QNN options
  LiteRtAcceleratorCompilationOptions qnn_options;
  LITERT_RETURN_IF_ERROR(
      LiteRtCreateQualcommAcceleratorCompilationOptions(&qnn_options));

  LITERT_RETURN_IF_ERROR(
      LiteRtSetQualcommAcceleratorLogLevel(qnn_options, kLogLevelInfo));

  LITERT_RETURN_IF_ERROR(LiteRtSetQualcommAcceleratorHtpPerformanceMode(
      qnn_options, kHtpHighPerformance));

  // Add QNN options to the compilation options.
  //
  // Note: we don't need to manually destroy qnn_options after this call.
  // Management is transferred to compilation_options.
  LITERT_RETURN_IF_ERROR(
      LiteRtAddAcceleratorCompilationOptions(compilation_options, qnn_options));

  // Pass the compilation options to model creation.
  //
  // Note: we don't need to manually destroy compilation_options after this
  // call. Management is transferred to the compiled model.
  LiteRtCompiledModel compiled_model;
  LiteRtCreateCompiledModel(/*environment=*/nullptr, /*model=*/nullptr,
                            compilation_options, &compiled_model);

  return kLiteRtStatusOk;
}
