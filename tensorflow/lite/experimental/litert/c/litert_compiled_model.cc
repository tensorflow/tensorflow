// Copyright 2024 Google LLC.
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

#include "tensorflow/lite/experimental/litert/c/litert_compiled_model.h"

#include <stddef.h>

#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_compiled_model_options.h"
#include "tensorflow/lite/experimental/litert/c/litert_logging.h"
#include "tensorflow/lite/experimental/litert/c/litert_model.h"
#include "tensorflow/lite/experimental/litert/c/litert_tensor_buffer.h"
#include "tensorflow/lite/experimental/litert/c/litert_tensor_buffer_requirements.h"
#include "tensorflow/lite/experimental/litert/runtime/compiled_model.h"

LiteRtStatus LiteRtCreateCompiledModel(
    LiteRtModel model, LiteRtCompilationOptions compilation_options,
    LiteRtCompiledModel* compiled_model) {
  if (!model || !compiled_model) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  auto created_compiled_model =
      LiteRtCompiledModelT::Create(model, compilation_options);
  if (!created_compiled_model) {
    LITERT_LOG(LITERT_ERROR, "%s",
               created_compiled_model.Error().Message().data());
    return created_compiled_model.Error().Status();
  }
  *compiled_model = created_compiled_model->release();
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetCompiledModelInputBufferRequirements(
    LiteRtCompiledModel compiled_model, LiteRtParamIndex signature_index,
    LiteRtParamIndex input_index,
    LiteRtTensorBufferRequirements* buffer_requirements) {
  if (!compiled_model || !buffer_requirements) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  auto res = compiled_model->GetInputBufferRequirementsCApi(signature_index,
                                                            input_index);
  if (!res) {
    LITERT_LOG(LITERT_ERROR, "%s", res.Error().Message().data());
    return res.Error().Status();
  }
  *buffer_requirements = res.Value();
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetCompiledModelOutputBufferRequirements(
    LiteRtCompiledModel compiled_model, LiteRtParamIndex signature_index,
    LiteRtParamIndex output_index,
    LiteRtTensorBufferRequirements* buffer_requirements) {
  if (!compiled_model || !buffer_requirements) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  auto res = compiled_model->GetOutputBufferRequirementsCApi(signature_index,
                                                             output_index);
  if (!res) {
    LITERT_LOG(LITERT_ERROR, "%s", res.Error().Message().data());
    return res.Error().Status();
  }
  *buffer_requirements = res.Value();
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtRunCompiledModel(LiteRtCompiledModel compiled_model,
                                    LiteRtParamIndex signature_index,
                                    size_t num_input_buffers,
                                    LiteRtTensorBuffer* input_buffers,
                                    size_t num_output_buffers,
                                    LiteRtTensorBuffer* output_buffers) {
  if (!compiled_model || (num_input_buffers > 0 && !input_buffers) ||
      (num_output_buffers > 0 && !output_buffers)) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  auto res =
      compiled_model->RunCApi(signature_index, num_input_buffers, input_buffers,
                              num_output_buffers, output_buffers);
  if (!res) {
    LITERT_LOG(LITERT_ERROR, "%s", res.Error().Message().data());
    return res.Error().Status();
  }
  return kLiteRtStatusOk;
}

void LiteRtDestroyCompiledModel(LiteRtCompiledModel compiled_model) {
  delete compiled_model;
}
