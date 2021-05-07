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
#include "tensorflow/lite/delegates/gpu/common/testing/tflite_model_reader.h"

#include <stddef.h>

#include <memory>

#include "tensorflow/lite/core/api/op_resolver.h"
#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/model_builder.h"
#include "tensorflow/lite/delegates/gpu/common/model_transformer.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/transformations/model_transformations.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/model_builder.h"

namespace tflite {
namespace gpu {
namespace {

class DelegateContext {
 public:
  struct DelegateData {
    std::vector<int> input_ids;
    std::vector<int> output_ids;
    GraphFloat32* graph;
  };
  bool Init(TfLiteContext* context,
            const TfLiteDelegateParams* delegate_params) {
    const auto* delegate_data =
        reinterpret_cast<DelegateData*>(delegate_params->delegate->data_);

    return delegate_data->graph &&
           BuildModelEnforceIO(context, delegate_params,
                               delegate_data->input_ids,
                               delegate_data->output_ids, delegate_data->graph)
               .ok();
  }
};

TfLiteStatus DelegatePrepare(TfLiteContext* context, TfLiteDelegate* delegate) {
  const TfLiteRegistration kRegistration = {
      .init = [](TfLiteContext* context, const char* buffer, size_t) -> void* {
        auto* delegate_context = new DelegateContext();
        if (!delegate_context->Init(
                context,
                reinterpret_cast<const TfLiteDelegateParams*>(buffer))) {
          delete delegate_context;
          return nullptr;
        }
        return delegate_context;
      },
      .free = [](TfLiteContext* context, void* buffer) -> void {
        delete reinterpret_cast<DelegateContext*>(buffer);
      },
      .prepare = [](TfLiteContext* context, TfLiteNode* node) -> TfLiteStatus {
        return node->user_data ? kTfLiteOk : kTfLiteError;
      },
      .invoke = nullptr,
  };

  TfLiteIntArray* ops_to_replace = GetOpsToReplace(context);
  const auto status = context->ReplaceNodeSubsetsWithDelegateKernels(
      context, kRegistration, ops_to_replace, delegate);
  TfLiteIntArrayFree(ops_to_replace);
  return status;
}
}  // namespace

absl::Status BuildFromFlatBuffer(const tflite::FlatBufferModel& flatbuffer,
                                 const tflite::OpResolver& op_resolver,
                                 GraphFloat32* graph) {
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::InterpreterBuilder interpreter_builder(flatbuffer, op_resolver);
  if (interpreter_builder(&interpreter) != kTfLiteOk || !interpreter) {
    return absl::InternalError("Unable to prepare TfLite interpreter.");
  }
  TfLiteDelegate delegate;

  DelegateContext::DelegateData delegate_data{interpreter->inputs(),
                                              interpreter->outputs(), graph};

  delegate.data_ = &delegate_data;
  delegate.flags = kTfLiteDelegateFlagsNone;
  delegate.Prepare = DelegatePrepare;
  delegate.CopyFromBufferHandle = nullptr;
  delegate.CopyToBufferHandle = nullptr;
  delegate.FreeBufferHandle = nullptr;

  if (interpreter->ModifyGraphWithDelegate(&delegate) != kTfLiteOk) {
    return absl::InternalError("Conversion from TfLite model failed.");
  }

  NullTransformationReporter reporter;
  ModelTransformer transformer(graph, &reporter);
  if (!ApplyModelTransformations(&transformer)) {
    return absl::InternalError("Graph transformations failed");
  }

  return absl::OkStatus();
}

}  // namespace gpu
}  // namespace tflite
