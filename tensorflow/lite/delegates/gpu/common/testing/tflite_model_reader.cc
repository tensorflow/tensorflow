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

#include <memory>

#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/core/api/op_resolver.h"
#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/model_builder.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/transformations/general_transformations.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model_builder.h"

namespace tflite {
namespace gpu {
namespace {

class DelegateContext {
 public:
  bool Init(TfLiteContext* context,
            const TfLiteDelegateParams* delegate_params) {
    auto denormalized_graph =
        reinterpret_cast<GraphFloat32*>(delegate_params->delegate->data_);
    return denormalized_graph
               ? BuildModel(context, delegate_params, denormalized_graph).ok()
               : false;
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
  interpreter->UseNNAPI(false);
  TfLiteDelegate delegate;
  delegate.data_ = graph;
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
  if (!ApplyGeneralTransformations(&transformer)) {
    return absl::InternalError("Graph general transformations failed");
  }

  return absl::OkStatus();
}

}  // namespace gpu
}  // namespace tflite
