/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_COMMON_MODEL_BUILDER_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_COMMON_MODEL_BUILDER_H_

#include <cstdint>
#include <string>

#include "tensorflow/lite/context.h"
#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"

namespace tflite {
namespace gpu {

// Validates which operations are supported and returns array of operations to
// replace with GPU kernels. The caller must free the pointer on TfLiteIntArray.
TfLiteIntArray* GetOpsToReplace(TfLiteContext* context);

// Extracts TFLite delegate execution plan from the input TFLite context and
// converts it into generic graph format.
Status BuildModel(TfLiteContext* context,
                  const TfLiteDelegateParams* delegate_params,
                  GraphFloat32* graph);

// Same as above but also apply all transformations on the final graph.
// Prefer using this method instead of BuildModel.
Status BuildFinalModel(TfLiteContext* context,
                       const TfLiteDelegateParams* delegate_params,
                       GraphFloat32* graph);

// Module-internal converter, exposed for unit testing purpose only.
Status ConvertTfLiteTensorToTensorRef(const TfLiteTensor& tflite_tensor,
                                      TensorRef<BHWC>* tensor_ref);

}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_COMMON_MODEL_BUILDER_H_
