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
#ifndef TENSORFLOW_LITE_TOOLS_EVALUATION_EVALUATION_STAGE_FACTORY_H_
#define TENSORFLOW_LITE_TOOLS_EVALUATION_EVALUATION_STAGE_FACTORY_H_

#include "absl/memory/memory.h"
#include "tensorflow/lite/tools/evaluation/evaluation_stage.h"
#include "tensorflow/lite/tools/evaluation/identity_stage.h"
#include "tensorflow/lite/tools/evaluation/proto/evaluation_config.pb.h"
#include "tensorflow/lite/tools/evaluation/proto/evaluation_stages.pb.h"

namespace tflite {
namespace evaluation {

// The canonical way to generate EvaluationStages.
// TODO(b/122482115): Implement a Factory class for registration of classes.
std::unique_ptr<EvaluationStage> CreateEvaluationStageFromConfig(
    const EvaluationStageConfig& config) {
  if (!config.has_specification() ||
      !config.specification().has_process_class()) {
    LOG(ERROR) << "Process specification not present in config: "
               << config.name();
    return nullptr;
  }
  switch (config.specification().process_class()) {
    case UNKNOWN:
      return nullptr;
    case IDENTITY:
      return absl::make_unique<IdentityStage>(config);
  }
}

}  // namespace evaluation
}  // namespace tflite

#endif  // TENSORFLOW_LITE_TOOLS_EVALUATION_EVALUATION_STAGE_FACTORY_H_
