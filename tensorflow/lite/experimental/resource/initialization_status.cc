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

#include "tensorflow/lite/experimental/resource/initialization_status.h"

namespace tflite {
namespace resource {

void InitializationStatus::MarkInitializationIsDone() {
  is_initialized_ = true;
}

bool InitializationStatus::IsInitialized() { return is_initialized_; }

InitializationStatus* GetInitializationStatus(InitializationStatusMap* map,
                                              int subgraph_id) {
  auto it = map->find(subgraph_id);
  if (it != map->end()) {
    return static_cast<InitializationStatus*>(it->second.get());
  }

  InitializationStatus* status = new InitializationStatus();
  map->emplace(subgraph_id, std::unique_ptr<InitializationStatus>(status));
  return status;
}

}  // namespace resource
}  // namespace tflite
