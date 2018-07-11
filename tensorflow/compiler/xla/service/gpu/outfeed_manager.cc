/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/outfeed_manager.h"

#include "tensorflow/compiler/xla/map_util.h"
#include "tensorflow/compiler/xla/ptr_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {
namespace gpu {

void OutfeedManager::EnqueueOutfeedDestination(
    ShapeTree<std::unique_ptr<OutfeedBuffer>>* buffers) {
  tensorflow::mutex_lock l(mu_);
  enqueued_buffers_.push_back(buffers);
  cv_.notify_one();
}

ShapeTree<std::unique_ptr<OutfeedBuffer>>*
OutfeedManager::BlockingGetNextOutfeedDestination() {
  tensorflow::mutex_lock l(mu_);
  while (enqueued_buffers_.empty()) {
    cv_.wait(l);
  }
  ShapeTree<std::unique_ptr<OutfeedBuffer>>* current_buffer =
      enqueued_buffers_.front();
  enqueued_buffers_.pop_front();
  return current_buffer;
}

OutfeedManager* GetOrCreateOutfeedManager() {
  static auto* manager = new OutfeedManager;
  return manager;
}

}  // namespace gpu
}  // namespace xla
