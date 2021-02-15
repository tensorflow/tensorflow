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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TASK_WEIGHTS_LAYOUT_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TASK_WEIGHTS_LAYOUT_H_

#include <vector>

namespace tflite {
namespace gpu {

enum class WeightsLayout {
  kUnknown,
  kOHWIOGroupI4O4,
  kOHWIOGroupO4I4,
  kOICustomSpatialI4O4,
  kOICustomSpatialO4I4,
  k2DX4I4YIsHWIAndXIsOOGroupO4,
  k2DX4O4YIsHWIAndXIsOOGroupI4,
};

struct WeightsDescription {
  WeightsLayout layout;
  // applicable with layouts that have OGroup.
  int output_group_size;  // OGroup size
  // applicable with layouts that have CustomSpatial
  std::vector<int> spatial_remap;

  int GetOutputGroupSize() const;
  bool IsI4O4() const;
  bool IsO4I4() const;
};

}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TASK_WEIGHTS_LAYOUT_H_
