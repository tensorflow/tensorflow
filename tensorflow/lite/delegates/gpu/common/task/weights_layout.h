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

#include "tensorflow/lite/delegates/gpu/common/data_type.h"

namespace tflite {
namespace gpu {

enum class WeightsLayout {
  kUnknown,
  // Spatial is DHW/HW depending on amount of spatial dimensions (Depth, Height,
  // Width).
  kOSpatialIOGroupI4O4,
  kOSpatialIOGroupO4I4,
  kOICustomSpatialI4O4,
  kOICustomSpatialO4I4,
  k2DX4I4YIsSpatialIAndXIsOOGroupO4,
  k2DX4O4YIsSpatialIAndXIsOOGroupI4,
};

struct WeightsDescription {
  DataType type;
  WeightsLayout layout;
  // applicable with layouts that have OGroup.
  int output_group_size;  // OGroup size
  // applicable with layouts that have CustomSpatial
  std::vector<int> spatial_remap;

  int GetOutputGroupSize() const;
  bool IsI4O4() const;
  bool IsO4I4() const;
  bool IsCustomSpatial() const;

  bool operator==(const WeightsDescription& t) const;
};

}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TASK_WEIGHTS_LAYOUT_H_
