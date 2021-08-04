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

#include "tensorflow/lite/delegates/gpu/common/task/weights_layout.h"

namespace tflite {
namespace gpu {

int WeightsDescription::GetOutputGroupSize() const {
  if (layout == WeightsLayout::kOSpatialIOGroupI4O4 ||
      layout == WeightsLayout::kOSpatialIOGroupO4I4 ||
      layout == WeightsLayout::k2DX4I4YIsSpatialIAndXIsOOGroupO4 ||
      layout == WeightsLayout::k2DX4O4YIsSpatialIAndXIsOOGroupI4) {
    return output_group_size;
  } else {
    return 1;
  }
}

bool WeightsDescription::IsI4O4() const {
  return layout == WeightsLayout::kOSpatialIOGroupI4O4 ||
         layout == WeightsLayout::kOICustomSpatialI4O4 ||
         layout == WeightsLayout::k2DX4I4YIsSpatialIAndXIsOOGroupO4;
}

bool WeightsDescription::IsO4I4() const {
  return layout == WeightsLayout::kOSpatialIOGroupO4I4 ||
         layout == WeightsLayout::kOICustomSpatialO4I4 ||
         layout == WeightsLayout::k2DX4O4YIsSpatialIAndXIsOOGroupI4;
}

}  // namespace gpu
}  // namespace tflite
