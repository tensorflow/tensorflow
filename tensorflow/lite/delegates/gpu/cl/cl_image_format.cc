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

#include "tensorflow/lite/delegates/gpu/cl/cl_image_format.h"

namespace tflite {
namespace gpu {
namespace cl {

cl_channel_order ToChannelOrder(int num_channels) {
  switch (num_channels) {
    case 1:
      return CL_R;
    case 2:
      return CL_RG;
    case 3:
      return CL_RGB;
    case 4:
      return CL_RGBA;
    default:
      return -1;
  }
}

cl_channel_type ToImageChannelType(DataType data_type) {
  switch (data_type) {
    case DataType::FLOAT32:
      return CL_FLOAT;
    case DataType::FLOAT16:
      return CL_HALF_FLOAT;
    default:
      return -1;
  }
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
