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

#include "tensorflow/lite/delegates/gpu/cl/storage_type_util.h"

#include "tensorflow/lite/delegates/gpu/cl/cl_context.h"
#include "tensorflow/lite/delegates/gpu/cl/cl_device.h"
#include "tensorflow/lite/delegates/gpu/cl/tensor_type.h"
#include "tensorflow/lite/delegates/gpu/common/data_type.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"

namespace tflite {
namespace gpu {
namespace cl {

bool CanCreateTensorWithShape(const CLContext& context, const CLDevice& device,
                              const BHWDC& shape,
                              const TensorDescriptor& descriptor) {
  const int slices = DivideRoundUp(shape.c, 4);
  switch (descriptor.storage_type) {
    case TensorStorageType::BUFFER: {
      const int flt4_size =
          4 * (descriptor.data_type == DataType::FLOAT32 ? 4 : 2);
      const int buffer_size =
          shape.b * shape.w * shape.h * shape.d * slices * flt4_size;
      return buffer_size <= device.GetInfo().buffer_max_size;
    }
    case TensorStorageType::IMAGE_BUFFER:
      return shape.b * shape.w * shape.h * shape.d * slices <=
             device.GetInfo().image_buffer_max_size;
    case TensorStorageType::TEXTURE_3D:
      if (device.cl_version() < OpenCLVersion::CL_1_2 && slices == 1) {
        // clCreateImage3D (that used in CL 1.0/1.1) can not create image with
        // depth = 1 by specification;
        return false;
      }
      return shape.w * shape.b <= device.GetInfo().image3d_max_width &&
             shape.h <= device.GetInfo().image3d_max_height &&
             slices * shape.d <= device.GetInfo().image3d_max_depth;
    case TensorStorageType::TEXTURE_ARRAY:
      // Bug on some Adreno. b/131099086
      if (slices == 1 && !device.SupportsOneLayerTextureArray()) {
        return false;
      }
      return shape.w * shape.b <= device.GetInfo().image2d_max_width &&
             shape.h <= device.GetInfo().image2d_max_height &&
             slices * shape.d <= device.GetInfo().image_array_max_layers;
    case TensorStorageType::TEXTURE_2D:
      return shape.w * shape.b * shape.d <=
                 device.GetInfo().image2d_max_width &&
             shape.h * slices <= device.GetInfo().image2d_max_height;
    case TensorStorageType::SINGLE_TEXTURE_2D:
      return shape.c <= 4 &&
             context.IsFloatTexture2DSupported(shape.c, descriptor.data_type) &&
             shape.w * shape.b * shape.d <=
                 device.GetInfo().image2d_max_width &&
             shape.h <= device.GetInfo().image2d_max_height;
    default:
      return false;
  }
}

bool CanCreateTensorWithShape(const CLContext& context, const CLDevice& device,
                              const BHWC& shape,
                              const TensorDescriptor& descriptor) {
  const BHWDC shape5D(shape.b, shape.h, shape.w, 1, shape.c);
  return CanCreateTensorWithShape(context, device, shape5D, descriptor);
}

TensorStorageType SelectBestStorageType(const CLContext& context,
                                        const CLDevice& device,
                                        const BHWC& shape,
                                        const TensorStorageType& desired,
                                        const DataType& data_type,
                                        const Layout& layout) {
  if (CanCreateTensorWithShape(context, device, shape,
                               TensorDescriptor{data_type, desired, layout})) {
    return desired;
  }
  auto GetBestTypeAfterTextureArray = [&]() {
    if (device.SupportsImageBuffer() &&
        CanCreateTensorWithShape(
            context, device, shape,
            TensorDescriptor{data_type, TensorStorageType::IMAGE_BUFFER,
                             layout})) {
      return TensorStorageType::IMAGE_BUFFER;
    } else {
      return TensorStorageType::BUFFER;
    }
  };
  auto GetBestTypeAfterTexture2D = [&]() {
    if (device.SupportsTextureArray() &&
        CanCreateTensorWithShape(
            context, device, shape,
            TensorDescriptor{data_type, TensorStorageType::TEXTURE_ARRAY,
                             layout})) {
      return TensorStorageType::TEXTURE_ARRAY;
    } else {
      return GetBestTypeAfterTextureArray();
    }
  };
  auto GetBestTypeAfterTexture3D = [&]() {
    if (CanCreateTensorWithShape(
            context, device, shape,
            TensorDescriptor{data_type, TensorStorageType::TEXTURE_2D,
                             layout})) {
      return TensorStorageType::TEXTURE_2D;
    } else {
      return GetBestTypeAfterTexture2D();
    }
  };
  switch (desired) {
    case TensorStorageType::TEXTURE_2D:
    case TensorStorageType::SINGLE_TEXTURE_2D:
      return GetBestTypeAfterTexture2D();
    case TensorStorageType::TEXTURE_ARRAY:
      return GetBestTypeAfterTextureArray();
    case TensorStorageType::TEXTURE_3D:
      return GetBestTypeAfterTexture3D();
    case TensorStorageType::IMAGE_BUFFER:
    case TensorStorageType::BUFFER:
      return TensorStorageType::BUFFER;
    default:
      return TensorStorageType::BUFFER;
  }
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
