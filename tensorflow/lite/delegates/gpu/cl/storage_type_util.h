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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_CL_STORAGE_TYPE_UTIL_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_CL_STORAGE_TYPE_UTIL_H_

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
                              const TensorDescriptor& descriptor);

bool CanCreateTensorWithShape(const CLContext& context, const CLDevice& device,
                              const BHWC& shape,
                              const TensorDescriptor& descriptor);

TensorStorageType SelectBestStorageType(const CLContext& context,
                                        const CLDevice& device,
                                        const BHWC& shape,
                                        const TensorStorageType& desired,
                                        const DataType& data_type,
                                        const Layout& layout);

}  // namespace cl
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_CL_STORAGE_TYPE_UTIL_H_
