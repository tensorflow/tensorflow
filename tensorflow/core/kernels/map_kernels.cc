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
#include "tensorflow/core/kernels/map_kernels.h"

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/variant_encode_decode.h"

namespace tensorflow {

REGISTER_KERNEL_BUILDER(Name("EmptyTensorMap").Device(DEVICE_CPU),
                        EmptyTensorMap);

REGISTER_KERNEL_BUILDER(Name("TensorMapSize").Device(DEVICE_CPU),
                        TensorMapSize);

REGISTER_KERNEL_BUILDER(Name("TensorMapLookup").Device(DEVICE_CPU),
                        TensorMapLookup);

REGISTER_KERNEL_BUILDER(Name("TensorMapInsert").Device(DEVICE_CPU),
                        TensorMapInsert);

REGISTER_KERNEL_BUILDER(Name("TensorMapErase").Device(DEVICE_CPU),
                        TensorMapErase);

REGISTER_KERNEL_BUILDER(Name("TensorMapHasKey").Device(DEVICE_CPU),
                        TensorMapHasKey);

}  // namespace tensorflow
