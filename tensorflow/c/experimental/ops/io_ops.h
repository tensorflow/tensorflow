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

// This file is MACHINE GENERATED! Do not edit.

#ifndef TENSORFLOW_C_EXPERIMENTAL_OPS_IO_OPS_H_
#define TENSORFLOW_C_EXPERIMENTAL_OPS_IO_OPS_H_

#include "absl/status/status.h"
#include "absl/types/span.h"
#include "tensorflow/c/eager/abstract_context.h"
#include "tensorflow/c/eager/abstract_tensor_handle.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/status.h"

namespace tensorflow {
namespace ops {

// Restores tensors from a V2 checkpoint.
absl::Status RestoreV2(AbstractContext* ctx, AbstractTensorHandle* const prefix,
                       AbstractTensorHandle* const tensor_names,
                       AbstractTensorHandle* const shape_and_slices,
                       absl::Span<AbstractTensorHandle*> tensors,
                       absl::Span<DataType> dtypes, const char* name = nullptr,
                       const char* raw_device_name = nullptr);

// Saves tensors in V2 checkpoint format.
absl::Status SaveV2(AbstractContext* ctx, AbstractTensorHandle* const prefix,
                    AbstractTensorHandle* const tensor_names,
                    AbstractTensorHandle* const shape_and_slices,
                    absl::Span<AbstractTensorHandle* const> tensors,
                    const char* name = nullptr,
                    const char* raw_device_name = nullptr);

}  // namespace ops
}  // namespace tensorflow

#endif  // TENSORFLOW_C_EXPERIMENTAL_OPS_IO_OPS_H_
