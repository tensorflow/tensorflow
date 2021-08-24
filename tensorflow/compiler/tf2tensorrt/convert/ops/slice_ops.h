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
#ifndef TENSORFLOW_COMPILER_TF2TENSORRT_CONVERT_OPS_SLICE_OPS_H_
#define TENSORFLOW_COMPILER_TF2TENSORRT_CONVERT_OPS_SLICE_OPS_H_
#if GOOGLE_CUDA && GOOGLE_TENSORRT

#include "tensorflow/compiler/tf2tensorrt/convert/convert_nodes.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/util/strided_slice_op.h"

namespace tensorflow {
namespace tensorrt {
namespace convert {

using SliceDims = absl::InlinedVector<int64, 4>;

Status ConvertStridedSliceHelper(
    OpConverterParams* params, const TRT_TensorOrWeights& input,
    const PartialTensorShape& input_dims, const SliceDims& begin,
    const SliceDims& stride, const SliceDims& end,
    absl::optional<nvinfer1::Dims> final_shape = absl::nullopt,
    absl::optional<int> op_instance = absl::nullopt,
    absl::optional<StridedSliceShapeSpec> strided_slice_spec = absl::nullopt);

}  // namespace convert
}  // namespace tensorrt
}  // namespace tensorflow

#endif  // GOOGLE_CUDA && GOOGLE_TENSORRT
#endif  // TENSORFLOW_COMPILER_TF2TENSORRT_CONVERT_OPS_SLICE_OPS_H_
