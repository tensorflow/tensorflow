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

// Creates a strided slice operation using the given information. This function
// expects that the begin, stride, and end vectors have already been validated.
// This function converts the [begin:stride:end] specification to the TensorRT
// [begin:stride:size] ISliceLayer specification. The following algorithm is
// used to perform this conversion: 1) The given (input_dims,
// [begin:stride:end]) specification is dividied into
//  "static dimensions" and "dynamic dimensions". "Dynamic dimensions"
//  includes all dimensions of the slice where input_dims[i] == -1.
// 2a) If there are no dynamic dimensions, then the "begin", "stride", and
//  "size" variables are passed to the ISLiceLayer creation as build-time
//  constants in the form of nvinfer1::Dims objects.
// 2b) If there are any dynamic dimensions, then the "begin", "stride", and
//  "size" variables are treated as runtime dynamic shape Tensors in the
//  TensorRT graph. In this case, we must calculate "size" at runtime for all
//  dynamic dimensions, while static dimensions use the constant values.
//
// Note that when any dynamic indices are present (2b), the "strided_slice_spec"
// must be specified. This structure can be obtained through the
// "tensorflow::ValidateStridedSliceOp" function, or it can be constructed
// directly. When the ValidateStridedSliceOp helper function is used, it will
// also return the "begin", "stride", and "end" vectors. When all dimensions are
// static (2a), the "strided_slice_spec" variable is not required.
//
// If the "final_shape" variable is specified, then a reshape operation will be
// added to the graph to achieve this shape. The shape must be fully specified.
//
// "op_instance" is only required if the caller needs to pass this variable
// through to the Converter functions optionally accept it (SetLayerName,
// PrepareTensorForShape).
Status ConvertStridedSliceHelper(
    OpConverterParams* params, const TRT_TensorOrWeights& input,
    const PartialTensorShape& input_dims, const SliceDims& begin,
    const SliceDims& stride, const SliceDims& end,
    std::optional<nvinfer1::Dims> final_shape = std::nullopt,
    std::optional<int> op_instance = std::nullopt,
    std::optional<StridedSliceShapeSpec> strided_slice_spec = std::nullopt);

}  // namespace convert
}  // namespace tensorrt
}  // namespace tensorflow

#endif  // GOOGLE_CUDA && GOOGLE_TENSORRT
#endif  // TENSORFLOW_COMPILER_TF2TENSORRT_CONVERT_OPS_SLICE_OPS_H_
