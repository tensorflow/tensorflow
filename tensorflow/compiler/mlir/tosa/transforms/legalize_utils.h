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

#ifndef TENSORFLOW_COMPILER_MLIR_TOSA_TRANSFORMS_LEGALIZE_UTILS_H
#define TENSORFLOW_COMPILER_MLIR_TOSA_TRANSFORMS_LEGALIZE_UTILS_H

#include <climits>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <numeric>

#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/core/framework/kernel_shape_util.h"
#include "tensorflow/core/kernels/conv_grad_shape_utils.h"
#include "tensorflow/core/util/padding.h"
#include "tensorflow/core/util/tensor_format.h"

namespace mlir {
namespace tosa {

// Create a TOSA rescale op from TFLite scaling, zero points and rounding mode
Value buildRescale(PatternRewriter& rewriter, Operation* op,
                   RankedTensorType output_type, Value input_val, double scale,
                   int64_t input_zp, int64_t output_zp,
                   bool double_round = false);

// Creates TOSA rescale op with int32 output
Value buildRescaleToInt32(PatternRewriter& rewriter, Operation* op,
                          Value input_val, double input_scale,
                          int64_t input_zp);

// Creates TOSA rescale op with int32 input
Value buildRescaleFromInt32(PatternRewriter& rewriter, Operation* op,
                            RankedTensorType output_type, Value input_val,
                            double output_scale, int64_t output_zp);

// Creates a TOSA rescale op based on conv2d parameters.
Value buildRescaleOpConvOutput(PatternRewriter& rewriter, Operation* op,
                               Value conv_val, RankedTensorType input_type,
                               RankedTensorType weight_type,
                               RankedTensorType output_type);

// Create a 513 entry TOSA constant tensor suitable for the Table operator based
// on the values from an int32_t func(int32_t) lambda function.
Value getTosa1DConstTensorTable(PatternRewriter& rewriter, Operation* op,
                                std::function<int32_t(int32_t)> func);

// Create a 32-bit float constant operator from a float
Value getTosaConstTensorSingleF32(PatternRewriter& rewriter, Operation* op,
                                  float val);

// Create a 32-bit integer constant operator from an int
Value getTosaConstTensorSingleI32(PatternRewriter& rewriter, Operation* op,
                                  int32_t val);

// Create a vector from a 32-bit value tensor.  Returns vector size on success
// or -1 on error.
int getVectorFromValue32(Value val, SmallVector<int32_t, 4>& vec);

// Calculates the TOSA padding values based on TF operators padded with
// SAME/VALID.
bool getPaddingValuesFromPadType(
    tensorflow::Padding tf_pad, tensorflow::TensorFormat data_format_tf,
    uint32_t first_filter_spatial_dim, RankedTensorType input_type,
    RankedTensorType filter_type, ArrayAttr strides, ArrayAttr dilations,
    PatternRewriter& rewriter, ArrayAttr& explicit_pad);

// Calculates the TOSA padding values for explicit-padded TF operators.
ArrayAttr getPaddingValuesFromExplicitPadAttr(
    ArrayAttr explicit_pad, tensorflow::TensorFormat data_format_tf,
    PatternRewriter& rewriter);

// Calculates the TOSA padding values for transposeConv2d
bool getTransposeConv2dPaddingValues(
    tensorflow::Padding tf_pad, tensorflow::TensorFormat data_format_tf,
    uint32_t first_filter_spatial_dim, RankedTensorType input_type,
    RankedTensorType filter_type, RankedTensorType output_type,
    ArrayAttr strides, ArrayAttr dilations, PatternRewriter& rewriter,
    ArrayAttr& explicit_pad);

// Templated function to create a constant op in a given dialect and with a
// given type.  Specializations below.

// T0: target dialect constant op
// T1: native c++ integer type
template <typename T0, typename T1>
Value get1DConstTensor(PatternRewriter& rewriter, Operation* op,
                       SmallVector<T1, 8> arr);

// Same as get1DConstTensor, but int48 is not native c++ type, needs additional
// interface
Value get1DConstTensorInt48(PatternRewriter& rewriter, Operation* op,
                            SmallVector<int64_t, 8> arr);

// Strip off quantization information for bias tensor and return a unquantized
// bias
Value getUnquantizedBias(PatternRewriter& rewriter, Operation* op, Value input);

}  // namespace tosa
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_TOSA_TRANSFORMS_LEGALIZE_UTILS_H
