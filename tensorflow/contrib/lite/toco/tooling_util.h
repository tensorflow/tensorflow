/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#ifndef THIRD_PARTY_TENSORFLOW_CONTRIB_LITE_TOCO_TOOLING_UTIL_H_
#define THIRD_PARTY_TENSORFLOW_CONTRIB_LITE_TOCO_TOOLING_UTIL_H_

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "google/protobuf/text_format.h"
#include "tensorflow/core/platform/logging.h"
#if TOCO_SUPPORT_PORTABLE_PROTOS
#include "third_party/protobuf/src/google/protobuf/text_format.h"
#endif  // TOCO_SUPPORT_PORTABLE_PROTOS
#include "tensorflow/contrib/lite/toco/model.h"
#include "tensorflow/contrib/lite/toco/model_flags.pb.h"
#include "tensorflow/contrib/lite/toco/runtime/types.h"
#include "tensorflow/contrib/lite/toco/toco_flags.pb.h"
#include "tensorflow/contrib/lite/toco/toco_port.h"
#include "tensorflow/contrib/lite/toco/types.pb.h"

// TODO(aselle): Replace with using a container specific hash override instead.
namespace std {
template <>
struct hash<toco::OperatorType> {
  size_t operator()(const toco::OperatorType& op) const {
    return std::hash<size_t>()(static_cast<size_t>(op));
  }
};
}  // namespace std

namespace toco {

constexpr int kLogLevelModelChanged = 1;
constexpr int kLogLevelModelUnchanged = 2;

string LogName(const Operator& op);

bool IsInputArray(const Model& model, const string& name);
bool IsArrayConsumed(const Model& model, const string& name);
int CountTrueOutputs(const Model& model, const Operator& op);

int CountOpsWithInput(const Model& model, const string& array_name);
bool DeleteArrayIfUnused(const string& array_name, Model* model);

std::vector<std::unique_ptr<Operator>>::const_iterator FindOpWithOutput(
    const Model& model, const string& array_name);
Operator* GetOpWithOutput(const Model& model, const string& array_name);

std::vector<std::unique_ptr<Operator>>::iterator FindOpWithOutput(
    Model& model, const string& array_name);
Operator* GetOpWithOutput(const Model& model, const string& array_name);

std::vector<std::unique_ptr<Operator>>::const_iterator FindOpWithInput(
    const Model& model, const string& array_name);
Operator* GetOpWithInput(const Model& model, const string& array_name);
Operator* GetFirstOpWithInput(const Model& model, const string& array_name);

std::vector<std::unique_ptr<Operator>>::const_iterator FindOp(
    const Model& model, const Operator* op);
std::vector<std::unique_ptr<Operator>>::iterator FindOp(Model& model,
                                                        const Operator* op);

const char* OperatorTypeName(OperatorType type);
string HelpfulOperatorTypeName(const Operator& op);

void DumpGraphvizVideoFrame(const Model& model);
void LogDump(int log_level, const string& message, const Model& model);
void LogSummary(int log_level, const string& message, const Model& model);

inline bool ParseFromStringOverload(const std::string& in,
                                    TFLITE_PROTO_NS::Message* proto) {
  return TFLITE_PROTO_NS::TextFormat::ParseFromString(in, proto);
}

template <typename Proto>
bool ParseFromStringEitherTextOrBinary(const std::string& input_file_contents,
                                       Proto* proto) {
  if (proto->ParseFromString(input_file_contents)) {
    return true;
  }

  if (ParseFromStringOverload(input_file_contents, proto)) {
    return true;
  }

  return false;
}

// TODO(b/36075966): Clean up when dims superseded by array shape.
void ExtendShape(Shape* shape, int new_shape_size);

// TODO(b/36075966): Clean up when dims superseded by array shape.
void UnextendShape(Shape* shape, int new_shape_size);

// Checks (using CHECK) that all dimensions of 'shape' are at least 1.
void CheckShapeDimensions(const Shape& shape);

// Given two shapes with potentially different dimensionality and dimension
// arrays d0 and d1. Without loss of generality, assume that shape0 may have
// higher dimensionality (length(d0) >= length(d1)). Then shape0 and shape1
// "agree up to broadcasting" if:
// - When walking the d0 and d1 from back to front with indices i0, i1,
//   d0[i0] == d1[i1] or d0[i0] == 1 or d1[i1] == 1, for each dimension until
//   i1 == 0 (inclusive).
bool ShapesAgreeUpToBroadcasting(const Shape& shape0, const Shape& shape1);

// A stricter constraint than ShapesAgreeUpToBroadcasting().
//
// Given two shapes with potentially different dimensionality and dimension
// arrays d0 and d1. Without loss of generality, assume that shape0 may have
// higher dimensionality (length(d0) >= length(d1)). Then shape0 and shape1
// "agree up to extending" if:
// - When walking the d0 and d1 from back to front with indices i0, i1,
//   d0[i0] == d1[i1] for each dimension until i1 == 0 (inclusive).
// - For the remaining indices [0..i0), d0[i0] == 1.
bool ShapesAgreeUpToExtending(const Shape& shape0, const Shape& shape1);

bool IsArrayFullyConnectedWeights(const Model& model, const string& name);

// If there is a wildcard dimension (-1), this may return a negative value.
int RequiredBufferSizeForShape(const Shape& shape);

bool IsConstantParameterArray(const Model& model, const string& name);

void CheckNoMissingArray(const Model& model);
void CheckInvariants(const Model& model);

void CheckModelCounts(const Model& model);

void FixOperatorOrdering(Model* model);
void FixNoMissingArray(Model* model);
void FixNoOrphanedArray(Model* model);

void ResolveModelFlags(const ModelFlags& model_flags, Model* model);

template <ArrayDataType A>
void GetQuantizationParamsFromMinMax(const ModelFlags& model_flags,
                                     const MinMax& minmax,
                                     QuantizationParams* quantization_params) {
  using Integer = DataType<A>;
  const Integer qmin = std::numeric_limits<Integer>::min();
  const Integer qmax = std::numeric_limits<Integer>::max();
  const double qmin_double = qmin;
  const double qmax_double = qmax;
  const double rmin = minmax.min;
  const double rmax = minmax.max;
  // 0 should always be a representable value. Let's assume that the initial
  // min,max range contains 0.
  CHECK_LE(rmin, 0.);
  CHECK_GE(rmax, 0.);
  if (rmin == rmax) {
    // Special case where the min,max range is a point. Should be {0}.
    CHECK_EQ(rmin, 0.);
    CHECK_EQ(rmax, 0.);
    quantization_params->zero_point = 0;
    quantization_params->scale = 0.;
    return;
  }

  // General case.
  //
  // First determine the scale.
  const double scale = (rmax - rmin) / (qmax_double - qmin_double);

  // Zero-point computation.
  // First the initial floating-point computation. The zero-point can be
  // determined from solving an affine equation for any known pair
  // (real value, corresponding quantized value).
  // We know two such pairs: (rmin, qmin) and (rmax, qmax).
  // The arithmetic error on the zero point computed from either pair
  // will be roughly machine_epsilon * (sum of absolute values of terms)
  // so we want to use the variant that adds the smaller terms.
  const double zero_point_from_min = qmin_double - rmin / scale;
  const double zero_point_from_max = qmax_double - rmax / scale;
  const double zero_point_from_min_error =
      std::abs(qmin_double) + std::abs(rmin / scale);
  const double zero_point_from_max_error =
      std::abs(qmax_double) + std::abs(rmax / scale);

  const double zero_point_double =
      zero_point_from_min_error < zero_point_from_max_error
          ? zero_point_from_min
          : zero_point_from_max;

  // Now we need to nudge the zero point to be an integer
  // (our zero points are integer, and this is motivated by the requirement
  // to be able to represent the real value "0" exactly as a quantized value,
  // which is required in multiple places, for example in Im2col with SAME
  // padding).
  Integer nudged_zero_point = 0;
  if (zero_point_double < qmin_double) {
    nudged_zero_point = qmin;
  } else if (zero_point_double > qmax_double) {
    nudged_zero_point = qmax;
  } else {
    nudged_zero_point = static_cast<Integer>(std::round(zero_point_double));
  }
  // The zero point should always be in the range of quantized value,
  // [qmin, qmax].
  CHECK_GE(nudged_zero_point, qmin);
  CHECK_LE(nudged_zero_point, qmax);

  // Finally, store the result nudged quantization params.
  quantization_params->zero_point = nudged_zero_point;
  quantization_params->scale = scale;
}

void CheckIsReadyForQuantization(const Model& model);
void UseDefaultMinMaxRangeValues(Model* model, double default_ranges_min,
                                 double default_ranges_max);

inline int Offset(const Shape& shape, const std::vector<int>& indices) {
  DCHECK_EQ(shape.dimensions_count(), indices.size());
  const int dims_count = shape.dimensions_count();
  int offset = 0;
  for (int i = 0; i < dims_count; i++) {
    const int index = indices[i];
    DCHECK(index >= 0 && index < shape.dims(i));
    offset *= shape.dims(i);
    offset += index;
  }
  return offset;
}

inline std::vector<int> ReverseOffset(const Shape& shape, int index) {
  DCHECK_GE(index, 0);
  DCHECK_LT(index, RequiredBufferSizeForShape(shape));
  const int dims_count = shape.dimensions_count();
  std::vector<int> indices(dims_count);
  int residual = index;
  for (int i = dims_count - 1; i >= 0; i--) {
    indices[i] = residual % shape.dims(i);
    residual /= shape.dims(i);
  }
  return indices;
}

int ElementSize(ArrayDataType data_type);

void DropMinMax(Model* model, const string& array_name);

bool IsAllocatableTransientArray(const Model& model, const string& array_name);

void CreateOrCheckRnnStateArray(const string& name, int size, Model* model);

string AvailableArrayName(const Model& model, const string& name);

// Formats a shape as a string: [ dims(0), dims(1), ..., dims(num_dims-1) ].
string ShapeToString(const Shape& shape);

void PrintArrayShape(Model* model, const string& name);

void MakeArrayDims(int num_dims, int batch, int height, int width, int depth,
                   std::vector<int>* out_dims);

bool EstimateArithmeticOpsCount(const Model& model, int64* result);

int AxesCount(AxesOrder axes_order);

void ShuffleDims(const Shape& input_shape, AxesOrder input_axes_order,
                 AxesOrder output_axes_order, Shape* output_shape);
void ShuffleArray(const Shape& input_shape, AxesOrder input_axes_order,
                  AxesOrder output_axes_order, const Shape& output_shape,
                  const float* input_data, float* output_data);

// Returns true if it may be OK for any graph transformation to ever discard
// that array. The idea is that we can't ever discard arrays that are either
// an input or an output of the whole graph, or that appear in RNN back-edges,
// as that would undercut explicit flags that the user might pass.
bool IsDiscardableArray(const Model& model, const string& array_name);

void CheckFinalDataTypesSatisfied(const Model& model);

ArrayDataType ConvertIODataTypeToArrayDataType(IODataType type);

}  // namespace toco

#endif  // THIRD_PARTY_TENSORFLOW_CONTRIB_LITE_TOCO_TOOLING_UTIL_H_
