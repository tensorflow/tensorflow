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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_PROCESS_BROADCAST_SHAPES_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_PROCESS_BROADCAST_SHAPES_H_

#include <algorithm>

#include "tensorflow/lite/kernels/internal/types.h"

namespace tflite {

namespace reference_ops {

// Consolidates dimensions in broadcast inputs, checks for five-fold pattern.
//
// For example, if sequence of dimensions of one input is
// ..., 1, 3, 1, 7, 9, 5,... and the other is ..., 2, 3, 1, 7, 1, 1, ...
// we can consolidate these as
// ..., 1, 3*7, 9*5, ... and 2, 3*7, 1.
//
// The category is updated in the less-frequent case of shapes that are
// not suited to a fivefold-loop broadcast.
//
// Falls back to generic pattern when it does not know how to process properly.
//
// Returns true iff there is some sort of broadcast, which includes five-fold
// patterns and falling back to generic broadcast.
inline bool ProcessBroadcastShapes(const RuntimeShape& shape0,
                                   const RuntimeShape& shape1,
                                   tflite::ArithmeticParams* params) {
  const int dims_count =
      std::max(shape0.DimensionsCount(), shape1.DimensionsCount());

  params->broadcast_category = BroadcastableOpCategory::kGenericBroadcast;
  RuntimeShape scalar_shape(dims_count, 1);

  auto extended_shape0 = RuntimeShape::ExtendedShape(dims_count, shape0);
  auto extended_shape1 = RuntimeShape::ExtendedShape(dims_count, shape1);

  // Check for "exact" match, implicitly accepting any scalar shapes.
  if (extended_shape0 == extended_shape1) {
    params->broadcast_category = BroadcastableOpCategory::kNonBroadcast;
    return false;
  }

  for (int i = dims_count - 1; i >= 0; --i) {
    if (extended_shape0.Dims(i) == extended_shape1.Dims(i)) {
      continue;
    } else if (extended_shape0.Dims(i) == 1) {
      params->broadcast_category =
          BroadcastableOpCategory::kFirstInputBroadcastsFast;
      break;
    } else if (extended_shape1.Dims(i) == 1) {
      params->broadcast_category =
          BroadcastableOpCategory::kSecondInputBroadcastsFast;
      break;
    } else {
      // This case is erroneous: there is a dimension that does not match and
      // is not a broadcast from one shape to the other.
      params->broadcast_category = BroadcastableOpCategory::kGenericBroadcast;
      return true;
    }
  }

  if (params->broadcast_category !=
          BroadcastableOpCategory::kFirstInputBroadcastsFast &&
      params->broadcast_category !=
          BroadcastableOpCategory::kSecondInputBroadcastsFast) {
    // This is unreachable because at least one else clause in the above loop
    // must be reached.
    TFLITE_DCHECK(false);
    params->broadcast_category = BroadcastableOpCategory::kNonBroadcast;
    return false;
  }

  // From this point it is assumed contractually that corresponding dimensions
  // in shape0 and shape1 are either (a) equal or (b) one or other equals 1.
  const bool swap_inputs = params->broadcast_category ==
                           BroadcastableOpCategory::kSecondInputBroadcastsFast;
  const RuntimeShape* shape_a =
      swap_inputs ? &extended_shape1 : &extended_shape0;
  const RuntimeShape* shape_b =
      swap_inputs ? &extended_shape0 : &extended_shape1;

  int i = dims_count - 1;
  params->broadcast_shape[0] = 1;
  params->broadcast_shape[1] = 1;
  params->broadcast_shape[2] = 1;
  params->broadcast_shape[3] = 1;
  params->broadcast_shape[4] = 1;
  // y_0 is greedy: include dims if both or neither equal 1: in other words,
  // test for equality rather than (shape_a->Dims(i) != 1).
  while (i >= 0 && shape_a->Dims(i) == shape_b->Dims(i)) {
    params->broadcast_shape[4] *= shape_b->Dims(i);
    --i;
  }
  // Here either input_a or input_b has dim of 1 (if i >= 0).  If it is input_b
  // that has the unit dimension, the next two loops are not entered.
  while (i >= 0 && shape_a->Dims(i) == 1) {
    params->broadcast_shape[3] *= shape_b->Dims(i);
    --i;
  }
  while (i >= 0 && shape_a->Dims(i) == shape_b->Dims(i)) {
    params->broadcast_shape[2] *= shape_a->Dims(i);
    --i;
  }
  // Here either input_a or input_b has dim of 1 (if i >= 0).
  while (i >= 0 && shape_b->Dims(i) == 1) {
    params->broadcast_shape[1] *= shape_a->Dims(i);
    --i;
  }
  while (i >= 0 && shape_a->Dims(i) == shape_b->Dims(i)) {
    params->broadcast_shape[0] *= shape_b->Dims(i);
    --i;
  }

  // Rarer case is when the broadcast dimensions cannot be handled by a fivefold
  // loop.
  if (i >= 0) {
    params->broadcast_category = BroadcastableOpCategory::kGenericBroadcast;
  }
  return true;
}

}  // namespace reference_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_PROCESS_BROADCAST_SHAPES_H_
