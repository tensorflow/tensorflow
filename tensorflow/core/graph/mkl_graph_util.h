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

#ifndef TENSORFLOW_CORE_GRAPH_MKL_GRAPH_UTIL_H_
#define TENSORFLOW_CORE_GRAPH_MKL_GRAPH_UTIL_H_
#ifdef INTEL_MKL

#include <string>
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {
// Since our ops are going to produce and also consume N addition tensors
// (Mkl) for N Tensorflow tensors, we can have following different
// orderings among these 2N tensors.
//
// E.g., for Tensorflow tensors A, B, and C, our ops will produce and
// consume A_m, B_m, and C_m additionally.
//
// INTERLEAVED: in this case 2N tensors are interleaved. So for above
//              example, the ordering looks like: A, A_m, B, B_m, C, C_m.
//
// CONTIGUOUS: in thi case N Tensorflow tensors are contiguous followed
//             by N Mkl tensors. So for above example, the ordering looks
//             like: A, B, C, A_m, B_m, C_m
//
// Following APIs map index of original Tensorflow tensors to their
// appropriate position based on selected ordering. For contiguous ordering,
// we need to know the total number of tensors (parameter total).
//
typedef enum { TENSORS_INTERLEAVED, TENSORS_CONTIGUOUS } MklTfTensorOrdering;
// NOTE: Currently, we use contiguous ordering. If you change this, then you
// would need to change Mkl op definitions in nn_ops.cc.
static MklTfTensorOrdering kTensorOrdering = TENSORS_CONTIGUOUS;

// Get index of MetaData tensor from index 'n' of Data tensor.
inline int DataIndexToMetaDataIndex(int n, int total_tensors) {
  if (kTensorOrdering == MklTfTensorOrdering::TENSORS_INTERLEAVED) {
    // For interleaved ordering, Mkl tensor follows immediately after
    // Tensorflow tensor.
    return n + 1;
  } else {
    CHECK_EQ(kTensorOrdering, MklTfTensorOrdering::TENSORS_CONTIGUOUS);
    // For contiguous ordering, Mkl tensor is n+total_tensors / 2 away.
    return n + total_tensors / 2;
  }
}

int inline GetTensorDataIndex(int n, int total_tensors) {
  if (kTensorOrdering == MklTfTensorOrdering::TENSORS_INTERLEAVED) {
    return 2 * n;  // index corresponding to nth input/output tensor
  } else {
    CHECK_EQ(kTensorOrdering, MklTfTensorOrdering::TENSORS_CONTIGUOUS);
    return n;
  }
}

int inline GetTensorMetaDataIndex(int n, int total_tensors) {
  // Get index for TensorData first and then use mapping function
  // to get TensorMetaData index from TensorData index.
  int tidx = GetTensorDataIndex(n, total_tensors);
  return DataIndexToMetaDataIndex(tidx, total_tensors);
}

namespace mkl_op_registry {
static const char* kMklOpLabel = "MklOp";
static const char* kMklOpLabelPattern = "label='MklOp'";
// Prefix that we add to Tensorflow op name to construct Mkl op name.
static const char* const kMklOpPrefix = "_Mkl";

// Get the name of Mkl op from original TensorFlow op
// We prefix 'Mkl' to the original op to get Mkl op.
inline string GetMklOpName(const string& name) {
  return string(kMklOpPrefix) + name;
}

// Check whether opname with type T is registered as MKL-compliant.
//
// @input: name of the op
// @input: T datatype to be used for checking op
// @return: true if opname is registered as Mkl op; false otherwise
static inline bool IsMklOp(const std::string& op_name, DataType T) {
  string kernel = KernelsRegisteredForOp(op_name);
  bool result =
      kernel.find(kMklOpLabelPattern) != string::npos && (T == DT_FLOAT);
  return result;
}

// Check whether opname with type T is registered as MKL-compliant and
// is element-wise.
//
// @input: name of the op
// @input: T datatype to be used for checking op
// @return: true if opname is registered as element-wise Mkl op;
// false otherwise
static inline bool IsMklElementWiseOp(const std::string& op_name, DataType T) {
  if (!IsMklOp(op_name, T)) {
    return false;
  }
  bool result = (0 == op_name.compare(GetMklOpName("Add")) ||
                 0 == op_name.compare(GetMklOpName("Sub")) ||
                 0 == op_name.compare(GetMklOpName("Mul")) ||
                 0 == op_name.compare(GetMklOpName("Maximum")) ||
                 0 == op_name.compare(GetMklOpName("SquaredDifference")));

  return result;
}
}  // namespace mkl_op_registry
}  // namespace tensorflow
#endif  // INTEL_MKL
#endif  // TENSORFLOW_CORE_GRAPH_MKL_GRAPH_UTIL_H_
