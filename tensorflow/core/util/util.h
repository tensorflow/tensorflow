/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_UTIL_UTIL_H_
#define TENSORFLOW_CORE_UTIL_UTIL_H_

#include <string>

#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/platform/cpu_info.h"

namespace tensorflow {

// If op_name has '/' in it, then return everything before the first '/'.
// Otherwise return empty string.
StringPiece NodeNamePrefix(const StringPiece& op_name);

// If op_name has '/' in it, then return everything before the last '/'.
// Otherwise return empty string.
StringPiece NodeNameFullPrefix(const StringPiece& op_name);

class MovingAverage {
 public:
  explicit MovingAverage(int window);
  ~MovingAverage();

  void Clear();

  double GetAverage() const;
  void AddValue(double v);

 private:
  const int window_;  // Max size of interval
  double sum_;        // Sum over interval
  double* data_;      // Actual data values
  int head_;          // Offset of the newest statistic in data_
  int count_;         // # of valid data elements in window
};

// Returns a string printing bytes in ptr[0..n).  The output looks
// like "00 01 ef cd cd ef".
std::string PrintMemory(const char* ptr, size_t n);

// Given a flattened index into a tensor, computes a string s so that
// StrAppend("tensor", s) is a Python indexing expression.  E.g.,
// "tensor", "tensor[i]", "tensor[i, j]", etc.
std::string SliceDebugString(const TensorShape& shape, int64_t flat);

// Check if MKL is enabled in runtime
bool IsMKLEnabled();

// Flag a warning if input type is unsupported on CPU when oneDNN is enabled
void DataTypeUnsupportedWarning(const DataType& dt);

// Check if input type is supported on CPU when oneDNN is enabled
bool IsDataTypeSupportedByOneDNNOnThisCPU(const DataType& dt);

// Check if input type supports AMX on CPU when oneDNN is enabled
bool IsAMXDataTypeSupportedByOneDNNOnThisCPU(const DataType& dt);

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_UTIL_UTIL_H_
