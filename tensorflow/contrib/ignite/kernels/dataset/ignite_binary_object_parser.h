/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CONTRIB_IGNITE_KERNELS_DATASET_IGNITE_BINARY_OBJECT_PARSER_H_
#define TENSORFLOW_CONTRIB_IGNITE_KERNELS_DATASET_IGNITE_BINARY_OBJECT_PARSER_H_

#include <vector>
#include "tensorflow/contrib/ignite/kernels/client/ignite_byte_swapper.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {

class BinaryObjectParser {
 public:
  BinaryObjectParser();
  Status Parse(uint8_t** ptr, std::vector<Tensor>* out_tensors,
               std::vector<int32_t>* types) const;

 private:
  uint8_t ParseByte(uint8_t** ptr) const;
  int16_t ParseShort(uint8_t** ptr) const;
  uint16_t ParseUnsignedShort(uint8_t** ptr) const;
  int32_t ParseInt(uint8_t** ptr) const;
  int64_t ParseLong(uint8_t** ptr) const;
  float ParseFloat(uint8_t** ptr) const;
  double ParseDouble(uint8_t** ptr) const;
  bool ParseBool(uint8_t** ptr) const;
  string ParseString(uint8_t** ptr) const;
  uint8_t* ParseByteArr(uint8_t** ptr, int length) const;
  int16_t* ParseShortArr(uint8_t** ptr, int length) const;
  uint16_t* ParseUnsignedShortArr(uint8_t** ptr, int length) const;
  int32_t* ParseIntArr(uint8_t** ptr, int length) const;
  int64_t* ParseLongArr(uint8_t** ptr, int length) const;
  float* ParseFloatArr(uint8_t** ptr, int length) const;
  double* ParseDoubleArr(uint8_t** ptr, int length) const;
  bool* ParseBoolArr(uint8_t** ptr, int length) const;

  const ByteSwapper byte_swapper_;
};

enum ObjectType {
  BYTE = 1,
  SHORT = 2,
  INT = 3,
  LONG = 4,
  FLOAT = 5,
  DOUBLE = 6,
  USHORT = 7,
  BOOL = 8,
  STRING = 9,
  DATE = 11,
  BYTE_ARR = 12,
  SHORT_ARR = 13,
  INT_ARR = 14,
  LONG_ARR = 15,
  FLOAT_ARR = 16,
  DOUBLE_ARR = 17,
  USHORT_ARR = 18,
  BOOL_ARR = 19,
  STRING_ARR = 20,
  DATE_ARR = 22,
  WRAPPED_OBJ = 27,
  COMPLEX_OBJ = 103
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CONTRIB_IGNITE_KERNELS_DATASET_IGNITE_BINARY_OBJECT_PARSER_H_
