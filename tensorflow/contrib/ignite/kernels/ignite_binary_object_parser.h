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

#include <vector>
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {

class BinaryObjectParser {
 public:
  Status Parse(uint8_t** ptr, std::vector<Tensor>* out_tensors,
               std::vector<int32_t>* types);
};

enum ObjectType {
  BYTE = 1,
  SHORT = 2,
  INT = 3,
  LONG = 4,
  FLOAT = 5,
  DOUBLE = 6,
  UCHAR = 7,
  BOOL = 8,
  STRING = 9,
  DATE = 11,
  BYTE_ARR = 12,
  SHORT_ARR = 13,
  INT_ARR = 14,
  LONG_ARR = 15,
  FLOAT_ARR = 16,
  DOUBLE_ARR = 17,
  UCHAR_ARR = 18,
  BOOL_ARR = 19,
  STRING_ARR = 20,
  DATE_ARR = 22,
  WRAPPED_OBJ = 27,
  COMPLEX_OBJ = 103
};

}  // namespace tensorflow
