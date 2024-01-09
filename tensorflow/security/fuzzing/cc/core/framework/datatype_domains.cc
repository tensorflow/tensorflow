/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/security/fuzzing/cc/core/framework/datatype_domains.h"

namespace tensorflow::fuzzing {

fuzztest::Domain<DataType> AnyValidDataType() {
  return fuzztest::ElementOf({
      DT_FLOAT, DT_DOUBLE, DT_INT32, DT_UINT8, DT_INT16, DT_INT8, DT_INT64,
      DT_BOOL, DT_UINT16, DT_UINT32, DT_UINT64
      // TODO(b/268338352): add unsupported types
      // DT_STRING, DT_COMPLEX64, DT_QINT8, DT_QUINT8, DT_QINT32,
      // DT_BFLOAT16, DT_QINT16, DT_COMPLEX128, DT_HALF, DT_RESOURCE,
      // DT_VARIANT, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN
  });
}

}  // namespace tensorflow::fuzzing
