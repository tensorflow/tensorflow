// Copyright 2024 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_QUALCOMM_COMPILER_LEGALIZATIONS_LEGALIZATION_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_QUALCOMM_COMPILER_LEGALIZATIONS_LEGALIZATION_H_

#include "third_party/qairt/latest/include/QNN/QnnTypes.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/cc/litert_model.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/compiler/graph_mapper.h"

#define STRINGIFY(x) #x
#define QNN_OP_NAME(prefix) STRINGIFY(prefix##__COUNTER__)

namespace litert::qnn {

class Legalization {
 public:
  Legalization() = default;
  virtual ~Legalization() = default;

  virtual LiteRtStatus LegalizeOp(const Op& src, Qnn_OpConfig_t& dest,
                                  GraphMapper& graph_mapper) = 0;

  // Sets the op name, package name, and type.
  // Note: All argument strings can't be de-allocated until the op has been
  // registered with the qnn api. i.e graphAddNode().
  inline LiteRtStatus SetOpInfo(const char* name, const char* op_package_name,
                                const char* op_type, Qnn_OpConfig_t& op) {
    op.v1.name = name;
    op.v1.packageName = op_package_name;
    op.v1.typeName = op_type;
    return kLiteRtStatusOk;
  }
};

}  // namespace litert::qnn

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_QUALCOMM_COMPILER_LEGALIZATIONS_LEGALIZATION_H_
