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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LRT_VENDORS_QUALCOMM_COMPILER_LEGALIZATIONS_UTIL_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LRT_VENDORS_QUALCOMM_COMPILER_LEGALIZATIONS_UTIL_H_

#include "third_party/qairt/latest/include/QNN/QnnTypes.h"
#include "tensorflow/lite/experimental/lrt/c/litert_common.h"
#include "tensorflow/lite/experimental/lrt/cc/litert_op.h"
#include "tensorflow/lite/experimental/lrt/vendors/qualcomm/compiler/graph_mapper.h"

namespace litert::qnn {

// Use this function to legalize a LiteRtOp to a Qnn Op when:
// 1. Source input/output tensor and destination input/ouptut tensor are 1 : 1
// mapped
// 2. Assigning params to destination OP does not depending on input tensor of
// source OP.
LiteRtStatus LegalizeSimpleOp(LiteRtOpManager& src, Qnn_OpConfig_t& dest,
                              GraphMapper& graph_mapper);

}  // namespace litert::qnn

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LRT_VENDORS_QUALCOMM_COMPILER_LEGALIZATIONS_UTIL_H_
