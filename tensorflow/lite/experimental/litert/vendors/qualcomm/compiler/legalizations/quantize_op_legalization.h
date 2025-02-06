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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_QUALCOMM_COMPILER_LEGALIZATIONS_QUANTIZE_OP_LEGALIZATION_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_QUALCOMM_COMPILER_LEGALIZATIONS_QUANTIZE_OP_LEGALIZATION_H_

#include <alloca.h>
#include <stdio.h>

#include <cstdint>
#include <memory>

#include "third_party/qairt/latest/include/QNN/QnnTypes.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/cc/litert_model.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/compiler/graph_mapper.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/compiler/legalizations/legalization.h"

namespace litert::qnn {

class QuantizeOpLegalization : public Legalization {
 public:
  QuantizeOpLegalization() = default;
  ~QuantizeOpLegalization() = default;
  using Ptr = std::unique_ptr<QuantizeOpLegalization>;
  static Ptr Create() { return std::make_unique<QuantizeOpLegalization>(); }

  LiteRtStatus LegalizeOp(const litert::Op& src, Qnn_OpConfig_t& dest,
                          GraphMapper& graph_mapper);

  LiteRtStatus ConfigureQnnOp(const litert::Op& src, Qnn_OpConfig_t& dest);

 private:
  // Requantization: legalize quantize to QNN Convert Op.
  // Quantization range is not within QNN cast Op range.
  LiteRtStatus LegalizeQuantizeOpAsConvertOp(const litert::Op& src,
                                             Qnn_OpConfig_t& dest);

  // Ignore Requantization: legalize quantize to QNN Cast Op.
  // Quantization range is within QNN cast Op range. Directly use QNN Cast Op.
  LiteRtStatus LegalizeQuantizeOpAsCastOp(const litert::Op& src,
                                          Qnn_OpConfig_t& dest);

  // Quantization: legalize quantize to QNN Quantize Op.
  LiteRtStatus LegalizeQuantizeOpAsQuantizeOp(const litert::Op& src,
                                              Qnn_OpConfig_t& dest);

  // Counter to ensure unique op names.
  uint32_t op_counter_ = 0;
};

}  // namespace litert::qnn

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_QUALCOMM_COMPILER_LEGALIZATIONS_QUANTIZE_OP_LEGALIZATION_H_
