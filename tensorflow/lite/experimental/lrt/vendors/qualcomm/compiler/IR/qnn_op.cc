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

#include "tensorflow/lite/experimental/lrt/vendors/qualcomm/compiler/IR/qnn_op.h"

#include "third_party/qairt/include/QNN/QnnTypes.h"
#include "tensorflow/lite/experimental/lrt/c/lite_rt_common.h"
#include "tensorflow/lite/experimental/lrt/c/lite_rt_model.h"
#include "tensorflow/lite/experimental/lrt/c/lite_rt_op_code.h"
#include "tensorflow/lite/experimental/lrt/c/lite_rt_support.h"
#include "tensorflow/lite/experimental/lrt/cc/lite_rt_op.h"

// A macro dance to create a unique literal string given a prefix.
#define STRINGIFY(x) #x
#define QNN_OP_NAME(prefix) STRINGIFY(prefix##__COUNTER)

namespace lrt::qnn {

using ::lrt::LrtOpManager;

namespace {

// Maps "op-code" related information (name, packageName, typeName) from src
// to dest.
LrtStatus LegalizeOpType(const LrtOpManager& src, Qnn_OpConfig_t& dest) {
  switch (src.Code()) {
    case kLrtOpCodeTflMul:
      dest.v1.name = QNN_OP_NAME(mul_);
      dest.v1.packageName = "qti.aisw";
      dest.v1.typeName = "ElementWiseMultiply";
      break;
    default:
      return kLrtStatusErrorUnsupported;
  }
  return kLrtStatusOk;
}

}  // namespace

Qnn_OpConfig_t BuildDefaultOp() {
  Qnn_OpConfig_t op = QNN_OPCONFIG_INIT;
  ResetOp(op);
  return op;
}

void ResetOp(Qnn_OpConfig_t& op) {
  op = QNN_OPCONFIG_INIT;
  op.version = QNN_OPCONFIG_VERSION_1;
  op.v1 = QNN_OPCONFIG_V1_INIT;
}

LrtStatus LegalizeOp(LrtOp src, Qnn_OpConfig_t& dest) {
  ResetOp(dest);

  LrtOpManager::Unique src_op;
  LRT_RETURN_STATUS_IF_NOT_OK(LrtOpManager::MakeFromOp(src, src_op));

  LRT_RETURN_STATUS_IF_NOT_OK(LegalizeOpType(*src_op, dest));

  return kLrtStatusOk;
}

}  // namespace lrt::qnn
