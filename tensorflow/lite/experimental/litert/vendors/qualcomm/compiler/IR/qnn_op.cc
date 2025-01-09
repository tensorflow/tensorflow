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

#include "tensorflow/lite/experimental/litert/vendors/qualcomm/compiler/IR/qnn_op.h"

#include "third_party/qairt/latest/include/QNN/QnnTypes.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_model.h"
#include "tensorflow/lite/experimental/litert/c/litert_op_code.h"
#include "tensorflow/lite/experimental/litert/cc/litert_model.h"

// A macro dance to create a unique literal string given a prefix.
#define STRINGIFY(x) #x
#define QNN_OP_NAME(prefix) STRINGIFY(prefix##__COUNTER)

namespace litert::qnn {

namespace {

// Maps "op-code" related information (name, packageName, typeName) from src
// to dest.
LiteRtStatus LegalizeOpType(const Op& src, Qnn_OpConfig_t& dest) {
  switch (src.Code()) {
    case kLiteRtOpCodeTflMul:
      dest.v1.name = QNN_OP_NAME(mul_);
      dest.v1.packageName = "qti.aisw";
      dest.v1.typeName = "ElementWiseMultiply";
      break;
    case kLiteRtOpCodeTflAdd:
      dest.v1.name = QNN_OP_NAME("add");
      dest.v1.packageName = "qti.aisw";
      dest.v1.typeName = "ElementWiseAdd";
      break;
    case kLiteRtOpCodeTflBatchMatmul:
      dest.v1.name = QNN_OP_NAME("batch_matmul");
      dest.v1.packageName = "qti.aisw";
      dest.v1.typeName = "MatMul";
      break;
    case kLiteRtOpCodeTflConcatenation:
      dest.v1.name = QNN_OP_NAME("concatenation");
      dest.v1.packageName = "qti.aisw";
      dest.v1.typeName = "Concat";
      break;
    case kLiteRtOpCodeTflDiv:
      dest.v1.name = QNN_OP_NAME("div");
      dest.v1.packageName = "qti.aisw";
      dest.v1.typeName = "ElementWiseDivide";
      break;
    case kLiteRtOpCodeTflFullyConnected:
      dest.v1.name = QNN_OP_NAME("fully_connected");
      dest.v1.packageName = "qti.aisw";
      dest.v1.typeName = "FullyConnected";
      break;
    case kLiteRtOpCodeTflReshape:
      dest.v1.name = QNN_OP_NAME("reshape");
      dest.v1.packageName = "qti.aisw";
      dest.v1.typeName = "Reshape";
      break;
    case kLiteRtOpCodeTflRsqrt:
      dest.v1.name = QNN_OP_NAME("rsqrt");
      dest.v1.packageName = "qti.aisw";
      dest.v1.typeName = "ElementWiseRsqrt";
      break;
    case kLiteRtOpCodeTflSelectV2:
      dest.v1.name = QNN_OP_NAME("select_v2");
      dest.v1.packageName = "qti.aisw";
      dest.v1.typeName = "ElementWiseSelect";
      break;
    case kLiteRtOpCodeTflSelect:
      dest.v1.name = QNN_OP_NAME("select");
      dest.v1.packageName = "qti.aisw";
      dest.v1.typeName = "ElementWiseSelect";
      break;
    case kLiteRtOpCodeTflStridedSlice:
      dest.v1.name = QNN_OP_NAME("strided_slice");
      dest.v1.packageName = "qti.aisw";
      dest.v1.typeName = "StridedSlice";
      break;
    case kLiteRtOpCodeTflSlice:
      dest.v1.name = QNN_OP_NAME("slice");
      dest.v1.packageName = "qti.aisw";
      dest.v1.typeName = "StridedSlice";
      break;
    case kLiteRtOpCodeTflSoftmax:
      dest.v1.name = QNN_OP_NAME("softmax");
      dest.v1.packageName = "qti.aisw";
      dest.v1.typeName = "Softmax";
      break;
    case kLiteRtOpCodeTflSub:
      dest.v1.name = QNN_OP_NAME("sub");
      dest.v1.packageName = "qti.aisw";
      dest.v1.typeName = "ElementWiseSubtract";
      break;
    case kLiteRtOpCodeTflTanh:
      dest.v1.name = QNN_OP_NAME("tanh");
      dest.v1.packageName = "qti.aisw";
      dest.v1.typeName = "Tanh";
      break;
    case kLiteRtOpCodeTflTranspose:
      dest.v1.name = QNN_OP_NAME("transpose");
      dest.v1.packageName = "qti.aisw";
      dest.v1.typeName = "Transpose";
      break;
    default:
      return kLiteRtStatusErrorUnsupported;
  }
  return kLiteRtStatusOk;
}

}  // namespace

Qnn_OpConfig_t BuildDefaultOp() {
  Qnn_OpConfig_t op = QNN_OPCONFIG_INIT;
  ResetOp(op);
  return op;
}
Qnn_Param_t BuildDefaultParam() {
  Qnn_Param_t param = QNN_PARAM_INIT;
  ResetParam(param);
  return param;
}

void ResetOp(Qnn_OpConfig_t& op) {
  op = QNN_OPCONFIG_INIT;
  op.version = QNN_OPCONFIG_VERSION_1;
  op.v1 = QNN_OPCONFIG_V1_INIT;
}

void ResetParam(Qnn_Param_t& param) { param = QNN_PARAM_INIT; }
LiteRtStatus LegalizeOp(LiteRtOp src, Qnn_OpConfig_t& dest) {
  ResetOp(dest);
  Op op(src);
  return LegalizeOpType(op, dest);
}

}  // namespace litert::qnn
