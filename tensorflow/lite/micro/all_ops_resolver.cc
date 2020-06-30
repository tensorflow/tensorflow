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

#include "tensorflow/lite/micro/all_ops_resolver.h"

#include "tensorflow/lite/micro/kernels/micro_ops.h"

namespace tflite {
namespace ops {
namespace micro {
namespace custom {
TfLiteRegistration* Register_ETHOSU();
const char* GetString_ETHOSU();
}  // namespace custom
}  // namespace micro
}  // namespace ops

AllOpsResolver::AllOpsResolver() {
  // Please keep this list of Builtin Operators in alphabetical order.
  AddAbs();
  AddAdd();
  AddArgMax();
  AddArgMin();
  AddAveragePool2D();
  AddCeil();
  AddConcatenation();
  AddConv2D();
  AddCos();
  AddDepthwiseConv2D();
  AddDequantize();
  AddEqual();
  AddFloor();
  AddFullyConnected();
  AddGreater();
  AddGreaterEqual();
  AddHardSwish();
  AddL2Normalization();
  AddLess();
  AddLessEqual();
  AddLog();
  AddLogicalAnd();
  AddLogicalNot();
  AddLogicalOr();
  AddLogistic();
  AddMaximum();
  AddMaxPool2D();
  AddMean();
  AddMinimum();
  AddMul();
  AddNeg();
  AddNotEqual();
  AddPack();
  AddPad();
  AddPadV2();
  AddPrelu();
  AddQuantize();
  AddRelu();
  AddRelu6();
  AddReshape();
  AddResizeNearestNeighbor();
  AddRound();
  AddRsqrt();
  AddSin();
  AddSoftmax();
  AddSplit();
  AddSqrt();
  AddSquare();
  AddStridedSlice();
  AddSub();
  AddSvdf();
  AddTanh();
  AddUnpack();

  // TODO(b/159644355): Figure out if custom Ops belong in AllOpsResolver.
  TfLiteRegistration* registration =
      tflite::ops::micro::custom::Register_ETHOSU();
  if (registration) {
    AddCustom(tflite::ops::micro::custom::GetString_ETHOSU(), registration);
  }
}

}  // namespace tflite
