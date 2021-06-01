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

AllOpsResolver::AllOpsResolver() {
  // Please keep this list of Builtin Operators in alphabetical order.
  AddAbs();
  AddAdd();
  AddAddN();
  AddArgMax();
  AddArgMin();
  AddAveragePool2D();
  AddBatchToSpaceNd();
  AddCeil();
  AddConcatenation();
  AddConv2D();
  AddCos();
  AddCumSum();
  AddDepthToSpace();
  AddDepthwiseConv2D();
  AddDequantize();
  AddDetectionPostprocess();
  AddElu();
  AddEqual();
  AddEthosU();
  AddFloor();
  AddFloorDiv();
  AddFloorMod();
  AddFullyConnected();
  AddGreater();
  AddGreaterEqual();
  AddHardSwish();
  AddL2Normalization();
  AddL2Pool2D();
  AddLeakyRelu();
  AddLess();
  AddLessEqual();
  AddLog();
  AddLogicalAnd();
  AddLogicalNot();
  AddLogicalOr();
  AddLogistic();
  AddMaxPool2D();
  AddMaximum();
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
  AddReduceMax();
  AddRelu();
  AddRelu6();
  AddReshape();
  AddResizeBilinear();
  AddResizeNearestNeighbor();
  AddRound();
  AddRsqrt();
  AddShape();
  AddSin();
  AddSoftmax();
  AddSpaceToBatchNd();
  AddSplit();
  AddSplitV();
  AddSqrt();
  AddSquare();
  AddSqueeze();
  AddStridedSlice();
  AddSub();
  AddSvdf();
  AddTanh();
  AddTransposeConv();
  AddTranspose();
  AddUnpack();
}

}  // namespace tflite
