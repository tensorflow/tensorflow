/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#ifdef GET_SUPPORTED_OP_LIST
#undef GET_SUPPORTED_OP_LIST

// clang-format off

// Unary Elementwise Ops
lmhlo::AbsOp, lmhlo::CeilOp, lmhlo::FloorOp, lmhlo::ConvertOp, lmhlo::CosOp,
lmhlo::ExpOp, lmhlo::LogOp, lmhlo::NegOp, lmhlo::RsqrtOp, lmhlo::SqrtOp,
lmhlo::SignOp, lmhlo::TanhOp,

// Binary Elementwise Ops
lmhlo::AddOp, lmhlo::DivOp, lmhlo::MaxOp, lmhlo::MinOp, lmhlo::MulOp,
lmhlo::RemOp, lmhlo::SubOp, lmhlo::AndOp, lmhlo::OrOp, lmhlo::CompareOp,
lmhlo::PowOp,

// Ternary Elementwise Ops
lmhlo::SelectOp,

// Other Ops
lmhlo::ConstOp,
lmhlo::RealDynamicSliceOp,
lmhlo::DynamicBroadcastInDimOp,
lmhlo::BroadcastInDimOp

// clang-format on

#endif
