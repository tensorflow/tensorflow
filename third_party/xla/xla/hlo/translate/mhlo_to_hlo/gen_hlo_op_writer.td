/* Copyright 2025 The OpenXLA Authors.

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

include "xla/mlir_hlo/mhlo/IR/hlo_ops.td"

// List of StableHLO and MHLO ops that need a custom HLO converter.
defvar CustomHloConverterOps = [
  // MHLO ops.
  // go/keep-sorted start
  MHLO_AddDependencyOp,
  MHLO_AllGatherOp,
  MHLO_AllReduceOp,
  MHLO_AllToAllOp,
  MHLO_AsyncDoneOp,
  MHLO_AsyncStartOp,
  MHLO_AsyncUpdateOp,
  MHLO_BatchNormGradOp,
  MHLO_BatchNormTrainingOp,
  MHLO_BitcastConvertOp,
  MHLO_BitcastOp,
  MHLO_BroadcastInDimOp,
  MHLO_CaseOp,
  MHLO_CollectiveBroadcastOp,
  MHLO_CompareOp,
  MHLO_CompositeOp,
  MHLO_ConstantOp,
  MHLO_ConvertOp,
  MHLO_ConvolutionOp,
  MHLO_CopyOp,
  MHLO_CosineOp,
  MHLO_CustomCallOp,
  MHLO_DomainOp,
  MHLO_DotGeneralOp,
  MHLO_DotOp,
  MHLO_DynamicBroadcastInDimOp,
  MHLO_DynamicConvOp,
  MHLO_DynamicGatherOp,
  MHLO_DynamicIotaOp,
  MHLO_DynamicPadOp,
  MHLO_DynamicReshapeOp,
  MHLO_FusionOp,
  MHLO_IfOp,
  MHLO_InfeedOp,
  MHLO_IotaOp,
  MHLO_MapOp,
  MHLO_MinimumBroadcastShapesOp,
  MHLO_OptimizationBarrierOp,
  MHLO_OutfeedOp,
  MHLO_PadOp,
  MHLO_PartitionIdOp,
  MHLO_RaggedDotOp,
  MHLO_RealDynamicSliceOp,
  MHLO_RecvOp,
  MHLO_ReduceOp,
  MHLO_ReduceScatterOp,
  MHLO_ReduceWindowOp,
  MHLO_ReshapeOp,
  MHLO_ReturnOp,
  MHLO_RngBitGeneratorOp,
  MHLO_RngOp,
  MHLO_ScatterOp,
  MHLO_SelectAndScatterOp,
  MHLO_SendOp,
  MHLO_SetDimensionSizeOp,
  MHLO_SineOp,
  MHLO_SortOp,
  MHLO_SparseDotOp,
  MHLO_StochasticConvertOp,
  MHLO_SubtractOp,
  MHLO_TanOp,
  MHLO_TopKOp,
  MHLO_TraceOp,
  MHLO_UniformDequantizeOp,
  MHLO_UniformQuantizeOp,
  MHLO_WhileOp,
  MHLO_XlaRngGetAndUpdateStateOp,
  // go/keep-sorted end
];
