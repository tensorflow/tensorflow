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
include "stablehlo/dialect/StablehloOps.td"

// List of allowed MHLO ops. All StableHLO ops are allowed by default.
defvar HloConversionAllowedOps = [
  // MHLO ops
  // go/keep-sorted start
  MHLO_AcosOp,
  MHLO_AcoshOp,
  MHLO_AddDependencyOp,
  MHLO_AllToAllOp,
  MHLO_AsinOp,
  MHLO_AsinhOp,
  MHLO_AsyncDoneOp,
  MHLO_AsyncStartOp,
  MHLO_AsyncUpdateOp,
  MHLO_AtanhOp,
  MHLO_BitcastOp,
  MHLO_CopyOp,
  MHLO_CoshOp,
  MHLO_CustomCallOp,
  MHLO_DomainOp,
  MHLO_ErfOp,
  MHLO_FusionOp,
  MHLO_MinimumBroadcastShapesOp,
  MHLO_RaggedDotOp,
  MHLO_ScanOp,
  MHLO_SinhOp,
  MHLO_StochasticConvertOp,
  MHLO_TopKOp,
  MHLO_TraceOp,
  MHLO_XlaRngGetAndUpdateStateOp,
  // go/keep-sorted end
];

// List of StableHLO and MHLO ops that need custom HLO converters.
defvar CustomHloConverterOps = [
  // StableHLO ops
  // go/keep-sorted start
  StableHLO_AllGatherOp,
  StableHLO_AllReduceOp,
  StableHLO_AllToAllOp,
  StableHLO_BatchNormGradOp,
  StableHLO_BatchNormTrainingOp,
  StableHLO_BitcastConvertOp,
  StableHLO_BroadcastInDimOp,
  StableHLO_CaseOp,
  StableHLO_CollectiveBroadcastOp,
  StableHLO_CompareOp,
  StableHLO_CompositeOp,
  StableHLO_ConstantOp,
  StableHLO_ConvertOp,
  StableHLO_ConvolutionOp,
  StableHLO_CosineOp,
  StableHLO_CustomCallOp,
  StableHLO_DotGeneralOp,
  StableHLO_DotOp,
  StableHLO_DynamicBroadcastInDimOp,
  StableHLO_DynamicConvOp,
  StableHLO_DynamicGatherOp,
  StableHLO_DynamicIotaOp,
  StableHLO_DynamicPadOp,
  StableHLO_DynamicReshapeOp,
  StableHLO_IfOp,
  StableHLO_InfeedOp,
  StableHLO_IotaOp,
  StableHLO_MapOp,
  StableHLO_OptimizationBarrierOp,
  StableHLO_OutfeedOp,
  StableHLO_PadOp,
  StableHLO_PartitionIdOp,
  StableHLO_RealDynamicSliceOp,
  StableHLO_RecvOp,
  StableHLO_ReduceOp,
  StableHLO_ReduceScatterOp,
  StableHLO_ReduceWindowOp,
  StableHLO_ReshapeOp,
  StableHLO_ReturnOp,
  StableHLO_RngBitGeneratorOp,
  StableHLO_RngOp,
  StableHLO_ScatterOp,
  StableHLO_SelectAndScatterOp,
  StableHLO_SendOp,
  StableHLO_SetDimensionSizeOp,
  StableHLO_SineOp,
  StableHLO_SortOp,
  StableHLO_SubtractOp,
  StableHLO_TanOp,
  StableHLO_UnaryEinsumOp,
  StableHLO_UniformDequantizeOp,
  StableHLO_UniformQuantizeOp,
  StableHLO_WhileOp,
  // go/keep-sorted end

  // MHLO ops
  // go/keep-sorted start
  MHLO_AcosOp,
  MHLO_AcoshOp,
  MHLO_AddDependencyOp,
  MHLO_AllToAllOp,
  MHLO_AsinOp,
  MHLO_AsinhOp,
  MHLO_AsyncDoneOp,
  MHLO_AsyncStartOp,
  MHLO_AsyncUpdateOp,
  MHLO_AtanhOp,
  MHLO_BitcastOp,
  MHLO_CopyOp,
  MHLO_CoshOp,
  MHLO_CustomCallOp,
  MHLO_DomainOp,
  MHLO_FusionOp,
  MHLO_MinimumBroadcastShapesOp,
  MHLO_RaggedDotOp,
  MHLO_ScanOp,
  MHLO_SinhOp,
  MHLO_StochasticConvertOp,
  MHLO_TopKOp,
  MHLO_TraceOp,
  MHLO_XlaRngGetAndUpdateStateOp,
  // go/keep-sorted end
];
