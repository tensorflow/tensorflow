// Copyright 2021 The TensorFlow Runtime Authors
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

#ifndef TENSORFLOW_COMPILER_MLIR_TFRT_TRANSFORMS_LMHLO_TO_GPU_PATTERN_UTILS_H_
#define TENSORFLOW_COMPILER_MLIR_TFRT_TRANSFORMS_LMHLO_TO_GPU_PATTERN_UTILS_H_

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Types.h"
#include "tensorflow/stream_executor/dnn.h"
#include "tfrt/gpu/wrapper/blas_wrapper.h"  // from @tf_runtime
#include "tfrt/gpu/wrapper/dnn_wrapper.h"  // from @tf_runtime

namespace tensorflow {

extern const tfrt::gpu::wrapper::Platform kGpuTargetPlatform;
extern const tfrt::gpu::wrapper::BlasGemmAlgo kBlasGemmDefaultAlgo;
extern const tfrt::gpu::wrapper::BlasOperation kBlasOperationTranspose;
extern const tfrt::gpu::wrapper::BlasOperation kBlasOperationConjTranspose;
extern const tfrt::gpu::wrapper::BlasOperation kBlasOperationNone;
extern const tfrt::gpu::wrapper::BlasFillMode kBlasFillModeLower;
extern const tfrt::gpu::wrapper::BlasFillMode kBlasFillModeUpper;
extern const tfrt::gpu::wrapper::BlasSideMode kBlasSideLeft;
extern const tfrt::gpu::wrapper::BlasSideMode kBlasSideRight;
extern const tfrt::gpu::wrapper::BlasDiagType kBlasDiagUnit;
extern const tfrt::gpu::wrapper::BlasDiagType kBlasDiagNonUnit;

// Converts from mlir::Type to the corresponding
// tfrt::gpu::wrapper::BlasDataType.
tfrt::gpu::wrapper::BlasDataType MlirTypeToBlasDataType(mlir::Type type);

// Converts from mlir::Type to the corresponding
// tfrt::gpu::wrapper::BlasComputeType.
tfrt::gpu::wrapper::BlasComputeType MlirTypeToBlasComputeType(mlir::Type type);

// Converts from mlir::Type to the corresponding
// tfrt::gpu::wrapper::DnnDataType.
tfrt::gpu::wrapper::DnnDataType MlirTypeToDnnDataType(mlir::Type type);
tfrt::gpu::wrapper::DnnDataType MlirTypeToDnnDataType(
    mlir::Type type, se::dnn::DataLayout data_layout);
tfrt::gpu::wrapper::DnnDataType MlirTypeToDnnDataType(
    mlir::Type type, se::dnn::FilterLayout filter_layout);

// Creates a TFRT constant op for the specified numerical value of specified
// type.
mlir::Value MakeScalingFactorConstant(mlir::OpBuilder& builder,
                                      mlir::Location loc, mlir::Type type,
                                      llvm::APFloat value_real,
                                      llvm::APFloat value_imaginary);

// Creates a TFRT constant op for the specified 32-bit pattern value of
// specified type.
mlir::Value MakeBitPatternConstant(mlir::OpBuilder& builder, mlir::Location loc,
                                   mlir::Type type, uint32_t bit_pattern);

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_TFRT_TRANSFORMS_LMHLO_TO_GPU_PATTERN_UTILS_H_
