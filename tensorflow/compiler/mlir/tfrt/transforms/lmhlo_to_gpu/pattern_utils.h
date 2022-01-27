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

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Types.h"
#include "tensorflow/stream_executor/dnn.h"
#include "tfrt/gpu/wrapper/cublas_wrapper.h"  // from @tf_runtime
#include "tfrt/gpu/wrapper/cudnn_wrapper.h"  // from @tf_runtime

namespace tensorflow {

// TODO(hanbinyoon): Consider making this return a
// tfrt::gpu::wrapper::BlasDataType (also rename to MlirTypeToBlasDataType).
// Converts from mlir::Type to the corresponding cudaDataType_t.
cudaDataType_t MlirTypeToCudaDataType(mlir::Type type);

// TODO(hanbinyoon): Consider making this return a
// tfrt::gpu::wrapper::BlasComputeType (also rename to
// MlirTypeToBlasComputeType).
// Converts from mlir::Type to the corresponding cublasComputeType_t.
cublasComputeType_t MlirTypeToCublasComputeType(mlir::Type type);

// TODO(hanbinyoon): Consider making this return a
// tfrt::gpu::wrapper::DnnDataType (also rename to MlirTypeToDnnDataType).
// Converts from mlir::Type to the corresponding cudnnDataType_t.
cudnnDataType_t MlirTypeToCudnnDataType(mlir::Type type);
cudnnDataType_t MlirTypeToCudnnDataType(mlir::Type type,
                                        se::dnn::DataLayout data_layout);
cudnnDataType_t MlirTypeToCudnnDataType(mlir::Type type,
                                        se::dnn::FilterLayout filter_layout);

// Creates a TFRT constant op for the specified value of specified type.
mlir::Value MakeScalingFactorConstant(mlir::OpBuilder& builder,
                                      mlir::Location loc, mlir::Type type,
                                      llvm::APFloat value_real,
                                      llvm::APFloat value_imaginary);

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_TFRT_TRANSFORMS_LMHLO_TO_GPU_PATTERN_UTILS_H_
