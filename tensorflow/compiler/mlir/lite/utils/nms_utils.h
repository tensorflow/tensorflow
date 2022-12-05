/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

// This header file defines common utils used by TFLite transformation
// passes to work with NMS ops in TFLite.

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_UTILS_NMS_UTILS_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_UTILS_NMS_UTILS_H_

#include <string>

#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_attributes.h"

namespace mlir {
namespace TFL {

// Abstracts the conversion of the padded NMS composite function.
class ConvertNMSPaddedFunc {
 public:
  explicit ConvertNMSPaddedFunc(func::FuncOp func) : func_(func) {}

  void RewriteFunc();

  LogicalResult VerifySignature();

 private:
  func::FuncOp func_;
};

// Abstracts the conversion of the SSD post-processing composite function to
// TFLite.
class ConvertSSDPostProcessFunc {
 public:
  explicit ConvertSSDPostProcessFunc(func::FuncOp func, mlir::TF::FuncAttr attr)
      : func_(func), attr_(attr) {}

  LogicalResult RewriteFunc();

  LogicalResult VerifySignature();

 private:
  LogicalResult CreateNMSCustomOptions(func::FuncOp func, DictionaryAttr attrs,
                                       std::string& custom_option_buffer);

  LogicalResult AddIntAttr(func::FuncOp func, DictionaryAttr attrs,
                           const std::string& attribute,
                           flexbuffers::Builder* builder);

  LogicalResult AddFloatAttr(func::FuncOp func, DictionaryAttr attrs,
                             const std::string& attribute,
                             flexbuffers::Builder* builder);

  LogicalResult HasIntAttr(func::FuncOp func, DictionaryAttr attrs,
                           const std::string& attribute);

  LogicalResult HasFloatAttr(func::FuncOp func, DictionaryAttr attrs,
                             const std::string& attribute);

  func::FuncOp func_;
  mlir::TF::FuncAttr attr_;
};

}  // end namespace TFL
}  // end namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_UTILS_TFTEXT_UTILS_H_
