/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_TRANSFORMS_PASSES_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_TRANSFORMS_PASSES_H_

namespace mlir {
class FunctionPassBase;
class ModulePassBase;

namespace TFL {

// Creates an instance of the TensorFlow Lite dialect LegalizeTF pass.
FunctionPassBase *CreateLegalizeTFPass();

// Creates an instance of the TensorFlow Lite dialect Optimize pass.
FunctionPassBase *CreateOptimizePass();

// Creates an instance of the TensorFlow Lite dialect PrepareTF pass.
FunctionPassBase *CreatePrepareTFPass();

// Creates an instance of the TensorFlow Lite dialect LowerStaticTensorList
// pass.
ModulePassBase *CreateLowerStaticTensorListPass();

// Creates an instance of the TensorFlow Lite dialect Quantize pass.
FunctionPassBase *CreateQuantizePass();

// Creates an instance of the TensorFlow Lite dialect PrepareQuantize pass.
FunctionPassBase *CreatePrepareQuantizePass();

// Creates a instance of the TensorFlow Lite dialect PostQuantize pass.
FunctionPassBase *CreatePostQuantizePass(bool emit_quant_adaptor_ops);
}  // namespace TFL

}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_TRANSFORMS_PASSES_H_
