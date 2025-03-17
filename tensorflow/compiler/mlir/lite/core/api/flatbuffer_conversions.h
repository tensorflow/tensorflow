/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_COMPILER_MLIR_LITE_CORE_API_FLATBUFFER_CONVERSIONS_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_CORE_API_FLATBUFFER_CONVERSIONS_H_

#include <cstddef>
#include <type_traits>

#include "absl/status/status.h"
#include "tensorflow/compiler/mlir/lite/core/c/tflite_types.h"
#include "tensorflow/compiler/mlir/lite/schema/schema_generated.h"

// The namespace tflite_file is for the data structures that define the .tflite
// file format, and code that is tightly coupled with those data structures.
// The .tflite file format is the serialized flatbuffer representation of
// computations on tensors that TF Lite uses for distribution of compiled ML
// models.
namespace tflite_file {

// This namespace contains functions that transform code and data structures
// that are defined in the flatbuffer serialization format into
// in-memory values that are used by the runtime API, interpreter and compiler.
namespace flatbuffer_conversions {

using tflite::Operator;

// Interface class for builtin data allocations.
class BuiltinDataAllocator {
 public:
  virtual void* Allocate(size_t size, size_t alignment_hint) = 0;
  virtual void Deallocate(void* data) = 0;

  // Allocate a structure, but make sure it is a POD structure that doesn't
  // require constructors to run. The reason we do this, is that Interpreter's C
  // extension part will take ownership so destructors  will not be run during
  // deallocation.
  template <typename T>
  T* AllocatePOD() {
    // TODO(b/154346074): Change this to is_trivially_destructible when all
    // platform targets support that properly.
    static_assert(std::is_pod<T>::value, "Builtin data structure must be POD.");
    void* allocated_memory = this->Allocate(sizeof(T), alignof(T));
    return new (allocated_memory) T();
  }

  virtual ~BuiltinDataAllocator() = default;
};

// Parse the appropriate data out of the op.
//
// This handles builtin data explicitly as there are flatbuffer schemas.
// If it returns kTfLiteOk, it passes the data out with `builtin_data`. The
// calling function has to pass in an allocator object, and this allocator
// will be called to reserve space for the output data. If the calling
// function's allocator reserves memory on the heap, then it's the calling
// function's responsibility to free it.
// If it returns kTfLiteError, `builtin_data` will be `nullptr`.
absl::Status ParseOpData(const tflite::Operator* op,
                         tflite::BuiltinOperator op_type,
                         BuiltinDataAllocator* allocator, void** builtin_data);

// Converts the tensor data type used in the flat buffer to the representation
// used by the runtime.
absl::Status ConvertTensorType(tflite::TensorType tensor_type,
                               TfLiteType* type);

absl::Status ParseAbs(const Operator* op, BuiltinDataAllocator* allocator,
                      void** builtin_data);

absl::Status ParseAdd(const Operator* op, BuiltinDataAllocator* allocator,
                      void** builtin_data);

absl::Status ParseAddN(const Operator* op, BuiltinDataAllocator* allocator,
                       void** builtin_data);

absl::Status ParseArgMax(const Operator* op, BuiltinDataAllocator* allocator,
                         void** builtin_data);

absl::Status ParseArgMin(const Operator* op, BuiltinDataAllocator* allocator,
                         void** builtin_data);

absl::Status ParseAssignVariable(const Operator* op,
                                 BuiltinDataAllocator* allocator,
                                 void** builtin_data);

absl::Status ParseBatchMatMul(const Operator* op,
                              BuiltinDataAllocator* allocator,
                              void** builtin_data);

absl::Status ParseBatchToSpaceNd(const Operator* op,
                                 BuiltinDataAllocator* allocator,
                                 void** builtin_data);

absl::Status ParseBroadcastArgs(const Operator* op,
                                BuiltinDataAllocator* allocator,
                                void** builtin_data);

absl::Status ParseBroadcastTo(const Operator* op,
                              BuiltinDataAllocator* allocator,
                              void** builtin_data);

absl::Status ParseCallOnce(const Operator* op, BuiltinDataAllocator* allocator,
                           void** builtin_data);

absl::Status ParseCeil(const Operator* op, BuiltinDataAllocator* allocator,
                       void** builtin_data);

absl::Status ParseCast(const Operator* op, BuiltinDataAllocator* allocator,
                       void** builtin_data);

absl::Status ParseConcatenation(const Operator* op,
                                BuiltinDataAllocator* allocator,
                                void** builtin_data);

absl::Status ParseConv2D(const Operator* op, BuiltinDataAllocator* allocator,
                         void** builtin_data);

absl::Status ParseCos(const Operator* op, BuiltinDataAllocator* allocator,
                      void** builtin_data);

absl::Status ParseCumsum(const Operator* op, BuiltinDataAllocator* allocator,
                         void** builtin_data);

absl::Status ParseDepthToSpace(const Operator* op,
                               BuiltinDataAllocator* allocator,
                               void** builtin_data);

absl::Status ParseDepthwiseConv2D(const Operator* op,
                                  BuiltinDataAllocator* allocator,
                                  void** builtin_data);

absl::Status ParseDequantize(const Operator* op,
                             BuiltinDataAllocator* allocator,
                             void** builtin_data);

absl::Status ParseDiv(const Operator* op, BuiltinDataAllocator* allocator,
                      void** builtin_data);

absl::Status ParseElu(const Operator* op, BuiltinDataAllocator* allocator,
                      void** builtin_data);

absl::Status ParseEmbeddingLookup(const Operator* op,
                                  BuiltinDataAllocator* allocator,
                                  void** builtin_data);

absl::Status ParseEqual(const Operator* op, BuiltinDataAllocator* allocator,
                        void** builtin_data);

absl::Status ParseExp(const Operator* op, BuiltinDataAllocator* allocator,
                      void** builtin_data);

absl::Status ParseExpandDims(const Operator* op,
                             BuiltinDataAllocator* allocator,
                             void** builtin_data);

absl::Status ParseFill(const Operator* op, BuiltinDataAllocator* allocator,
                       void** builtin_data);

absl::Status ParseFloor(const Operator* op, BuiltinDataAllocator* allocator,
                        void** builtin_data);

absl::Status ParseFloorDiv(const Operator* op, BuiltinDataAllocator* allocator,
                           void** builtin_data);

absl::Status ParseFloorMod(const Operator* op, BuiltinDataAllocator* allocator,
                           void** builtin_data);

absl::Status ParseFullyConnected(const Operator* op,
                                 BuiltinDataAllocator* allocator,
                                 void** builtin_data);

absl::Status ParseGather(const Operator* op, BuiltinDataAllocator* allocator,
                         void** builtin_data);

absl::Status ParseGatherNd(const Operator* op, BuiltinDataAllocator* allocator,
                           void** builtin_data);

absl::Status ParseGreater(const Operator* op, BuiltinDataAllocator* allocator,
                          void** builtin_data);

absl::Status ParseGreaterEqual(const Operator* op,
                               BuiltinDataAllocator* allocator,
                               void** builtin_data);

absl::Status ParseHardSwish(const Operator* op, BuiltinDataAllocator* allocator,
                            void** builtin_data);

absl::Status ParseIf(const Operator* op, BuiltinDataAllocator* allocator,
                     void** builtin_data);

absl::Status ParseL2Normalization(const Operator* op,
                                  BuiltinDataAllocator* allocator,
                                  void** builtin_data);

absl::Status ParseLeakyRelu(const Operator* op, BuiltinDataAllocator* allocator,
                            void** builtin_data);

absl::Status ParseLess(const Operator* op, BuiltinDataAllocator* allocator,
                       void** builtin_data);

absl::Status ParseLessEqual(const Operator* op, BuiltinDataAllocator* allocator,
                            void** builtin_data);

absl::Status ParseLog(const Operator* op, BuiltinDataAllocator* allocator,
                      void** builtin_data);

absl::Status ParseLogicalAnd(const Operator* op,
                             BuiltinDataAllocator* allocator,
                             void** builtin_data);

absl::Status ParseLogicalNot(const Operator* op,
                             BuiltinDataAllocator* allocator,
                             void** builtin_data);

absl::Status ParseLogicalOr(const Operator* op, BuiltinDataAllocator* allocator,
                            void** builtin_data);

absl::Status ParseLogistic(const Operator* op, BuiltinDataAllocator* allocator,
                           void** builtin_data);

absl::Status ParseLogSoftmax(const Operator* op,
                             BuiltinDataAllocator* allocator,
                             void** builtin_data);

absl::Status ParseLSTM(const Operator* op, BuiltinDataAllocator* allocator,
                       void** builtin_data);

absl::Status ParseMaximum(const Operator* op, BuiltinDataAllocator* allocator,
                          void** builtin_data);

absl::Status ParseMinimum(const Operator* op, BuiltinDataAllocator* allocator,
                          void** builtin_data);

absl::Status ParseMirrorPad(const Operator* op, BuiltinDataAllocator* allocator,
                            void** builtin_data);

absl::Status ParseMul(const Operator* op, BuiltinDataAllocator* allocator,
                      void** builtin_data);

absl::Status ParseNeg(const Operator* op, BuiltinDataAllocator* allocator,
                      void** builtin_data);

absl::Status ParseNotEqual(const Operator* op, BuiltinDataAllocator* allocator,
                           void** builtin_data);

absl::Status ParsePack(const Operator* op, BuiltinDataAllocator* allocator,
                       void** builtin_data);

absl::Status ParsePad(const Operator* op, BuiltinDataAllocator* allocator,
                      void** builtin_data);

absl::Status ParsePadV2(const Operator* op, BuiltinDataAllocator* allocator,
                        void** builtin_data);

absl::Status ParsePool(const Operator* op, BuiltinDataAllocator* allocator,
                       void** builtin_data);

absl::Status ParsePow(const Operator* op, BuiltinDataAllocator* allocator,
                      void** builtin_data);

absl::Status ParsePrelu(const Operator* op, BuiltinDataAllocator* allocator,
                        void** builtin_data);

absl::Status ParseQuantize(const Operator* op, BuiltinDataAllocator* allocator,
                           void** builtin_data);

absl::Status ParseReadVariable(const Operator* op,
                               BuiltinDataAllocator* allocator,
                               void** builtin_data);

absl::Status ParseReducer(const Operator* op, BuiltinDataAllocator* allocator,
                          void** builtin_data);

absl::Status ParseRelu(const Operator* op, BuiltinDataAllocator* allocator,
                       void** builtin_data);

absl::Status ParseRelu6(const Operator* op, BuiltinDataAllocator* allocator,
                        void** builtin_data);

absl::Status ParseReshape(const Operator* op, BuiltinDataAllocator* allocator,
                          void** builtin_data);

absl::Status ParseResizeBilinear(const Operator* op,
                                 BuiltinDataAllocator* allocator,
                                 void** builtin_data);

absl::Status ParseResizeNearestNeighbor(const Operator* op,
                                        BuiltinDataAllocator* allocator,
                                        void** builtin_data);

absl::Status ParseRound(const Operator* op, BuiltinDataAllocator* allocator,
                        void** builtin_data);

absl::Status ParseRsqrt(const Operator* op, BuiltinDataAllocator* allocator,
                        void** builtin_data);

absl::Status ParseSelectV2(const Operator* op, BuiltinDataAllocator* allocator,
                           void** builtin_data);

absl::Status ParseShape(const Operator* op, BuiltinDataAllocator* allocator,
                        void** builtin_data);

absl::Status ParseSin(const Operator* op, BuiltinDataAllocator* allocator,
                      void** builtin_data);

absl::Status ParseSlice(const Operator* op, BuiltinDataAllocator* allocator,
                        void** builtin_data);

absl::Status ParseSoftmax(const Operator* op, BuiltinDataAllocator* allocator,
                          void** builtin_data);

absl::Status ParseSpaceToBatchNd(const Operator* op,
                                 BuiltinDataAllocator* allocator,
                                 void** builtin_data);

absl::Status ParseSpaceToDepth(const Operator* op,
                               BuiltinDataAllocator* allocator,
                               void** builtin_data);

absl::Status ParseSplit(const Operator* op, BuiltinDataAllocator* allocator,
                        void** builtin_data);

absl::Status ParseSplitV(const Operator* op, BuiltinDataAllocator* allocator,
                         void** builtin_data);
absl::Status ParseSqueeze(const Operator* op, BuiltinDataAllocator* allocator,
                          void** builtin_data);

absl::Status ParseSqrt(const Operator* op, BuiltinDataAllocator* allocator,
                       void** builtin_data);
absl::Status ParseSquare(const Operator* op, BuiltinDataAllocator* allocator,
                         void** builtin_data);

absl::Status ParseSquaredDifference(const Operator* op,
                                    BuiltinDataAllocator* allocator,
                                    void** builtin_data);

absl::Status ParseStridedSlice(const Operator* op,
                               BuiltinDataAllocator* allocator,
                               void** builtin_data);

absl::Status ParseSub(const Operator* op, BuiltinDataAllocator* allocator,
                      void** builtin_data);

absl::Status ParseSvdf(const Operator* op, BuiltinDataAllocator* allocator,
                       void** builtin_data);

absl::Status ParseTanh(const Operator* op, BuiltinDataAllocator* allocator,
                       void** builtin_data);

absl::Status ParseTranspose(const Operator* op, BuiltinDataAllocator* allocator,
                            void** builtin_data);

absl::Status ParseTransposeConv(const Operator* op,
                                BuiltinDataAllocator* allocator,
                                void** builtin_data);

absl::Status ParseUnpack(const Operator* op, BuiltinDataAllocator* allocator,
                         void** builtin_data);

absl::Status ParseUnidirectionalSequenceLSTM(const Operator* op,
                                             BuiltinDataAllocator* allocator,
                                             void** builtin_data);

absl::Status ParseVarHandle(const Operator* op, BuiltinDataAllocator* allocator,
                            void** builtin_data);

absl::Status ParseWhile(const Operator* op, BuiltinDataAllocator* allocator,
                        void** builtin_data);

absl::Status ParseZerosLike(const Operator* op, BuiltinDataAllocator* allocator,
                            void** builtin_data);

absl::Status ParseBitwiseXor(const Operator* op,
                             BuiltinDataAllocator* allocator,
                             void** builtin_data);

absl::Status ParseRightShift(const Operator* op,
                             BuiltinDataAllocator* allocator,
                             void** builtin_data);

absl::Status ParseStablehloScatter(const Operator* op,
                                   BuiltinDataAllocator* allocator,
                                   void** builtin_data);

absl::Status ParseStablehloRngBitGenerator(const Operator* op,
                                           BuiltinDataAllocator* allocator,
                                           void** builtin_data);

absl::Status ParseStablehloGather(const Operator* op,
                                  BuiltinDataAllocator* allocator,
                                  void** builtin_data);

absl::Status ParseStablehloReduceWindow(const Operator* op,
                                        BuiltinDataAllocator* allocator,
                                        void** builtin_data);

absl::Status ParseStablehloPad(const Operator* op,
                               BuiltinDataAllocator* allocator,
                               void** builtin_data);

absl::Status ParseStablehloComposite(const Operator* op,
                                     BuiltinDataAllocator* allocator,
                                     void** builtin_data);

absl::Status ParseStablehloShiftLeft(const Operator* op,
                                     BuiltinDataAllocator* allocator,
                                     void** builtin_data);

absl::Status ParseStablehloCase(const Operator* op,
                                BuiltinDataAllocator* allocator,
                                void** builtin_data);

}  // namespace flatbuffer_conversions
}  // namespace tflite_file

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_CORE_API_FLATBUFFER_CONVERSIONS_H_
