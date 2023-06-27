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
#ifndef TENSORFLOW_LITE_CORE_API_FLATBUFFER_CONVERSIONS_H_
#define TENSORFLOW_LITE_CORE_API_FLATBUFFER_CONVERSIONS_H_

// These functions transform codes and data structures that are defined in the
// flatbuffer serialization format into in-memory values that are used by the
// runtime API and interpreter.

#include <cstddef>
#include <new>
#include <type_traits>

#include "tensorflow/lite/core/api/error_reporter.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {

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

  virtual ~BuiltinDataAllocator() {}
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
TfLiteStatus ParseOpData(const Operator* op, BuiltinOperator op_type,
                         ErrorReporter* error_reporter,
                         BuiltinDataAllocator* allocator, void** builtin_data);

// Converts the tensor data type used in the flat buffer to the representation
// used by the runtime.
TfLiteStatus ConvertTensorType(TensorType tensor_type, TfLiteType* type,
                               ErrorReporter* error_reporter);

TfLiteStatus ParseAbs(const Operator* op, ErrorReporter* error_reporter,
                      BuiltinDataAllocator* allocator, void** builtin_data);

TfLiteStatus ParseAdd(const Operator* op, ErrorReporter* error_reporter,
                      BuiltinDataAllocator* allocator, void** builtin_data);

TfLiteStatus ParseAddN(const Operator* op, ErrorReporter* error_reporter,
                       BuiltinDataAllocator* allocator, void** builtin_data);

TfLiteStatus ParseArgMax(const Operator* op, ErrorReporter* error_reporter,
                         BuiltinDataAllocator* allocator, void** builtin_data);

TfLiteStatus ParseArgMin(const Operator* op, ErrorReporter* error_reporter,
                         BuiltinDataAllocator* allocator, void** builtin_data);

TfLiteStatus ParseAssignVariable(const Operator* op,
                                 ErrorReporter* error_reporter,
                                 BuiltinDataAllocator* allocator,
                                 void** builtin_data);

TfLiteStatus ParseBatchMatMul(const Operator* op, ErrorReporter* error_reporter,
                              BuiltinDataAllocator* allocator,
                              void** builtin_data);

TfLiteStatus ParseBatchToSpaceNd(const Operator* op,
                                 ErrorReporter* error_reporter,
                                 BuiltinDataAllocator* allocator,
                                 void** builtin_data);

TfLiteStatus ParseBroadcastArgs(const Operator* op,
                                ErrorReporter* error_reporter,
                                BuiltinDataAllocator* allocator,
                                void** builtin_data);

TfLiteStatus ParseBroadcastTo(const Operator* op, ErrorReporter* error_reporter,
                              BuiltinDataAllocator* allocator,
                              void** builtin_data);

TfLiteStatus ParseCallOnce(const Operator* op, ErrorReporter* error_reporter,
                           BuiltinDataAllocator* allocator,
                           void** builtin_data);

TfLiteStatus ParseCeil(const Operator* op, ErrorReporter* error_reporter,
                       BuiltinDataAllocator* allocator, void** builtin_data);

TfLiteStatus ParseCast(const Operator* op, ErrorReporter* error_reporter,
                       BuiltinDataAllocator* allocator, void** builtin_data);

TfLiteStatus ParseConcatenation(const Operator* op,
                                ErrorReporter* error_reporter,
                                BuiltinDataAllocator* allocator,
                                void** builtin_data);

TfLiteStatus ParseConv2D(const Operator* op, ErrorReporter* error_reporter,
                         BuiltinDataAllocator* allocator, void** builtin_data);

TfLiteStatus ParseCos(const Operator* op, ErrorReporter* error_reporter,
                      BuiltinDataAllocator* allocator, void** builtin_data);

TfLiteStatus ParseCumsum(const Operator* op, ErrorReporter* error_reporter,
                         BuiltinDataAllocator* allocator, void** builtin_data);

TfLiteStatus ParseDepthToSpace(const Operator* op,
                               ErrorReporter* error_reporter,
                               BuiltinDataAllocator* allocator,
                               void** builtin_data);

TfLiteStatus ParseDepthwiseConv2D(const Operator* op,
                                  ErrorReporter* error_reporter,
                                  BuiltinDataAllocator* allocator,
                                  void** builtin_data);

TfLiteStatus ParseDequantize(const Operator* op, ErrorReporter* error_reporter,
                             BuiltinDataAllocator* allocator,
                             void** builtin_data);

TfLiteStatus ParseDiv(const Operator* op, ErrorReporter* error_reporter,
                      BuiltinDataAllocator* allocator, void** builtin_data);

TfLiteStatus ParseElu(const Operator* op, ErrorReporter* error_reporter,
                      BuiltinDataAllocator* allocator, void** builtin_data);

TfLiteStatus ParseEmbeddingLookup(const Operator* op,
                                  ErrorReporter* error_reporter,
                                  BuiltinDataAllocator* allocator,
                                  void** builtin_data);

TfLiteStatus ParseEqual(const Operator* op, ErrorReporter* error_reporter,
                        BuiltinDataAllocator* allocator, void** builtin_data);

TfLiteStatus ParseExp(const Operator* op, ErrorReporter* error_reporter,
                      BuiltinDataAllocator* allocator, void** builtin_data);

TfLiteStatus ParseExpandDims(const Operator* op, ErrorReporter* error_reporter,
                             BuiltinDataAllocator* allocator,
                             void** builtin_data);

TfLiteStatus ParseFill(const Operator* op, ErrorReporter* error_reporter,
                       BuiltinDataAllocator* allocator, void** builtin_data);

TfLiteStatus ParseFloor(const Operator* op, ErrorReporter* error_reporter,
                        BuiltinDataAllocator* allocator, void** builtin_data);

TfLiteStatus ParseFloorDiv(const Operator* op, ErrorReporter* error_reporter,
                           BuiltinDataAllocator* allocator,
                           void** builtin_data);

TfLiteStatus ParseFloorMod(const Operator* op, ErrorReporter* error_reporter,
                           BuiltinDataAllocator* allocator,
                           void** builtin_data);

TfLiteStatus ParseFullyConnected(const Operator* op,
                                 ErrorReporter* error_reporter,
                                 BuiltinDataAllocator* allocator,
                                 void** builtin_data);

TfLiteStatus ParseGather(const Operator* op, ErrorReporter* error_reporter,
                         BuiltinDataAllocator* allocator, void** builtin_data);

TfLiteStatus ParseGatherNd(const Operator* op, ErrorReporter* error_reporter,
                           BuiltinDataAllocator* allocator,
                           void** builtin_data);

TfLiteStatus ParseGreater(const Operator* op, ErrorReporter* error_reporter,
                          BuiltinDataAllocator* allocator, void** builtin_data);

TfLiteStatus ParseGreaterEqual(const Operator* op,
                               ErrorReporter* error_reporter,
                               BuiltinDataAllocator* allocator,
                               void** builtin_data);

TfLiteStatus ParseHardSwish(const Operator* op, ErrorReporter* error_reporter,
                            BuiltinDataAllocator* allocator,
                            void** builtin_data);

TfLiteStatus ParseIf(const Operator* op, ErrorReporter* error_reporter,
                     BuiltinDataAllocator* allocator, void** builtin_data);

TfLiteStatus ParseL2Normalization(const Operator* op,
                                  ErrorReporter* error_reporter,
                                  BuiltinDataAllocator* allocator,
                                  void** builtin_data);

TfLiteStatus ParseLeakyRelu(const Operator* op, ErrorReporter* error_reporter,
                            BuiltinDataAllocator* allocator,
                            void** builtin_data);

TfLiteStatus ParseLess(const Operator* op, ErrorReporter* error_reporter,
                       BuiltinDataAllocator* allocator, void** builtin_data);

TfLiteStatus ParseLessEqual(const Operator* op, ErrorReporter* error_reporter,
                            BuiltinDataAllocator* allocator,
                            void** builtin_data);

TfLiteStatus ParseLog(const Operator* op, ErrorReporter* error_reporter,
                      BuiltinDataAllocator* allocator, void** builtin_data);

TfLiteStatus ParseLogicalAnd(const Operator* op, ErrorReporter* error_reporter,
                             BuiltinDataAllocator* allocator,
                             void** builtin_data);

TfLiteStatus ParseLogicalNot(const Operator* op, ErrorReporter* error_reporter,
                             BuiltinDataAllocator* allocator,
                             void** builtin_data);

TfLiteStatus ParseLogicalOr(const Operator* op, ErrorReporter* error_reporter,
                            BuiltinDataAllocator* allocator,
                            void** builtin_data);

TfLiteStatus ParseLogistic(const Operator* op, ErrorReporter* error_reporter,
                           BuiltinDataAllocator* allocator,
                           void** builtin_data);

TfLiteStatus ParseLogSoftmax(const Operator* op, ErrorReporter* error_reporter,
                             BuiltinDataAllocator* allocator,
                             void** builtin_data);

TfLiteStatus ParseLSTM(const Operator* op, ErrorReporter* error_reporter,
                       BuiltinDataAllocator* allocator, void** builtin_data);

TfLiteStatus ParseMaximum(const Operator* op, ErrorReporter* error_reporter,
                          BuiltinDataAllocator* allocator, void** builtin_data);

TfLiteStatus ParseMinimum(const Operator* op, ErrorReporter* error_reporter,
                          BuiltinDataAllocator* allocator, void** builtin_data);

TfLiteStatus ParseMirrorPad(const Operator* op, ErrorReporter* error_reporter,
                            BuiltinDataAllocator* allocator,
                            void** builtin_data);

TfLiteStatus ParseMul(const Operator* op, ErrorReporter* error_reporter,
                      BuiltinDataAllocator* allocator, void** builtin_data);

TfLiteStatus ParseNeg(const Operator* op, ErrorReporter* error_reporter,
                      BuiltinDataAllocator* allocator, void** builtin_data);

TfLiteStatus ParseNotEqual(const Operator* op, ErrorReporter* error_reporter,
                           BuiltinDataAllocator* allocator,
                           void** builtin_data);

TfLiteStatus ParsePack(const Operator* op, ErrorReporter* error_reporter,
                       BuiltinDataAllocator* allocator, void** builtin_data);

TfLiteStatus ParsePad(const Operator* op, ErrorReporter* error_reporter,
                      BuiltinDataAllocator* allocator, void** builtin_data);

TfLiteStatus ParsePadV2(const Operator* op, ErrorReporter* error_reporter,
                        BuiltinDataAllocator* allocator, void** builtin_data);

TfLiteStatus ParsePool(const Operator* op, ErrorReporter* error_reporter,
                       BuiltinDataAllocator* allocator, void** builtin_data);

TfLiteStatus ParsePow(const Operator* op, ErrorReporter* error_reporter,
                      BuiltinDataAllocator* allocator, void** builtin_data);

TfLiteStatus ParsePrelu(const Operator* op, ErrorReporter* error_reporter,
                        BuiltinDataAllocator* allocator, void** builtin_data);

TfLiteStatus ParseQuantize(const Operator* op, ErrorReporter* error_reporter,
                           BuiltinDataAllocator* allocator,
                           void** builtin_data);

TfLiteStatus ParseReadVariable(const Operator* op,
                               ErrorReporter* error_reporter,
                               BuiltinDataAllocator* allocator,
                               void** builtin_data);

TfLiteStatus ParseReducer(const Operator* op, ErrorReporter* error_reporter,
                          BuiltinDataAllocator* allocator, void** builtin_data);

TfLiteStatus ParseRelu(const Operator* op, ErrorReporter* error_reporter,
                       BuiltinDataAllocator* allocator, void** builtin_data);

TfLiteStatus ParseRelu6(const Operator* op, ErrorReporter* error_reporter,
                        BuiltinDataAllocator* allocator, void** builtin_data);

TfLiteStatus ParseReshape(const Operator* op, ErrorReporter* error_reporter,
                          BuiltinDataAllocator* allocator, void** builtin_data);

TfLiteStatus ParseResizeBilinear(const Operator* op,
                                 ErrorReporter* error_reporter,
                                 BuiltinDataAllocator* allocator,
                                 void** builtin_data);

TfLiteStatus ParseResizeNearestNeighbor(const Operator* op,
                                        ErrorReporter* error_reporter,
                                        BuiltinDataAllocator* allocator,
                                        void** builtin_data);

TfLiteStatus ParseRound(const Operator* op, ErrorReporter* error_reporter,
                        BuiltinDataAllocator* allocator, void** builtin_data);

TfLiteStatus ParseRsqrt(const Operator* op, ErrorReporter* error_reporter,
                        BuiltinDataAllocator* allocator, void** builtin_data);

TfLiteStatus ParseSelectV2(const Operator* op, ErrorReporter* error_reporter,
                           BuiltinDataAllocator* allocator,
                           void** builtin_data);

TfLiteStatus ParseShape(const Operator* op, ErrorReporter* error_reporter,
                        BuiltinDataAllocator* allocator, void** builtin_data);

TfLiteStatus ParseSin(const Operator* op, ErrorReporter* error_reporter,
                      BuiltinDataAllocator* allocator, void** builtin_data);

TfLiteStatus ParseSlice(const Operator* op, ErrorReporter* error_reporter,
                        BuiltinDataAllocator* allocator, void** builtin_data);

TfLiteStatus ParseSoftmax(const Operator* op, ErrorReporter* error_reporter,
                          BuiltinDataAllocator* allocator, void** builtin_data);

TfLiteStatus ParseSpaceToBatchNd(const Operator* op,
                                 ErrorReporter* error_reporter,
                                 BuiltinDataAllocator* allocator,
                                 void** builtin_data);

TfLiteStatus ParseSpaceToDepth(const Operator* op,
                               ErrorReporter* error_reporter,
                               BuiltinDataAllocator* allocator,
                               void** builtin_data);

TfLiteStatus ParseSplit(const Operator* op, ErrorReporter* error_reporter,
                        BuiltinDataAllocator* allocator, void** builtin_data);

TfLiteStatus ParseSplitV(const Operator* op, ErrorReporter* error_reporter,
                         BuiltinDataAllocator* allocator, void** builtin_data);

TfLiteStatus ParseSqueeze(const Operator* op, ErrorReporter* error_reporter,
                          BuiltinDataAllocator* allocator, void** builtin_data);

TfLiteStatus ParseSqrt(const Operator* op, ErrorReporter* error_reporter,
                       BuiltinDataAllocator* allocator, void** builtin_data);

TfLiteStatus ParseSquare(const Operator* op, ErrorReporter* error_reporter,
                         BuiltinDataAllocator* allocator, void** builtin_data);

TfLiteStatus ParseSquaredDifference(const Operator* op,
                                    ErrorReporter* error_reporter,
                                    BuiltinDataAllocator* allocator,
                                    void** builtin_data);

TfLiteStatus ParseStridedSlice(const Operator* op,
                               ErrorReporter* error_reporter,
                               BuiltinDataAllocator* allocator,
                               void** builtin_data);

TfLiteStatus ParseSub(const Operator* op, ErrorReporter* error_reporter,
                      BuiltinDataAllocator* allocator, void** builtin_data);

TfLiteStatus ParseSvdf(const Operator* op, ErrorReporter* error_reporter,
                       BuiltinDataAllocator* allocator, void** builtin_data);

TfLiteStatus ParseTanh(const Operator* op, ErrorReporter* error_reporter,
                       BuiltinDataAllocator* allocator, void** builtin_data);

TfLiteStatus ParseTranspose(const Operator* op, ErrorReporter* error_reporter,
                            BuiltinDataAllocator* allocator,
                            void** builtin_data);

TfLiteStatus ParseTransposeConv(const Operator* op,
                                ErrorReporter* error_reporter,
                                BuiltinDataAllocator* allocator,
                                void** builtin_data);

TfLiteStatus ParseUnpack(const Operator* op, ErrorReporter* error_reporter,
                         BuiltinDataAllocator* allocator, void** builtin_data);

TfLiteStatus ParseUnidirectionalSequenceLSTM(const Operator* op,
                                             ErrorReporter* error_reporter,
                                             BuiltinDataAllocator* allocator,
                                             void** builtin_data);

TfLiteStatus ParseVarHandle(const Operator* op, ErrorReporter* error_reporter,
                            BuiltinDataAllocator* allocator,
                            void** builtin_data);

TfLiteStatus ParseWhile(const Operator* op, ErrorReporter* error_reporter,
                        BuiltinDataAllocator* allocator, void** builtin_data);

TfLiteStatus ParseZerosLike(const Operator* op, ErrorReporter* error_reporter,
                            BuiltinDataAllocator* allocator,
                            void** builtin_data);

TfLiteStatus ParseBitwiseXor(const Operator* op, ErrorReporter* error_reporter,
                             BuiltinDataAllocator* allocator,
                             void** builtin_data);

TfLiteStatus ParseRightShift(const Operator* op, ErrorReporter* error_reporter,
                             BuiltinDataAllocator* allocator,
                             void** builtin_data);

}  // namespace tflite

#endif  // TENSORFLOW_LITE_CORE_API_FLATBUFFER_CONVERSIONS_H_
