/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_CORE_LIGHT_MODEL_BUILDER_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_CORE_LIGHT_MODEL_BUILDER_H_

#include <stddef.h>

#include <map>
#include <memory>
#include <string>

#include "tensorflow/compiler/mlir/lite/schema/schema_generated.h"
#include "tensorflow/lite/allocation.h"

namespace mlir {

// A lighter version of tflite::FlatBufferModel defined in
// tensorflow/lite/core/model_builder.h.

class LightFlatBufferModel {
 public:
  /// Builds a model based on a file.
  /// Returns a nullptr in case of failure.
  static std::unique_ptr<LightFlatBufferModel> BuildFromFile(
      const char* filename);

  /// Builds a model based on a pre-loaded flatbuffer.
  /// Caller retains ownership of the buffer and should keep it alive until
  /// the returned object is destroyed. Returns a nullptr in case of failure.
  /// NOTE: this does NOT validate the buffer so it should NOT be called on
  /// invalid/untrusted input. Use VerifyAndBuildFromBuffer in that case
  static std::unique_ptr<LightFlatBufferModel> BuildFromBuffer(
      const char* caller_owned_buffer, size_t buffer_size);

  /// Verifies whether the content of the buffer is legit, then builds a model
  /// based on the pre-loaded flatbuffer. We always check with
  /// tflite::VerifyModelBuffer. The caller retains ownership of the buffer and
  /// should keep it alive until the returned object is destroyed. Returns a
  /// nullptr in case of failure.
  static std::unique_ptr<LightFlatBufferModel> VerifyAndBuildFromBuffer(
      const char* caller_owned_buffer, size_t buffer_size);

  /// Builds a model directly from an allocation.
  /// Ownership of the allocation is passed to the model. Returns a nullptr in
  /// case of failure (e.g., the allocation is invalid).
  static std::unique_ptr<LightFlatBufferModel> BuildFromAllocation(
      std::unique_ptr<tflite::Allocation> allocation);

  /// Verifies whether the content of the allocation is legit, then builds a
  /// model based on the provided allocation. We always check with
  /// tflite::VerifyModelBuffer. Ownership of the allocation is passed to the
  /// model. Returns a nullptr in case of failure.
  static std::unique_ptr<LightFlatBufferModel> VerifyAndBuildFromAllocation(
      std::unique_ptr<tflite::Allocation> allocation);

  // Releases memory or unmaps mmaped memory.
  ~LightFlatBufferModel();

  // Copying or assignment is disallowed to simplify ownership semantics.
  LightFlatBufferModel(const LightFlatBufferModel&) = delete;
  LightFlatBufferModel& operator=(const LightFlatBufferModel&) = delete;

  bool initialized() const { return model_ != nullptr; }
  const tflite::Model* operator->() const { return model_; }
  const tflite::Model* GetModel() const { return model_; }
  const tflite::Allocation* allocation() const { return allocation_.get(); }

 private:
  /// Loads a model from a given allocation. LightFlatBufferModel will take over
  /// the ownership of `allocation`, and delete it in destructor.
  explicit LightFlatBufferModel(std::unique_ptr<tflite::Allocation> allocation);

  /// Flatbuffer traverser pointer. (Model* is a pointer that is within the
  /// allocated memory of the data allocated by allocation's internals.
  const tflite::Model* model_ = nullptr;
  /// The allocator used for holding memory of the model.
  std::unique_ptr<tflite::Allocation> allocation_;
};

}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_CORE_LIGHT_MODEL_BUILDER_H_
