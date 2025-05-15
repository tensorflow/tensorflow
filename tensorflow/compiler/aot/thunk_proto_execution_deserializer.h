/* Copyright 2025 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_AOT_THUNK_PROTO_EXECUTION_DESERIALIZER_H_
#define TENSORFLOW_COMPILER_AOT_THUNK_PROTO_EXECUTION_DESERIALIZER_H_

#include <cstdint>
#include <string>

#include "absl/status/statusor.h"
#include "xla/backends/cpu/runtime/convolution_lib.h"
#include "xla/backends/cpu/runtime/thunk.pb.h"
#include "xla/service/cpu/executable.pb.h"
#include "xla/xla_data.pb.h"

namespace tensorflow {
namespace tfcompile {

// Helper class for deserializing the contents of specific thunks into C++ code
// that is used to codegen the `Run` method of the tfcompiled models.
class ThunkProtoExecutionDeserializer {
 public:
  absl::StatusOr<std::string> GetThunkSpecificRunImpl(
      const xla::cpu::CompilationResultProto& proto) &&;

  absl::StatusOr<std::string> ThunkSpecificRunImplFromThunkSequence(
      const xla::cpu::ThunkSequenceProto& thunk_sequence_proto);

 protected:
  absl::StatusOr<std::string> GetMatmulFunction(xla::PrimitiveType xla_type,
                                                bool is_single_threaded);

  absl::StatusOr<std::string> GetDotThunkRunImpl(
      const xla::cpu::ThunkProto& thunk);

  absl::StatusOr<std::string> GetConvolutionFunction(
      xla::PrimitiveType xla_type, bool is_single_threaded);

  absl::StatusOr<std::string> GetConvolution2DRunImpl(
      const xla::cpu::ConvolutionThunkProto& convolution_thunk,
      const xla::cpu::ConvolutionCanonicalDims& canonical_dims);

  absl::StatusOr<std::string> GetConvolutionFusionThunkRunImpl(
      const xla::cpu::ThunkProto& thunk);

  absl::StatusOr<std::string> GetRngGetAndUpdateStateThunkRunImpl(
      const xla::cpu::ThunkProto& thunk);

  absl::StatusOr<std::string> GetCallThunkRunImpl(
      const xla::cpu::ThunkProto& thunk);

  absl::StatusOr<std::string> GetKernelThunkRunImpl(
      const xla::cpu::ThunkProto& thunk);

  absl::StatusOr<std::string> GetCopyThunkRunImpl(
      const xla::cpu::ThunkProto& thunk);

  absl::StatusOr<std::string> GetConditionalThunkRunImpl(
      const xla::cpu::ThunkProto& thunk);

  absl::StatusOr<std::string> GetForLoopThunkRunImpl(
      const xla::cpu::WhileThunkProto& while_thunk);

  absl::StatusOr<std::string> GetWhileThunkRunImpl(
      const xla::cpu::ThunkProto& thunk);

  absl::StatusOr<std::string> CppDataTypeFromXlaType(
      xla::PrimitiveType xla_type);

 private:
  // The index of the next rng state to use when deserializing the rng state
  // from the ThunkProto.
  int64_t rng_state_index_ = 0;
};

}  // namespace tfcompile
}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_AOT_THUNK_PROTO_EXECUTION_DESERIALIZER_H_
