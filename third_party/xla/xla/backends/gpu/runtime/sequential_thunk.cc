/* Copyright 2017 The OpenXLA Authors.

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

#include "xla/backends/gpu/runtime/sequential_thunk.h"

#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/functional/function_ref.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/backends/gpu/runtime/annotation.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "tsl/profiler/lib/scoped_annotation.h"

namespace xla::gpu {

SequentialThunk::SequentialThunk(ThunkInfo thunk_info, ThunkSequence thunks)
    : Thunk(Kind::kSequential, thunk_info), executor_(std::move(thunks)) {}

std::string SequentialThunk::ToString(int indent) const {
  return executor_.thunks().ToString(indent);
}

absl::Status SequentialThunk::Prepare(const PrepareParams& params) {
  return executor_.Prepare(params);
}

absl::Status SequentialThunk::Initialize(const InitializeParams& params) {
  return executor_.Initialize(params);
}

absl::Status SequentialThunk::ExecuteOnStream(const ExecuteParams& params) {
  std::optional<tsl::profiler::ScopedAnnotation> seq_annotation =
      GetKernelAnnotation(profile_annotation());
  return executor_.ExecuteOnStream(params);
}

absl::Status SequentialThunk::WalkNested(Walker callback) {
  return executor_.thunks().WalkNested(callback);
}

absl::Status SequentialThunk::TransformNested(Transformer callback) {
  return executor_.thunks().TransformNested(callback);
}

absl::StatusOr<ThunkProto> SequentialThunk::ToProto() const {
  ThunkProto proto;
  *proto.mutable_thunk_info() = thunk_info().ToProto();

  // This sets the oneof-type to the sequential thunk, even if the thunk list is
  // empty.
  proto.mutable_sequential_thunk();
  for (const auto& thunk : executor_.thunks()) {
    TF_ASSIGN_OR_RETURN(*proto.mutable_sequential_thunk()->add_thunks(),
                        thunk->ToProto());
  }
  return proto;
}

absl::StatusOr<std::unique_ptr<SequentialThunk>> SequentialThunk::FromProto(
    ThunkInfo thunk_info, const SequentialThunkProto& thunk_proto,
    const Deserializer& deserializer) {
  ThunkSequence thunk_sequence;
  for (const auto& sub_thunk_proto : thunk_proto.thunks()) {
    TF_ASSIGN_OR_RETURN(std::unique_ptr<Thunk> sub_thunk,
                        deserializer(sub_thunk_proto));
    thunk_sequence.push_back(std::move(sub_thunk));
  }

  return std::make_unique<SequentialThunk>(std::move(thunk_info),
                                           std::move(thunk_sequence));
}

std::unique_ptr<SequentialThunk> SequentialThunk::FromThunk(
    std::unique_ptr<Thunk> thunk) {
  if (thunk->kind() == Thunk::kSequential) {
    return std::unique_ptr<SequentialThunk>(
        static_cast<SequentialThunk*>(thunk.release()));
  }

  ThunkSequence thunks;
  thunks.push_back(std::move(thunk));
  return std::make_unique<SequentialThunk>(Thunk::ThunkInfo(),
                                           std::move(thunks));
}

}  // namespace xla::gpu
