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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/functional/function_ref.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/runtime/annotation.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "tsl/profiler/lib/scoped_annotation.h"
#include "tsl/profiler/lib/traceme.h"

namespace xla {
namespace gpu {

SequentialThunk::SequentialThunk(ThunkInfo thunk_info, ThunkSequence thunks)
    : Thunk(Kind::kSequential, thunk_info), thunks_(std::move(thunks)) {}

std::string SequentialThunk::ToString(int indent) const {
  const std::string indent_str(indent * 2, ' ');
  if (thunks_.empty()) {
    return indent_str + "No thunks.";
  }

  auto thunk_with_longest_kind = absl::c_max_element(
      thunks_,
      [](const std::unique_ptr<Thunk>& a, const std::unique_ptr<Thunk>& b) {
        return Thunk::KindToString(a->kind()).length() <
               Thunk::KindToString(b->kind()).length();
      });
  int64_t max_thunk_kind_len =
      Thunk::KindToString(thunk_with_longest_kind->get()->kind()).length();
  std::string result;
  for (const std::unique_ptr<Thunk>& thunk : thunks_) {
    absl::StrAppend(&result, indent_str);
    absl::StrAppendFormat(&result,
                          "%03d: ", thunk->thunk_info().thunk_id.value());
    // Write out the thunk kind, padded out to max_thunk_kind_len.
    absl::string_view kind_str = Thunk::KindToString(thunk->kind());
    absl::StrAppend(&result, kind_str,
                    std::string(max_thunk_kind_len - kind_str.length(), ' '),
                    "\t");
    absl::StrAppend(&result, thunk->ToString(indent + 1));
    absl::StrAppend(&result, "\n");
  }
  return result;
}

absl::Status SequentialThunk::Prepare(const PrepareParams& params) {
  for (auto& thunk : thunks_) {
    TF_RETURN_IF_ERROR(thunk->Prepare(params));
  }
  return absl::OkStatus();
}

absl::Status SequentialThunk::Initialize(const InitializeParams& params) {
  for (auto& thunk : thunks_) {
    TF_RETURN_IF_ERROR(thunk->Initialize(params));
  }
  return absl::OkStatus();
}

absl::Status SequentialThunk::ExecuteOnStream(const ExecuteParams& params) {
  std::optional<tsl::profiler::ScopedAnnotation> seq_annotation =
      GetKernelAnnotation(profile_annotation());

  for (size_t i = 0; i < thunks_.size(); ++i) {
    const std::unique_ptr<Thunk>& thunk = thunks_[i];

    tsl::profiler::TraceMe trace(thunk->profile_annotation());

    std::optional<tsl::profiler::ScopedAnnotation> annotation =
        GetKernelAnnotation(thunk->profile_annotation());

    if (params.mock_collectives && thunk->IsCollective()) {
      XLA_VLOG_DEVICE(1, params.stream->parent()->device_ordinal())
          << absl::StreamFormat(
                 "[thunk=%d/%d] Skip SequentialThunk::ExecuteOnStream: %s", i,
                 thunks_.size(), thunk->profile_annotation());
      continue;
    }

    XLA_VLOG_DEVICE(1, params.stream->parent()->device_ordinal())
        << absl::StreamFormat(
               "[thunk=%d/%d] Start SequentialThunk::ExecuteOnStream: %s", i,
               thunks_.size(), thunk->profile_annotation());

    TF_RETURN_IF_ERROR(thunk->ExecuteOnStream(params));

    XLA_VLOG_DEVICE(1, params.stream->parent()->device_ordinal())
        << absl::StreamFormat(
               "[thunk=%d/%d] End SequentialThunk::ExecuteOnStream: %s", i,
               thunks_.size(), thunk->profile_annotation());
  }
  return absl::OkStatus();
}

absl::Status SequentialThunk::WalkNested(
    absl::FunctionRef<absl::Status(Thunk*)> callback) {
  for (const std::unique_ptr<Thunk>& thunk : thunks_) {
    TF_RETURN_IF_ERROR(thunk->Walk(callback));
  }
  return absl::OkStatus();
}

absl::Status SequentialThunk::TransformAllNestedThunks(
    absl::FunctionRef<
        absl::StatusOr<std::unique_ptr<Thunk>>(std::unique_ptr<Thunk>)>
        fn) {
  for (std::unique_ptr<Thunk>& thunk : thunks_) {
    TF_RETURN_IF_ERROR(thunk->TransformAllNestedThunks(fn));
    TF_ASSIGN_OR_RETURN(thunk, fn(std::move(thunk)));
  }
  return absl::OkStatus();
}

absl::StatusOr<ThunkProto> SequentialThunk::ToProto() const {
  ThunkProto proto;
  *proto.mutable_thunk_info() = thunk_info().ToProto();

  // This sets the oneof-type to the sequential thunk, even if the thunk list is
  // empty.
  proto.mutable_sequential_thunk();
  for (const auto& thunk : thunks_) {
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

  std::vector<std::unique_ptr<Thunk>> thunks;
  thunks.push_back(std::move(thunk));
  return std::make_unique<SequentialThunk>(Thunk::ThunkInfo(),
                                           std::move(thunks));
}

}  // namespace gpu
}  // namespace xla
