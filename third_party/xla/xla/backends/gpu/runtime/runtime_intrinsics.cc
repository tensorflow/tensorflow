/* Copyright 2022 The OpenXLA Authors.

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

#include "xla/backends/gpu/runtime/runtime_intrinsics.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/base/const_init.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/ascii.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "absl/synchronization/mutex.h"
#include "xla/backends/gpu/ffi.h"
#include "xla/ffi/ffi.h"
#include "xla/ffi/ffi_api.h"
#include "xla/layout_util.h"
#include "xla/literal.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/custom_call_status.h"
#include "xla/service/custom_call_target_registry.h"
#include "xla/service/platform_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/stream.h"
#include "xla/tsl/lib/io/record_writer.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/file_system.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/host_info.h"
#include "tsl/platform/path.h"

namespace xla {
namespace {

std::string GetGpuPlatformName() {
  return absl::AsciiStrToUpper(
      PlatformUtil::CanonicalPlatformName("gpu").value());
}

absl::Status AssertionCustomCall(
    se::Stream* stream, ffi::Buffer<PRED> buffer, absl::string_view error_msg,
    xla::ffi::Result<xla::ffi::Buffer<xla::TOKEN>> res) {
  if (!stream) {
    return Internal("Stream is nullptr.");
  }

  int8_t expected = false;
  int64_t byte_size = sizeof(int8_t);
  CHECK_EQ(byte_size, ShapeUtil::ByteSizeOfPrimitiveType(PrimitiveType::PRED));
  TF_RETURN_IF_ERROR(
      stream->Memcpy(&expected, buffer.device_memory(), byte_size));
  TF_RETURN_IF_ERROR(stream->BlockHostUntilDone());
  if (!static_cast<bool>(expected)) {
    return Internal("%s", error_msg);
  }

  return absl::OkStatus();
}

void NopReturnTokenCustomCall(void* stream_handle, void** buffers,
                              const char* opaque, int opaque_len,
                              XlaCustomCallStatus* status) {
  VLOG(1) << "NopReturnTokenCustomCall called.";
}

absl::StatusOr<Literal> ConvertToLiteral(se::Stream* stream,
                                         const ffi::AnyBuffer& arg) {
  Shape shape = ShapeUtil::MakeShape(arg.element_type(), arg.dimensions());
  LayoutUtil::SetToDefaultLayout(&shape);

  TF_ASSIGN_OR_RETURN(Literal literal, Literal::Make(shape));

  int64_t size_bytes = arg.size_bytes();
  TF_ASSIGN_OR_RETURN(std::unique_ptr<se::MemoryAllocation> host_buffer,
                      stream->parent()->HostMemoryAllocate(size_bytes));
  TF_RETURN_IF_ERROR(
      stream->Memcpy(literal.untyped_data(), arg.device_memory(), size_bytes));
  TF_RETURN_IF_ERROR(stream->BlockHostUntilDone());

  return literal;
}

absl::Status DebugPrintCustomCall(se::Stream* stream, ffi::RemainingArgs args,
                                  absl::string_view format,
                                  ffi::Result<ffi::Buffer<xla::TOKEN>> res) {
  if (!stream) {
    return Internal("Stream is nullptr.");
  }

  std::vector<ffi::AnyBuffer> args_buffers;
  args_buffers.reserve(args.size());
  for (int i = 0; i < args.size(); ++i) {
    absl::StatusOr<ffi::AnyBuffer> arg = args.get<ffi::AnyBuffer>(i);
    if (!arg.ok()) {
      return arg.status();
    }
    args_buffers.push_back(*arg);
  }

  std::string formatted(format);

  // Iterate in reverse order to match the longest string to substitute first.
  for (int i = args_buffers.size() - 1; i >= 0; --i) {
    std::string to_substitute = absl::StrCat("$", i);
    if (!absl::StrContains(formatted, to_substitute)) {
      return absl::FailedPreconditionError(absl::Substitute(
          "Missing formatter for argument $0 in debug print custom call", i));
    }
    TF_ASSIGN_OR_RETURN(Literal literal,
                        ConvertToLiteral(stream, args_buffers[i]));

    formatted =
        absl::StrReplaceAll(formatted, {{to_substitute, literal.ToString()}});
  }

  LOG(INFO) << formatted;

  return absl::OkStatus();
}

std::string GetUniqueFilenameForHost() {
  return absl::StrCat(tsl::port::Hostname(), tsl::port::JobName(),
                      tsl::port::TaskId(), tsl::port::JobUid(), ".tfrecord");
}

// This custom call copies its argument to the host and appends it to file.
absl::Status AppendToFileCustomCall(se::Stream* stream, ffi::AnyBuffer buffer,
                                    absl::string_view dir,
                                    absl::string_view metadata,
                                    ffi::Result<ffi::Buffer<xla::TOKEN>> res) {
  if (!stream) {
    return Internal("Stream is nullptr.");
  }
  static absl::Mutex host_mutex{absl::kConstInit};

  TF_ASSIGN_OR_RETURN(Literal literal, ConvertToLiteral(stream, buffer));

  auto* env = tsl::Env::Default();
  std::string destination{dir};
  TF_RETURN_IF_ERROR(env->RecursivelyCreateDir(destination));
  std::string path = tsl::io::JoinPath(destination, GetUniqueFilenameForHost());

  // Supports tensors 2+GB. Should not be serialized as proto.
  TF_ASSIGN_OR_RETURN(std::string serialized, literal.SerializeAsString());

  std::unique_ptr<tsl::WritableFile> file;
  std::string filename(path);

  {
    absl::MutexLock lock(host_mutex);

    TF_RETURN_IF_ERROR(env->NewAppendableFile(filename, &file));
    tsl::io::RecordWriter writer(file.get());

    TF_RETURN_IF_ERROR(writer.WriteRecord(metadata));
    TF_RETURN_IF_ERROR(writer.WriteRecord(serialized));

    TF_RETURN_IF_ERROR(writer.Close());
  }

  return absl::OkStatus();
}

}  // namespace

// This custom call copies its arguments to the host and pretty-prints them as
// an info log. It takes in a "format" attribute to help identify the arguments
// in the log. "Format" follows the convention of `absl::Substitute`, i.e.,
// positional arguments are specified by `$0`, `$1`, etc.
XLA_FFI_DEFINE_HANDLER(kXlaGpuDebugPrintCustomCall, DebugPrintCustomCall,
                       ffi::Ffi::Bind()
                           .Ctx<ffi::Stream>()
                           .RemainingArgs()
                           .Attr<absl::string_view>("format")
                           .Ret<xla::ffi::Buffer<xla::TOKEN>>());

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), kXlaGpuDebugPrintCustomCallTag,
                         GetGpuPlatformName(), kXlaGpuDebugPrintCustomCall);

XLA_FFI_DEFINE_HANDLER(kXlaGpuAppendToFileCustomCall, AppendToFileCustomCall,
                       ffi::Ffi::Bind()
                           .Ctx<ffi::Stream>()
                           .Arg<ffi::AnyBuffer>()
                           .Attr<absl::string_view>("dir")
                           .Attr<absl::string_view>("metadata")
                           .Ret<xla::ffi::Buffer<xla::TOKEN>>());

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), kXlaGpuAppendToFileCustomCallTag,
                         GetGpuPlatformName(), kXlaGpuAppendToFileCustomCall);

XLA_FFI_DEFINE_HANDLER(kXlaGpuAssertCustomCall, AssertionCustomCall,
                       ffi::Ffi::Bind()
                           .Ctx<ffi::Stream>()
                           .Arg<ffi::Buffer<xla::PRED>>()
                           .Attr<absl::string_view>("error_msg")
                           .Ret<xla::ffi::Buffer<xla::TOKEN>>());

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), kXlaGpuAssertCustomCallTag,
                         GetGpuPlatformName(), kXlaGpuAssertCustomCall);

// This allows measuring exported HLOs where kOutfeed and kSendDone has been
// replaced with NopReturnToken. In that case the runtime of the original
// kOutfeed and kSendDone operations is not measured.
XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM(
    std::string(kNopReturnTokenCustomCallTarget), NopReturnTokenCustomCall,
    GetGpuPlatformName());

}  // namespace xla
