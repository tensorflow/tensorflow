/* Copyright 2015 The OpenXLA Authors.

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

#include "xla/stream_executor/stream.h"

#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <utility>
#include <variant>
#include <vector>

#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "Eigen/Core"  // from @eigen_archive
#include "xla/stream_executor/blas.h"
#include "xla/stream_executor/numeric_options.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform/port.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/stream_executor_internal.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/stacktrace.h"

namespace stream_executor {

namespace {
// Code to turn parameters to functions on stream into strings that
// will be VLOG'ed. We need overloads, instead of
// e.g. BatchDescriptorToVlogString(), as the code that calls these
// functions does not know what the type of the parameter is.
std::string ToVlogString(const dnn::BatchDescriptor &descriptor) {
  return descriptor.ToShortString();
}

std::string ToVlogString(const dnn::FilterDescriptor &descriptor) {
  return descriptor.ToShortString();
}

std::string ToVlogString(const dnn::ConvolutionDescriptor &descriptor) {
  return descriptor.ToShortString();
}

std::string ToVlogString(const dnn::PoolingDescriptor &descriptor) {
  return descriptor.ToShortString();
}

std::string ToVlogString(const dnn::NormalizeDescriptor &descriptor) {
  return descriptor.ToShortString();
}

std::string ToVlogString(dnn::ActivationMode mode) {
  return dnn::ActivationModeString(mode);
}

std::string ToVlogString(const dnn::AlgorithmConfig &algo_config) {
  return algo_config.ToString();
}

std::string ToVlogString(dnn::ElementwiseOperation op) {
  return dnn::ElementwiseOperationString(op);
}

std::string ToVlogString(dnn::QuantizedActivationMode mode) {
  return dnn::QuantizedActivationModeString(mode);
}

std::string ToVlogString(blas::Transpose t) { return blas::TransposeString(t); }

std::string ToVlogString(blas::UpperLower ul) {
  return blas::UpperLowerString(ul);
}

std::string ToVlogString(blas::Diagonal d) { return blas::DiagonalString(d); }

std::string ToVlogString(blas::Side s) { return blas::SideString(s); }

std::string ToVlogString(blas::ComputationType ty) {
  return blas::ComputationTypeString(ty);
}

std::string ToVlogString(const void *ptr) {
  if (ptr == nullptr) {
    return "null";
  }

  // StrCat does not convert pointers to text.
  std::ostringstream out;
  out << ptr;
  return out.str();
}

template <class T>
std::string ToVlogString(const std::complex<T> &c) {
  // StrCat does not convert std::complex to text.
  std::ostringstream out;
  out << c;
  return out.str();
}

template <class T>
std::string ToVlogString(const std::function<T> &f) {
  return f == nullptr ? "null" : "<non-null function>";
}

template <class T>
std::string ToVlogString(const absl::AnyInvocable<T> &f) {
  return f == nullptr ? "null" : "<non-null function>";
}

std::string ToVlogString(const DeviceMemoryBase &memory) {
  return ToVlogString(memory.opaque());
}

std::string ToVlogString(const DeviceMemoryBase *memory) {
  return memory == nullptr ? "null" : ToVlogString(*memory);
}

std::string ToVlogString(const Eigen::half &h) {
  return absl::StrCat(static_cast<float>(h));
}

std::string ToVlogString(const Eigen::bfloat16 &bf) {  // NOLINT
  return absl::StrCat(static_cast<float>(bf));
}

std::string ToVlogString(int i) { return absl::StrCat(i); }

std::string ToVlogString(uint32_t i) { return absl::StrCat(i); }

std::string ToVlogString(uint64_t i) { return absl::StrCat(i); }

std::string ToVlogString(int64_t i) { return absl::StrCat(i); }

std::string ToVlogString(float f) { return absl::StrCat(f); }

std::string ToVlogString(double d) { return absl::StrCat(d); }

template <class T>
std::string ToVlogString(absl::Span<const T> elements) {
  std::string str = absl::StrCat(
      ToVlogString(reinterpret_cast<const void *>(elements.data())), "[",
      elements.size(), "]{");
  const char *separator = "";
  size_t max_to_show = std::numeric_limits<size_t>::max();
  if (!VLOG_IS_ON(2)) {
    max_to_show = 5;
  } else if (!VLOG_IS_ON(3)) {
    max_to_show = 20;
  } else if (!VLOG_IS_ON(11)) {
    max_to_show = 1000;
  }
  for (size_t i = 0; i < elements.size(); ++i) {
    if (i == max_to_show) {
      str += ", ...";
      break;
    }
    absl::StrAppend(&str, separator, ToVlogString(elements[i]));
    separator = ", ";
  }
  str += "}";
  return str;
}

template <class T>
std::string ToVlogString(absl::Span<T> elements) {
  return ToVlogString(absl::Span<const T>(elements));
}

std::string ToVlogString(dnn::DataType data_type) {
  switch (data_type) {
    case dnn::DataType::kFloat:
      return "dnn::DataType::kFloat";
    case dnn::DataType::kDouble:
      return "dnn::DataType::kDouble";
    case dnn::DataType::kHalf:
      return "dnn::DataType::kHalf";
    case dnn::DataType::kInt8:
      return "dnn::DataType::kInt8";
    case dnn::DataType::kInt32:
      return "dnn::DataType::kInt32";
    case dnn::DataType::kBF16:
      return "dnn::DataType::kBF16";
    default:
      return "unknown DataType";
  }
}

// Used together with PARAM to VLOG calls made to the stream. Intended
// to be used like this:
//
//   VLOG(1) << CallStr("MyFunction", this, {PARAM(a), PARAM(b)});
//
// where a and b are the parameters to MyFunction.
//
// See VLOG_CALL for a short-hand for this. This way of doing it saves
// a tremendous amount of boilerplate code given how many functions
// there are on Stream and how many parameters they each have.
std::string CallStr(const char *function_name, Stream *stream,
                    std::vector<std::pair<const char *, std::string>> params) {
  // Do not call this function unless VLOG is on since just
  // constructing all the strings in params is expensive.
  CHECK(VLOG_IS_ON(1));

  std::string str = absl::StrCat(stream->DebugStreamPointers(),
                                 " Called Stream::", function_name, "(");
  const char *separator = "";
  for (const auto &param : params) {
    absl::StrAppend(&str, separator, param.first, "=", param.second);
    separator = ", ";
  }
  absl::StrAppend(&str, ")");
  if (VLOG_IS_ON(10)) {
    absl::StrAppend(&str, " ", tsl::CurrentStackTrace(), "\n");
  }
  return str;
}

// Use this macro to avoid having to type every parameter twice to log
// it with VLOG and CallStr.
#define PARAM(parameter) \
  { #parameter, ToVlogString(parameter) }

// Use this macro to avoid having to type out the name of each
// function and to save some boilerplate. Intended to be used like this:
//
//   VLOG_CALL(PARAM(a), PARAM(b))
//
// This saves a tremendous amount of boilerplate compared to the alternative:
//
//   VLOG(1) << "Calling MyFunction(a=" << ToVlogString(a)
//           << ", b=" << ToVlogString(b);
//
// Note here that most of the parameter names are not short and that
// most of the functions take many more than 2 parameters.
#define VLOG_CALL(...) VLOG(1) << CallStr(__func__, this, {__VA_ARGS__})

}  // namespace

Stream::Stream(StreamExecutor *parent)
    : parent_(parent),
      implementation_(parent->implementation()->GetStreamImplementation()),
      allocated_(false),
      status_(absl::InternalError("Uninitialized stream")) {
  VLOG_CALL(PARAM(parent));
}

Stream::~Stream() {
  VLOG_CALL();

  // Ensure the stream is completed.
  auto status = BlockHostUntilDone();
  if (!status.ok()) {
    LOG(WARNING) << "Error blocking host until done in stream destructor: "
                 << status;
  }

  if (allocated_) {
    parent_->DeallocateStream(this);
  }
}

void Stream::SetPriority(StreamPriority priority) {
  implementation_->SetPriority(priority);
}

void Stream::SetPriority(int priority) {
  implementation_->SetPriority(priority);
}

std::variant<StreamPriority, int> Stream::priority() const {
  return implementation_->priority();
}

Stream::PlatformSpecificHandle Stream::platform_specific_handle() const {
  PlatformSpecificHandle handle;
  handle.stream = implementation_->platform_specific_stream();
  return handle;
}

absl::Status Stream::RefreshStatus() {
  absl::Status status = parent_->GetStatus(this);
  // We should not put the stream in an error state, just because the GetStatus
  // method is unimplemented.
  if (status != absl::UnimplementedError(
                    "GetStatus is not supported on this executor.")) {
    CheckStatus(status);
  }
  return status;
}

Stream &Stream::Init() {
  VLOG_CALL();

  absl::MutexLock lock(&mu_);
  CHECK_EQ(false, allocated_)
      << "stream appears to already have been initialized";
  CHECK(!status_.ok()) << "stream should be in !ok() state pre-initialization";

  if (parent_->AllocateStream(this)) {
    // Successful initialization!
    allocated_ = true;
    status_ = absl::OkStatus();
  } else {
    LOG(ERROR) << "failed to allocate stream during initialization";
  }

  return *this;
}

Stream &Stream::ThenRecordEvent(Event *event) {
  VLOG_CALL(PARAM(event));

  absl::Status status = parent_->RecordEvent(this, event);
  if (!status.ok()) {
    LOG(ERROR) << "Error recording event in stream: " << status.message()
               << "; not marking stream as bad, as the Event object may be "
               << "at fault. Monitor for further errors.";
  }

  return *this;
}

Stream &Stream::ThenDepthConcatenate(
    absl::Span<const dnn::BatchDescriptor> input_dimensions,
    absl::Span<const DeviceMemory<float> *const> input_data,
    DeviceMemory<float> *output_data) {
  VLOG_CALL(PARAM(input_dimensions), PARAM(input_data), PARAM(output_data));

  for (size_t i = 1; i < input_dimensions.size(); ++i) {
    if (input_dimensions[i].count() != input_dimensions[0].count() ||
        input_dimensions[i].height() != input_dimensions[0].height() ||
        input_dimensions[i].width() != input_dimensions[0].width()) {
      SetError();
      LOG(ERROR) << "Incompatible dimensions for depth concatenation.\n"
                 << "input_dimensions[0]: " << input_dimensions[0].ToString()
                 << "input_dimensions[" << i
                 << "]: " << input_dimensions[i].ToString();
      return *this;
    }
  }

  if (dnn::DnnSupport *dnn = parent_->AsDnn()) {
    CheckError(dnn->DoDepthConcatenate(this, input_dimensions, input_data,
                                       output_data));
  } else {
    SetErrorAndLogNoDnnSupport();
  }
  return *this;
}

Stream *Stream::GetOrCreateSubStream() {
  // Do not destroy bad streams when holding mu_ because ~Stream() may
  // BlockHostUntilDone and it's host callbacks might attempt to acquire mu_.
  std::vector<std::unique_ptr<Stream>> bad_streams;

  absl::MutexLock lock(&mu_);

  // Look for the first reusable sub_stream that is ok, dropping !ok sub_streams
  // we encounter along the way.
  for (size_t index = 0; index < sub_streams_.size();) {
    std::pair<std::unique_ptr<Stream>, bool> &pair = sub_streams_[index];
    if (pair.second) {
      // The sub_stream is reusable.
      Stream *sub_stream = pair.first.get();
      if (sub_stream->ok()) {
        VLOG(1) << DebugStreamPointers() << " reusing sub_stream "
                << sub_stream->DebugStreamPointers();
        pair.second = false;
        return sub_stream;
      }

      // The stream is reusable and not ok. Streams have a monotonic state
      // machine; the stream will remain in !ok forever. Swap it with the last
      // stream and pop it off.
      const int64_t last = sub_streams_.size() - 1;
      if (index != last) {
        std::swap(pair, sub_streams_[last]);
      }
      bad_streams.push_back(std::move(sub_streams_.back().first));
      sub_streams_.pop_back();
      VLOG(1) << DebugStreamPointers() << " dropped !ok sub_stream "
              << sub_stream->DebugStreamPointers();
    } else {
      // The sub_stream is not reusable, move on to the next one.
      ++index;
    }
  }

  // No streams are reusable; create a new stream.
  sub_streams_.emplace_back(std::unique_ptr<Stream>{new Stream{parent_}},
                            false);
  Stream *sub_stream = sub_streams_.back().first.get();
  sub_stream->Init();
  if (!sub_stream->ok()) {
    LOG(ERROR) << "sub-stream failed to be initialized";
  }
  VLOG(1) << DebugStreamPointers() << " created new sub_stream "
          << sub_stream->DebugStreamPointers();

  return sub_stream;
}

void Stream::ReturnSubStream(Stream *sub_stream) {
  // Do not destroy bad streams when holding mu_ because ~Stream() may
  // BlockHostUntilDone and it's host callbacks might attempt to acquire mu_.
  std::unique_ptr<Stream> bad_stream;

  absl::MutexLock lock(&mu_);

  // Look for the sub-stream.
  for (int64_t index = 0, end = sub_streams_.size(); index < end; ++index) {
    std::pair<std::unique_ptr<Stream>, bool> &pair = sub_streams_[index];
    if (pair.first.get() != sub_stream) {
      continue;
    }

    // Found the sub_stream.
    if (sub_stream->ok()) {
      VLOG(1) << DebugStreamPointers() << " returned ok sub_stream "
              << sub_stream->DebugStreamPointers();
      pair.second = true;
    } else {
      // The returned stream is not ok. Streams have a monotonic state
      // machine; the stream will remain in !ok forever. Swap it with the last
      // stream and pop it off.
      VLOG(1) << DebugStreamPointers() << " returned !ok sub_stream "
              << sub_stream->DebugStreamPointers();
      const int64_t last = sub_streams_.size() - 1;
      if (index != last) {
        std::swap(pair, sub_streams_[last]);
      }
      std::swap(bad_stream, sub_streams_.back().first);
      sub_streams_.pop_back();
    }
    return;
  }

  LOG(FATAL) << DebugStreamPointers()
             << " did not create the returned sub-stream "
             << sub_stream->DebugStreamPointers();
}

Stream &Stream::ThenWaitFor(Stream *other) {
  VLOG_CALL(PARAM(other));

  CHECK(this != other) << "stream cannot wait for itself";
  if (ok() && other->ok()) {
    CheckError(parent_->CreateStreamDependency(this, other));
  } else {
    SetError();
    LOG(INFO) << DebugStreamPointers() << " did not wait for "
              << other->DebugStreamPointers();
  }
  return *this;
}

Stream &Stream::ThenWaitFor(Event *event) {
  VLOG_CALL(PARAM(event));

  if (ok()) {
    absl::Status status = parent_->WaitForEvent(this, event);
    if (!status.ok()) {
      LOG(ERROR) << "Error waiting for event in stream: " << status.message()
                 << "; not marking stream as bad, as the Event object may be "
                 << "at fault. Monitor for further errors.";
    }
  } else {
    LOG(INFO) << DebugStreamPointers() << " did not wait for an event.";
  }
  return *this;
}

// A functor that implements ThenBlasXXX interfaces, which calls DoBlasXXX
// functions and logs for errors.
template <typename... Args>
struct ThenBlasImpl {
  // blas_func is the DoBlasXXX member function pointer, and args are its
  // arguments except the first one of Stream* type.
  Stream &operator()(Stream *stream,
                     bool (blas::BlasSupport::*blas_func)(Stream *, Args...),
                     Args... args) {
    return Run(stream, blas_func, /*record_error=*/true, args...);
  }

  // Like operator(), but only calls stream->CheckError() if record_error is
  // true.
  Stream &Run(Stream *stream,
              bool (blas::BlasSupport::*blas_func)(Stream *, Args...),
              bool record_error, Args... args);
};

template <typename... Args>
Stream &ThenBlasImpl<Args...>::Run(
    Stream *stream, bool (blas::BlasSupport::*blas_func)(Stream *, Args...),
    bool record_error, Args... args) {
  if (stream->ok()) {
    bool ok;
    if (blas::BlasSupport *blas = stream->parent_->AsBlas()) {
      ok = (blas->*blas_func)(stream, args...);
    } else {
      LOG(WARNING)
          << "attempting to perform BLAS operation using StreamExecutor "
             "without BLAS support";
      ok = false;
    }
    if (record_error) {
      stream->CheckError(ok);
    }
  }
  return *stream;
}

Stream &Stream::ThenBlasGemmBatchedWithScratch(
    blas::Transpose transa, blas::Transpose transb, uint64_t m, uint64 n,
    uint64_t k, float alpha, DeviceMemorySlice<Eigen::half> a, int lda,
    DeviceMemorySlice<Eigen::half> b, int ldb, float beta,
    DeviceMemorySlice<Eigen::half> c, int ldc, int batch_count,
    const NumericOptions &numeric_options, ScratchAllocator *scratch_allocator,
    blas::CallContext context) {
  VLOG_CALL(PARAM(transa), PARAM(transb), PARAM(m), PARAM(n), PARAM(k),
            PARAM(alpha), PARAM(a), PARAM(lda), PARAM(b), PARAM(ldb),
            PARAM(beta), PARAM(c), PARAM(ldc), PARAM(batch_count));

  ThenBlasImpl<blas::Transpose, blas::Transpose, uint64_t, uint64_t, uint64,
               float, DeviceMemorySlice<Eigen::half>, int,
               DeviceMemorySlice<Eigen::half>, int, float,
               DeviceMemorySlice<Eigen::half>, int, int, const NumericOptions &,
               ScratchAllocator *, blas::CallContext>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasGemmBatched, transa, transb, m, n,
              k, alpha, a, lda, b, ldb, beta, c, ldc, batch_count,
              numeric_options, scratch_allocator, context);
}

Stream &Stream::ThenBlasGemmBatchedWithScratch(
    blas::Transpose transa, blas::Transpose transb, uint64_t m, uint64 n,
    uint64_t k, float alpha, DeviceMemorySlice<Eigen::bfloat16> a, int lda,
    DeviceMemorySlice<Eigen::bfloat16> b, int ldb, float beta,
    DeviceMemorySlice<Eigen::bfloat16> c, int ldc, int batch_count,
    const NumericOptions &numeric_options, ScratchAllocator *scratch_allocator,
    blas::CallContext context) {
  VLOG_CALL(PARAM(transa), PARAM(transb), PARAM(m), PARAM(n), PARAM(k),
            PARAM(alpha), PARAM(a), PARAM(lda), PARAM(b), PARAM(ldb),
            PARAM(beta), PARAM(c), PARAM(ldc), PARAM(batch_count));

  ThenBlasImpl<blas::Transpose, blas::Transpose, uint64_t, uint64_t, uint64,
               float, DeviceMemorySlice<Eigen::bfloat16>, int,
               DeviceMemorySlice<Eigen::bfloat16>, int, float,
               DeviceMemorySlice<Eigen::bfloat16>, int, int,
               const NumericOptions &, ScratchAllocator *, blas::CallContext>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasGemmBatched, transa, transb, m, n,
              k, alpha, a, lda, b, ldb, beta, c, ldc, batch_count,
              numeric_options, scratch_allocator, context);
}

Stream &Stream::ThenBlasGemmBatchedWithScratch(
    blas::Transpose transa, blas::Transpose transb, uint64_t m, uint64 n,
    uint64_t k, float alpha, DeviceMemorySlice<float> a, int lda,
    DeviceMemorySlice<float> b, int ldb, float beta, DeviceMemorySlice<float> c,
    int ldc, int batch_count, const NumericOptions &numeric_options,
    ScratchAllocator *scratch_allocator, blas::CallContext context) {
  VLOG_CALL(PARAM(transa), PARAM(transb), PARAM(m), PARAM(n), PARAM(k),
            PARAM(alpha), PARAM(a), PARAM(lda), PARAM(b), PARAM(ldb),
            PARAM(beta), PARAM(c), PARAM(ldc), PARAM(batch_count));

  ThenBlasImpl<blas::Transpose, blas::Transpose, uint64_t, uint64_t, uint64,
               float, DeviceMemorySlice<float>, int, DeviceMemorySlice<float>,
               int, float, DeviceMemorySlice<float>, int, int,
               const NumericOptions &, ScratchAllocator *, blas::CallContext>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasGemmBatched, transa, transb, m, n,
              k, alpha, a, lda, b, ldb, beta, c, ldc, batch_count,
              numeric_options, scratch_allocator, context);
}

Stream &Stream::ThenBlasGemmBatchedWithScratch(
    blas::Transpose transa, blas::Transpose transb, uint64_t m, uint64 n,
    uint64_t k, double alpha, DeviceMemorySlice<double> a, int lda,
    DeviceMemorySlice<double> b, int ldb, double beta,
    DeviceMemorySlice<double> c, int ldc, int batch_count,
    const NumericOptions &numeric_options, ScratchAllocator *scratch_allocator,
    blas::CallContext context) {
  VLOG_CALL(PARAM(transa), PARAM(transb), PARAM(m), PARAM(n), PARAM(k),
            PARAM(alpha), PARAM(a), PARAM(lda), PARAM(b), PARAM(ldb),
            PARAM(beta), PARAM(c), PARAM(ldc), PARAM(batch_count));

  ThenBlasImpl<blas::Transpose, blas::Transpose, uint64_t, uint64_t, uint64,
               double, DeviceMemorySlice<double>, int,
               DeviceMemorySlice<double>, int, double,
               DeviceMemorySlice<double>, int, int, const NumericOptions &,
               ScratchAllocator *, blas::CallContext>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasGemmBatched, transa, transb, m, n,
              k, alpha, a, lda, b, ldb, beta, c, ldc, batch_count,
              numeric_options, scratch_allocator, context);
}

Stream &Stream::ThenBlasGemmBatchedWithScratch(
    blas::Transpose transa, blas::Transpose transb, uint64_t m, uint64 n,
    uint64_t k, std::complex<float> alpha,
    DeviceMemorySlice<std::complex<float>> a, int lda,
    DeviceMemorySlice<std::complex<float>> b, int ldb, std::complex<float> beta,
    DeviceMemorySlice<std::complex<float>> c, int ldc, int batch_count,
    const NumericOptions &numeric_options, ScratchAllocator *scratch_allocator,
    blas::CallContext context) {
  VLOG_CALL(PARAM(transa), PARAM(transb), PARAM(m), PARAM(n), PARAM(k),
            PARAM(alpha), PARAM(a), PARAM(lda), PARAM(b), PARAM(ldb),
            PARAM(beta), PARAM(c), PARAM(ldc), PARAM(batch_count));

  ThenBlasImpl<blas::Transpose, blas::Transpose, uint64_t, uint64_t, uint64,
               std::complex<float>, DeviceMemorySlice<std::complex<float>>, int,
               DeviceMemorySlice<std::complex<float>>, int, std::complex<float>,
               DeviceMemorySlice<std::complex<float>>, int, int,
               const NumericOptions &, ScratchAllocator *, blas::CallContext>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasGemmBatched, transa, transb, m, n,
              k, alpha, a, lda, b, ldb, beta, c, ldc, batch_count,
              numeric_options, scratch_allocator, context);
}

Stream &Stream::ThenBlasGemmBatchedWithScratch(
    blas::Transpose transa, blas::Transpose transb, uint64_t m, uint64 n,
    uint64_t k, std::complex<double> alpha,
    DeviceMemorySlice<std::complex<double>> a, int lda,
    DeviceMemorySlice<std::complex<double>> b, int ldb,
    std::complex<double> beta, DeviceMemorySlice<std::complex<double>> c,
    int ldc, int batch_count, const NumericOptions &numeric_options,
    ScratchAllocator *scratch_allocator, blas::CallContext context) {
  VLOG_CALL(PARAM(transa), PARAM(transb), PARAM(m), PARAM(n), PARAM(k),
            PARAM(alpha), PARAM(a), PARAM(lda), PARAM(b), PARAM(ldb),
            PARAM(beta), PARAM(c), PARAM(ldc), PARAM(batch_count));

  ThenBlasImpl<blas::Transpose, blas::Transpose, uint64_t, uint64_t, uint64,
               std::complex<double>, DeviceMemorySlice<std::complex<double>>,
               int, DeviceMemorySlice<std::complex<double>>, int,
               std::complex<double>, DeviceMemorySlice<std::complex<double>>,
               int, int, const NumericOptions &, ScratchAllocator *,
               blas::CallContext>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasGemmBatched, transa, transb, m, n,
              k, alpha, a, lda, b, ldb, beta, c, ldc, batch_count,
              numeric_options, scratch_allocator, context);
}

Stream &Stream::ThenMemcpy(void *host_dst, const DeviceMemoryBase &gpu_src,
                           uint64_t size) {
  VLOG_CALL(PARAM(host_dst), PARAM(gpu_src), PARAM(size));

  CheckError(parent_->Memcpy(this, host_dst, gpu_src, size));
  return *this;
}

Stream &Stream::ThenMemcpy(DeviceMemoryBase *gpu_dst, const void *host_src,
                           uint64_t size) {
  VLOG_CALL(PARAM(gpu_dst), PARAM(host_src), PARAM(size));

  CheckError(parent_->Memcpy(this, gpu_dst, host_src, size));
  return *this;
}

Stream &Stream::ThenMemcpy(DeviceMemoryBase *gpu_dst,
                           const DeviceMemoryBase &gpu_src, uint64_t size) {
  VLOG_CALL(PARAM(gpu_dst), PARAM(gpu_src), PARAM(size));

  CheckError(parent_->MemcpyDeviceToDevice(this, gpu_dst, gpu_src, size));
  return *this;
}

Stream &Stream::ThenMemZero(DeviceMemoryBase *location, uint64_t size) {
  VLOG_CALL(PARAM(location), PARAM(size));

  CheckStatus(parent_->MemZero(this, location, size));
  return *this;
}

Stream &Stream::ThenMemset32(DeviceMemoryBase *location, uint32_t pattern,
                             uint64_t size) {
  VLOG_CALL(PARAM(location), PARAM(pattern), PARAM(size));

  CheckStatus(parent_->Memset32(this, location, pattern, size));
  return *this;
}

Stream &Stream::ThenTransformTensor(const dnn::BatchDescriptor &input_desc,
                                    dnn::DataType input_type,
                                    const DeviceMemoryBase &input_data,
                                    const dnn::BatchDescriptor &output_desc,
                                    dnn::DataType output_type, float scale,
                                    DeviceMemoryBase *output_data) {
  VLOG_CALL(PARAM(input_desc), PARAM(input_type), PARAM(input_data),
            PARAM(output_desc), PARAM(output_type), PARAM(scale),
            PARAM(output_data));
  if (dnn::DnnSupport *dnn = parent_->AsDnn()) {
    CheckError(dnn->DoTransformTensor(this, input_desc, input_type, input_data,
                                      output_desc, output_type, scale,
                                      output_data));
  } else {
    SetErrorAndLogNoDnnSupport();
  }
  return *this;
}

Stream &Stream::ThenDoHostCallback(absl::AnyInvocable<void() &&> callback) {
  return ThenDoHostCallbackWithStatus([cb = std::move(callback)]() mutable {
    std::move(cb)();
    return absl::OkStatus();
  });
}

Stream &Stream::ThenDoHostCallbackWithStatus(
    absl::AnyInvocable<absl::Status() &&> callback) {
  VLOG_CALL(PARAM(callback));

  if (!ok()) {
    LOG(INFO) << DebugStreamPointers()
              << " was in error state before adding host callback";
  }
  CheckError(parent_->HostCallback(this, std::move(callback)));
  return *this;
}

void Stream::CheckError(bool operation_retcode) {
  if (operation_retcode) {
    return;
  }
  absl::MutexLock lock(&mu_);
  status_ = absl::InternalError("Unknown error");
}

// It looks confusing, but all this is doing is inserting a callback at the
// present point in the stream to then enqueue a task on the host executor.
Stream &Stream::ThenEnqueueOnBackgroundThread(
    std::function<void(StreamExecutor *)> task) {
  VLOG_CALL(PARAM(task));

  StreamExecutor *stream_executor = this->parent_;
  std::function<void()> bound_task = std::bind(task, stream_executor);

  return ThenDoHostCallback([stream_executor, bound_task]() {
    stream_executor->EnqueueOnBackgroundThread(bound_task);
  });
}

absl::Status Stream::BlockHostUntilDone() {
  VLOG_CALL();

  if (!ok()) {
    absl::MutexLock lock(&mu_);
    LOG(INFO) << status_.ToString();
    absl::Status status = absl::InternalError(
        "stream did not block host until done; was already in an error state");
    LOG(INFO) << DebugStreamPointers() << " " << status;
    return status;
  }

  absl::Status error = parent_->BlockHostUntilDone(this);
  CheckError(error.ok());

  return error;
}

std::string Stream::DebugStreamPointers() const {
  // Relies on the ToVlogString(const void*) overload above.
  return absl::StrCat("[stream=", ToVlogString(this),
                      ",impl=", ToVlogString(implementation_.get()), "]");
}

void Stream::CheckStatus(absl::Status status) {
  if (status.ok()) {
    return;
  }
  LOG(ERROR) << status;
  absl::MutexLock lock(&mu_);
  status_ = status;
}

}  // namespace stream_executor
