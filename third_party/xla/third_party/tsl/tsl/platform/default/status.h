/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_TSL_PLATFORM_DEFAULT_STATUS_H_
#define TENSORFLOW_TSL_PLATFORM_DEFAULT_STATUS_H_

#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/macros.h"
#include "tsl/platform/stacktrace.h"

#define MAYBE_ADD_SOURCE_LOCATION(status) \
  {}

#define ADD_SOURCE_LOCATION(status) status

namespace tsl {

// Stream object used to collect error messages in MAKE_ERROR macros
// or append error messages with APPEND_ERROR.  It accepts any
// arguments with operator<< to build an error string, and then has an
// implicit cast operator to Status, which converts the
// logged string to a Status object and returns it, after logging the
// error.  At least one call to operator<< is required; a compile time
// error will be generated if none are given. Errors will only be
// logged by default for certain status codes, as defined in
// IsLoggedByDefault. This class will give ERROR errors if you don't
// retrieve a Status exactly once before destruction.
//
// The class converts into an intermediate wrapper object
// MakeErrorStreamWithOutput to check that the error stream gets at least one
// item of input.
class MakeErrorStream {
 public:
  // Wrapper around MakeErrorStream that only allows for output. This
  // is created as output of the first operator<< call on
  // MakeErrorStream. The bare MakeErrorStream does not have a
  // Status operator. The net effect of that is that you
  // have to call operator<< at least once or else you'll get a
  // compile time error.
  class MakeErrorStreamWithOutput {
   public:
    explicit MakeErrorStreamWithOutput(MakeErrorStream* error_stream)
        : wrapped_error_stream_(error_stream) {}

    template <typename T>
    MakeErrorStreamWithOutput& operator<<(const T& value) {
      *wrapped_error_stream_ << value;
      return *this;
    }

    // Implicit cast operators to Status and StatusOr.
    // Exactly one of these must be called exactly once before destruction.
    operator absl::Status() { return wrapped_error_stream_->GetStatus(); }

    template <typename T>
    operator absl::StatusOr<T>() {
      return wrapped_error_stream_->GetStatus();
    }

   private:
    MakeErrorStream* wrapped_error_stream_;

    MakeErrorStreamWithOutput(const MakeErrorStreamWithOutput&) = delete;
    MakeErrorStreamWithOutput& operator=(const MakeErrorStreamWithOutput&) =
        delete;
  };

  // When starting from an existing error status, this determines whether we'll
  // append or prepend to that status's error message.
  enum PriorMessageHandling { kAppendToPriorMessage, kPrependToPriorMessage };

  TF_ATTRIBUTE_NOINLINE
  MakeErrorStream(const char* file, int line)
      : impl_(new Impl(file, line, absl::StatusCode::kInternal, this, true)) {}

  template <typename T>
  MakeErrorStreamWithOutput& operator<<(const T& value) {
    CheckNotDone();
    impl_->stream_ << value;
    return impl_->make_error_stream_with_output_wrapper_;
  }

  // When this message is logged (see with_logging()), include the stack trace.
  MakeErrorStream& with_log_stack_trace() {
    impl_->should_log_stack_trace_ = true;
    return *this;
  }

  // Adds RET_CHECK failure text to error message.
  MakeErrorStreamWithOutput& add_ret_check_failure(const char* condition) {
    return *this << "RET_CHECK failure (" << impl_->file_ << ":" << impl_->line_
                 << ") " << condition << " ";
  }

 private:
  class Impl {
   public:
    Impl(const char* file, int line, absl::StatusCode code,
         MakeErrorStream* error_stream, bool is_logged_by_default = true)
        : file_(file),
          line_(line),
          code_(static_cast<absl::StatusCode>(code)),
          is_done_(false),
          should_log_(is_logged_by_default),
          log_severity_(tsl::ERROR),
          should_log_stack_trace_(false),
          make_error_stream_with_output_wrapper_(error_stream) {}

    Impl(const absl::Status& status,
         PriorMessageHandling prior_message_handling, const char* file,
         int line, MakeErrorStream* error_stream)
        : file_(file),
          line_(line),
          // Make sure we show some error, even if the call is incorrect.
          code_(!status.ok() ? static_cast<absl::StatusCode>(status.code())
                             : absl::StatusCode::kUnknown),
          prior_message_handling_(prior_message_handling),
          prior_message_(status.message()),
          is_done_(false),
          // Error code type is not visible here, so we can't call
          // IsLoggedByDefault.
          should_log_(true),
          log_severity_(tsl::ERROR),
          should_log_stack_trace_(false),
          make_error_stream_with_output_wrapper_(error_stream) {
      DCHECK(!status.ok())
          << "Attempted to append/prepend error text to status OK";
    }

    ~Impl() {
      // Note: error messages refer to the public MakeErrorStream class.
      if (!is_done_) {
        LOG(ERROR) << "MakeErrorStream destructed without getting Status: "
                   << file_ << ":" << line_ << " " << stream_.str();
      }
    }

    // This must be called exactly once before destruction.
    absl::Status GetStatus() {
      // Note: error messages refer to the public MakeErrorStream class.

      // Getting a Status object out more than once is not harmful, but
      // it doesn't match the expected pattern, where the stream is constructed
      // as a temporary, loaded with a message, and then casted to Status.
      if (is_done_) {
        LOG(ERROR) << "MakeErrorStream got Status more than once: " << file_
                   << ":" << line_ << " " << stream_.str();
      }

      is_done_ = true;

      const std::string& stream_str = stream_.str();
      const std::string str = prior_message_handling_ == kAppendToPriorMessage
                                  ? absl::StrCat(prior_message_, stream_str)
                                  : absl::StrCat(stream_str, prior_message_);
      if (ABSL_PREDICT_FALSE(str.empty())) {
        return MakeError(
            file_, line_, code_,
            absl::StrCat(str, "Error without message at ", file_, ":", line_),
            true /* should_log */, tsl::ERROR /* log_severity */,
            should_log_stack_trace_);
      } else {
        return MakeError(file_, line_, code_, str, should_log_, log_severity_,
                         should_log_stack_trace_);
      }
    }

    void CheckNotDone() const {
      if (is_done_) {
        LOG(ERROR) << "MakeErrorStream shift called after getting Status: "
                   << file_ << ":" << line_ << " " << stream_.str();
      }
    }

   private:
    // Log the error at the given severity, optionally with a stack trace.
    // If log_severity is NUM_SEVERITIES, nothing is logged.
    static void LogError(const absl::Status& status, const char* filename,
                         int line, int log_severity,
                         bool should_log_stack_trace) {
      if (ABSL_PREDICT_TRUE(log_severity != tsl::NUM_SEVERITIES)) {
        std::string stack_trace;
        if (should_log_stack_trace) {
          stack_trace = absl::StrCat("\n", tsl::CurrentStackTrace());
        }
        switch (log_severity) {
          case tsl::INFO:
            LOG(INFO) << status << stack_trace;
            break;
          case tsl::WARNING:
            LOG(WARNING) << status << stack_trace;
            break;
          case tsl::ERROR:
            LOG(ERROR) << status << stack_trace;
            break;
          case tsl::FATAL:
            LOG(FATAL) << status << stack_trace;  // Crash OK
            break;
          case tsl::NUM_SEVERITIES:
            break;
          default:
            LOG(FATAL) << "Unknown LOG severity " << log_severity;  // Crash OK
        }
      }
    }

    // Make a Status with a code, error message and payload,
    // and also send it to LOG(<log_severity>) using the given filename
    // and line (unless should_log is false, or log_severity is
    // NUM_SEVERITIES).  If should_log_stack_trace is true, the stack
    // trace is included in the log message (ignored if should_log is
    // false).
    static absl::Status MakeError(const char* filename, int line,
                                  absl::StatusCode code,
                                  const std::string& message, bool should_log,
                                  int log_severity,
                                  bool should_log_stack_trace) {
      if (ABSL_PREDICT_FALSE(code == absl::StatusCode::kOk)) {
        LOG(ERROR) << "Cannot create error with status OK";
        code = absl::StatusCode::kUnknown;
      }
      const absl::Status status = absl::Status(code, message);
      if (ABSL_PREDICT_TRUE(should_log)) {
        LogError(status, filename, line, log_severity, should_log_stack_trace);
      }
      return status;
    }

    const char* file_;
    int line_;
    absl::StatusCode code_;

    PriorMessageHandling prior_message_handling_ = kAppendToPriorMessage;
    std::string prior_message_;
    bool is_done_;  // true after Status object has been returned
    std::ostringstream stream_;
    bool should_log_;
    int log_severity_;
    bool should_log_stack_trace_;

    // Wrapper around the MakeErrorStream object that has a
    // Status conversion. The first << operator called on
    // MakeErrorStream will return this object, and only this object
    // can implicitly convert to Status. The net effect of
    // this is that you'll get a compile time error if you call
    // MAKE_ERROR etc. without adding any output.
    MakeErrorStreamWithOutput make_error_stream_with_output_wrapper_;

    friend class MakeErrorStream;
    Impl(const Impl&) = delete;
    Impl& operator=(const Impl&) = delete;
  };

  void CheckNotDone() const { impl_->CheckNotDone(); }

  // Returns the status. Used by MakeErrorStreamWithOutput.
  absl::Status GetStatus() const { return impl_->GetStatus(); }

  // Store the actual data on the heap to reduce stack frame sizes.
  std::unique_ptr<Impl> impl_;

  MakeErrorStream(const MakeErrorStream&) = delete;
  MakeErrorStream& operator=(const MakeErrorStream&) = delete;
};

}  // namespace tsl

#define TF_RET_CHECK(condition)                   \
  while (ABSL_PREDICT_FALSE(!(condition)))        \
  return tsl::MakeErrorStream(__FILE__, __LINE__) \
      .with_log_stack_trace()                     \
      .add_ret_check_failure(#condition)

#endif  // TENSORFLOW_TSL_PLATFORM_DEFAULT_STATUS_H_
