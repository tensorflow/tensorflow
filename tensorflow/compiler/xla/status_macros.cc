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

#include "tensorflow/compiler/xla/status_macros.h"

#include <algorithm>

#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/stacktrace.h"

namespace xla {
namespace status_macros {

ABSL_CONST_INIT const char kPossibleAutoJitAlternative[] =
    "This error might be occurring with the use of xla.compile. If it is not "
    "necessary that every Op be compiled with XLA, an alternative is to use "
    "auto_jit with OptimizerOptions.global_jit_level = ON_2 or the environment "
    "variable TF_XLA_FLAGS=\"tf_xla_auto_jit=2\" which will attempt to use xla "
    "to compile as much of the graph as the compiler is able to.";

static Status MakeStatus(tensorflow::error::Code code,
                         const std::string& message) {
  return Status(code, message);
}

// Log the error at the given severity, optionally with a stack trace.
// If log_severity is NUM_SEVERITIES, nothing is logged.
static void LogError(const Status& status, const char* filename, int line,
                     int log_severity, bool should_log_stack_trace) {
  if (TF_PREDICT_TRUE(log_severity != tensorflow::NUM_SEVERITIES)) {
    std::string stack_trace;
    if (should_log_stack_trace) {
      stack_trace = absl::StrCat("\n", tensorflow::CurrentStackTrace());
    }
    switch (log_severity) {
      case tensorflow::INFO:
        LOG(INFO) << status << stack_trace;
        break;
      case tensorflow::WARNING:
        LOG(WARNING) << status << stack_trace;
        break;
      case tensorflow::ERROR:
        LOG(ERROR) << status << stack_trace;
        break;
      case tensorflow::FATAL:
        LOG(FATAL) << status << stack_trace;
        break;
      case tensorflow::NUM_SEVERITIES:
        break;
      default:
        LOG(FATAL) << "Unknown LOG severity " << log_severity;
    }
  }
}

// Make a Status with a code, error message and payload,
// and also send it to LOG(<log_severity>) using the given filename
// and line (unless should_log is false, or log_severity is
// NUM_SEVERITIES).  If should_log_stack_trace is true, the stack
// trace is included in the log message (ignored if should_log is
// false).
static Status MakeError(const char* filename, int line,
                        tensorflow::error::Code code,
                        const std::string& message, bool should_log,
                        int log_severity, bool should_log_stack_trace) {
  if (TF_PREDICT_FALSE(code == tensorflow::error::OK)) {
    LOG(ERROR) << "Cannot create error with status OK";
    code = tensorflow::error::UNKNOWN;
  }
  const Status status = MakeStatus(code, message);
  if (TF_PREDICT_TRUE(should_log)) {
    LogError(status, filename, line, log_severity, should_log_stack_trace);
  }
  return status;
}

// This method is written out-of-line rather than in the header to avoid
// generating a lot of inline code for error cases in all callers.
void MakeErrorStream::CheckNotDone() const { impl_->CheckNotDone(); }

MakeErrorStream::Impl::Impl(const char* file, int line,
                            tensorflow::error::Code code,
                            MakeErrorStream* error_stream,
                            bool is_logged_by_default)
    : file_(file),
      line_(line),
      code_(code),
      is_done_(false),
      should_log_(is_logged_by_default),
      log_severity_(tensorflow::ERROR),
      should_log_stack_trace_(false),
      make_error_stream_with_output_wrapper_(error_stream) {}

MakeErrorStream::Impl::Impl(const Status& status,
                            PriorMessageHandling prior_message_handling,
                            const char* file, int line,
                            MakeErrorStream* error_stream)
    : file_(file),
      line_(line),
      // Make sure we show some error, even if the call is incorrect.
      code_(!status.ok() ? status.code() : tensorflow::error::UNKNOWN),
      prior_message_handling_(prior_message_handling),
      prior_message_(status.error_message()),
      is_done_(false),
      // Error code type is not visible here, so we can't call
      // IsLoggedByDefault.
      should_log_(true),
      log_severity_(tensorflow::ERROR),
      should_log_stack_trace_(false),
      make_error_stream_with_output_wrapper_(error_stream) {
  DCHECK(!status.ok()) << "Attempted to append/prepend error text to status OK";
}

MakeErrorStream::Impl::~Impl() {
  // Note: error messages refer to the public MakeErrorStream class.

  if (!is_done_) {
    LOG(ERROR) << "MakeErrorStream destructed without getting Status: " << file_
               << ":" << line_ << " " << stream_.str();
  }
}

Status MakeErrorStream::Impl::GetStatus() {
  // Note: error messages refer to the public MakeErrorStream class.

  // Getting a Status object out more than once is not harmful, but
  // it doesn't match the expected pattern, where the stream is constructed
  // as a temporary, loaded with a message, and then casted to Status.
  if (is_done_) {
    LOG(ERROR) << "MakeErrorStream got Status more than once: " << file_ << ":"
               << line_ << " " << stream_.str();
  }

  is_done_ = true;

  const std::string& stream_str = stream_.str();
  const std::string str = prior_message_handling_ == kAppendToPriorMessage
                              ? absl::StrCat(prior_message_, stream_str)
                              : absl::StrCat(stream_str, prior_message_);
  if (TF_PREDICT_FALSE(str.empty())) {
    return MakeError(
        file_, line_, code_,
        absl::StrCat(str, "Error without message at ", file_, ":", line_),
        true /* should_log */, tensorflow::ERROR /* log_severity */,
        should_log_stack_trace_);
  } else {
    return MakeError(file_, line_, code_, str, should_log_, log_severity_,
                     should_log_stack_trace_);
  }
}

void MakeErrorStream::Impl::CheckNotDone() const {
  if (is_done_) {
    LOG(ERROR) << "MakeErrorStream shift called after getting Status: " << file_
               << ":" << line_ << " " << stream_.str();
  }
}

}  // namespace status_macros
}  // namespace xla
