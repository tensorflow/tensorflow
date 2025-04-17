/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/tsl/platform/default/logging.h"

#include <cstdint>
#include <limits>

// TODO(b/142492876): Avoid depending on absl internal.
#include "absl/base/internal/cycleclock.h"
#include "absl/base/internal/sysinfo.h"
#include "absl/base/log_severity.h"
#include "absl/base/optimization.h"
#include "absl/container/flat_hash_map.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/platform/env_time.h"
#include "xla/tsl/platform/macros.h"
#include "tsl/platform/mutex.h"

#if defined(PLATFORM_POSIX_ANDROID)
#include <android/log.h>

#include <iostream>
#include <sstream>
#endif

#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <algorithm>
#include <queue>
#include <unordered_map>

namespace tsl {

namespace internal {
namespace {

// This is an internal singleton class that manages the log sinks. It allows
// adding and removing the log sinks, as well as handling sending log messages
// to all the registered log sinks.
class TFLogSinks {
 public:
  // Gets the TFLogSinks instance. This is the entry point for using this class.
  static TFLogSinks& Instance();

  // Adds a log sink. The sink argument must not be a nullptr. TFLogSinks
  // takes ownership of the pointer, the user must not free the pointer.
  // The pointer will remain valid until the application terminates or
  // until TFLogSinks::Remove is called for the same pointer value.
  void Add(TFLogSink* sink);

  // Removes a log sink. This will also erase the sink object. The pointer
  // to the sink becomes invalid after this call.
  void Remove(TFLogSink* sink);

  // Gets the currently registered log sinks.
  std::vector<TFLogSink*> GetSinks() const;

  // Sends a log message to all registered log sinks.
  //
  // If there are no log sinks are registered:
  //
  // NO_DEFAULT_LOGGER is defined:
  // Up to 128 messages will be queued until a log sink is added.
  // The queue will then be logged to the first added log sink.
  //
  // NO_DEFAULT_LOGGER is not defined:
  // The messages will be logged using the default logger. The default logger
  // will log to stdout on all platforms except for Android. On Androit the
  // default Android logger will be used.
  void Send(const TFLogEntry& entry);

 private:
  TFLogSinks();
  void SendToSink(TFLogSink& sink, const TFLogEntry& entry);

  std::queue<TFLogEntry> log_entry_queue_;
  static const size_t kMaxLogEntryQueueSize = 128;

  mutable tsl::mutex mutex_;
  std::vector<TFLogSink*> sinks_;
};

TFLogSinks::TFLogSinks() {
#ifndef NO_DEFAULT_LOGGER
  static TFDefaultLogSink* const default_sink = new TFDefaultLogSink();
  sinks_.push_back(default_sink);
#endif
}

TFLogSinks& TFLogSinks::Instance() {
  static TFLogSinks* const instance = new TFLogSinks();
  return *instance;
}

void TFLogSinks::Add(TFLogSink* sink) {
  assert(sink != nullptr && "The sink must not be a nullptr");

  tsl::mutex_lock lock(mutex_);
  sinks_.push_back(sink);

  // If this is the only sink log all the queued up messages to this sink
  if (sinks_.size() == 1) {
    while (!log_entry_queue_.empty()) {
      for (const auto& sink : sinks_) {
        SendToSink(*sink, log_entry_queue_.front());
      }
      log_entry_queue_.pop();
    }
  }
}

void TFLogSinks::Remove(TFLogSink* sink) {
  assert(sink != nullptr && "The sink must not be a nullptr");

  tsl::mutex_lock lock(mutex_);
  auto it = std::find(sinks_.begin(), sinks_.end(), sink);
  if (it != sinks_.end()) sinks_.erase(it);
}

std::vector<TFLogSink*> TFLogSinks::GetSinks() const {
  tsl::mutex_lock lock(mutex_);
  return sinks_;
}

void TFLogSinks::Send(const TFLogEntry& entry) {
  tsl::mutex_lock lock(mutex_);

  // If we don't have any sinks registered, queue them up
  if (sinks_.empty()) {
    // If we've exceeded the maximum queue size, drop the oldest entries
    while (log_entry_queue_.size() >= kMaxLogEntryQueueSize) {
      log_entry_queue_.pop();
    }
    log_entry_queue_.push(entry);
    return;
  }

  // If we have items in the queue, push them out first
  while (!log_entry_queue_.empty()) {
    for (const auto& sink : sinks_) {
      SendToSink(*sink, log_entry_queue_.front());
    }
    log_entry_queue_.pop();
  }

  // ... and now we can log the current log entry
  for (const auto& sink : sinks_) {
    SendToSink(*sink, entry);
  }
}

void TFLogSinks::SendToSink(TFLogSink& sink, const TFLogEntry& entry) {
  sink.Send(entry);
  sink.WaitTillSent();
}

// A class for managing the text file to which VLOG output is written.
// If the environment variable TF_CPP_VLOG_FILENAME is set, all VLOG
// calls are redirected from stderr to a file with corresponding name.
class VlogFileMgr {
 public:
  // Determines if the env variable is set and if necessary
  // opens the file for write access.
  VlogFileMgr();
  // Closes the file.
  ~VlogFileMgr();
  // Returns either a pointer to the file or stderr.
  FILE* FilePtr() const;

 private:
  FILE* vlog_file_ptr;
  char* vlog_file_name;
};

VlogFileMgr::VlogFileMgr() {
  vlog_file_name = getenv("TF_CPP_VLOG_FILENAME");
  vlog_file_ptr =
      vlog_file_name == nullptr ? nullptr : fopen(vlog_file_name, "w");

  if (vlog_file_ptr == nullptr) {
    vlog_file_ptr = stderr;
  }
}

VlogFileMgr::~VlogFileMgr() {
  if (vlog_file_ptr != stderr) {
    fclose(vlog_file_ptr);
  }
}

FILE* VlogFileMgr::FilePtr() const { return vlog_file_ptr; }

int ParseInteger(absl::string_view str) {
  int level;
  if (!absl::SimpleAtoi(str, &level)) {
    return 0;
  }
  return level;
}

// Parse log level (int64) from environment variable (char*)
int64_t LogLevelStrToInt(const char* tf_env_var_val) {
  if (tf_env_var_val == nullptr) {
    return 0;
  }
  return ParseInteger(tf_env_var_val);
}

using VmoduleMap = absl::flat_hash_map<absl::string_view, int>;

// Returns a mapping from module name to VLOG level, derived from the
// TF_CPP_VMODULE environment variable; ownership is transferred to the caller.
VmoduleMap* VmodulesMapFromEnv() {
  // The value of the env var is supposed to be of the form:
  //    "foo=1,bar=2,baz=3"
  const char* env = getenv("TF_CPP_VMODULE");
  if (env == nullptr) {
    // If there is no TF_CPP_VMODULE configuration (most common case), return
    // nullptr so that the ShouldVlogModule() API can fast bail out of it.
    return nullptr;
  }
  // The memory returned by getenv() can be invalidated by following getenv() or
  // setenv() calls. And since we keep references to it in the VmoduleMap in
  // form of StringData objects, make a copy of it.
  const char* env_data = strdup(env);
  absl::string_view env_view(env_data);
  VmoduleMap* result = new VmoduleMap();
  while (!env_view.empty()) {
    size_t eq_pos = env_view.find('=');
    if (eq_pos == absl::string_view::npos) {
      break;
    }
    absl::string_view module_name = env_view.substr(0, eq_pos);
    env_view.remove_prefix(eq_pos + 1);

    // Comma either points at the next comma delimiter, or at a null terminator.
    // We check that the integer we parse ends at this delimiter.
    size_t level_end_pos = env_view.find(',');
    absl::string_view level_str = env_view.substr(0, level_end_pos);
    (*result)[module_name] = ParseInteger(level_str);
    if (level_end_pos != absl::string_view::npos) {
      env_view.remove_prefix(level_end_pos + 1);
    }
  }
  return result;
}

bool EmitThreadIdFromEnv() {
  const char* tf_env_var_val = getenv("TF_CPP_LOG_THREAD_ID");
  return tf_env_var_val == nullptr ? false : ParseInteger(tf_env_var_val) != 0;
}

}  // namespace

absl::LogSeverityAtLeast MinLogLevelFromEnv() {
  // We don't want to print logs during fuzzing as that would slow fuzzing down
  // by almost 2x. So, if we are in fuzzing mode (not just running a test), we
  // return a value so that nothing is actually printed. Since LOG uses >=
  // (see ~LogMessage in this file) to see if log messages need to be printed,
  // the value we're interested on to disable printing is the maximum severity.
  // See also http://llvm.org/docs/LibFuzzer.html#fuzzer-friendly-build-mode
#ifdef FUZZING_BUILD_MODE_UNSAFE_FOR_PRODUCTION
  return absl::LogSeverityAtLeast::kInfinity;
#else
  const char* tf_env_var_val = getenv("TF_CPP_MIN_LOG_LEVEL");
  return static_cast<absl::LogSeverityAtLeast>(
      LogLevelStrToInt(tf_env_var_val));
#endif
}

int MaxVLogLevelFromEnv() {
  // We don't want to print logs during fuzzing as that would slow fuzzing down
  // by almost 2x. So, if we are in fuzzing mode (not just running a test), we
  // return a value so that nothing is actually printed. Since VLOG uses <=
  // (see VLOG_IS_ON in logging.h) to see if log messages need to be printed,
  // the value we're interested on to disable printing is 0.
  // See also http://llvm.org/docs/LibFuzzer.html#fuzzer-friendly-build-mode
#ifdef FUZZING_BUILD_MODE_UNSAFE_FOR_PRODUCTION
  return 0;
#else
  const char* tf_env_var_val = getenv("TF_CPP_MAX_VLOG_LEVEL");
  return LogLevelStrToInt(tf_env_var_val);
#endif
}

LogMessage::LogMessage(const char* fname, int line, absl::LogSeverity severity)
    : fname_(fname), line_(line), severity_(severity) {}

LogMessage& LogMessage::AtLocation(const char* fname, int line) {
  fname_ = fname;
  line_ = line;
  return *this;
}

LogMessage::~LogMessage() {
  // Read the min log level once during the first call to logging.
  static absl::LogSeverityAtLeast min_log_level = MinLogLevelFromEnv();
  if (severity_ >= min_log_level) {
    GenerateLogMessage();
  }
}

void LogMessage::GenerateLogMessage() {
  TFLogSinks::Instance().Send(TFLogEntry(severity_, fname_, line_, str()));
}

int LogMessage::MaxVLogLevel() {
  static int max_vlog_level = MaxVLogLevelFromEnv();
  return max_vlog_level;
}

bool LogMessage::VmoduleActivated(const char* fname, int level) {
  if (level <= MaxVLogLevel()) {
    return true;
  }
  static VmoduleMap* vmodules = VmodulesMapFromEnv();
  if (ABSL_PREDICT_TRUE(vmodules == nullptr)) {
    return false;
  }
  absl::string_view module(fname);
  if (size_t last_slash = module.rfind('/');
      last_slash != absl::string_view::npos) {
    module.remove_prefix(last_slash + 1);
  }
  if (size_t dot_after = module.find('.');
      dot_after != absl::string_view::npos) {
    module.remove_suffix(module.size() - dot_after);
  }
  auto it = vmodules->find(module);
  return it != vmodules->end() && it->second >= level;
}

LogMessageFatal::LogMessageFatal(const char* file, int line)
    : LogMessage(file, line, absl::LogSeverity::kFatal) {}
LogMessageFatal::~LogMessageFatal() {
  // abort() ensures we don't return (we promised we would not via
  // ATTRIBUTE_NORETURN).
  GenerateLogMessage();
  abort();
}

void LogString(const char* fname, int line, absl::LogSeverity severity,
               const string& message) {
  LogMessage(fname, line, severity) << message;
}

template <>
void MakeCheckOpValueString(std::ostream* os, const char& v) {
  if (v >= 32 && v <= 126) {
    (*os) << "'" << v << "'";
  } else {
    (*os) << "char value " << static_cast<int16>(v);
  }
}

template <>
void MakeCheckOpValueString(std::ostream* os, const signed char& v) {
  if (v >= 32 && v <= 126) {
    (*os) << "'" << v << "'";
  } else {
    (*os) << "signed char value " << static_cast<int16>(v);
  }
}

template <>
void MakeCheckOpValueString(std::ostream* os, const unsigned char& v) {
  if (v >= 32 && v <= 126) {
    (*os) << "'" << v << "'";
  } else {
    (*os) << "unsigned char value " << static_cast<uint16>(v);
  }
}

template <>
void MakeCheckOpValueString(std::ostream* os, const std::nullptr_t& v) {
  (*os) << "nullptr";
}

CheckOpMessageBuilder::CheckOpMessageBuilder(const char* exprtext)
    : stream_(new std::ostringstream) {
  *stream_ << "Check failed: " << exprtext << " (";
}

CheckOpMessageBuilder::~CheckOpMessageBuilder() { delete stream_; }

std::ostream* CheckOpMessageBuilder::ForVar2() {
  *stream_ << " vs. ";
  return stream_;
}

string* CheckOpMessageBuilder::NewString() {
  *stream_ << ")";
  return new string(stream_->str());
}

namespace {
// The following code behaves like AtomicStatsCounter::LossyAdd() for
// speed since it is fine to lose occasional updates.
// Returns old value of *counter.
uint32 LossyIncrement(std::atomic<uint32>* counter) {
  const uint32 value = counter->load(std::memory_order_relaxed);
  counter->store(value + 1, std::memory_order_relaxed);
  return value;
}
}  // namespace

bool LogEveryNState::ShouldLog(int n) {
  return n != 0 && (LossyIncrement(&counter_) % n) == 0;
}

bool LogFirstNState::ShouldLog(int n) {
  const int counter_value =
      static_cast<int>(counter_.load(std::memory_order_relaxed));
  if (counter_value < n) {
    counter_.store(counter_value + 1, std::memory_order_relaxed);
    return true;
  }
  return false;
}

bool LogEveryPow2State::ShouldLog(int ignored) {
  const uint32 new_value = LossyIncrement(&counter_) + 1;
  return (new_value & (new_value - 1)) == 0;
}

bool LogEveryNSecState::ShouldLog(double seconds) {
  LossyIncrement(&counter_);
  const int64_t now_cycles = absl::base_internal::CycleClock::Now();
  int64_t next_cycles = next_log_time_cycles_.load(std::memory_order_relaxed);
  do {
    if (now_cycles <= next_cycles) return false;
  } while (!next_log_time_cycles_.compare_exchange_weak(
      next_cycles,
      now_cycles + seconds * absl::base_internal::CycleClock::Frequency(),
      std::memory_order_relaxed, std::memory_order_relaxed));
  return true;
}

}  // namespace internal

void TFAddLogSink(TFLogSink* sink) {
  internal::TFLogSinks::Instance().Add(sink);
}

void TFRemoveLogSink(TFLogSink* sink) {
  internal::TFLogSinks::Instance().Remove(sink);
}

std::vector<TFLogSink*> TFGetLogSinks() {
  return internal::TFLogSinks::Instance().GetSinks();
}

void TFDefaultLogSink::Send(const TFLogEntry& entry) {
#ifdef PLATFORM_POSIX_ANDROID
  int android_log_level;
  switch (entry.log_severity()) {
    case absl::LogSeverity::kInfo:
      android_log_level = ANDROID_LOG_INFO;
      break;
    case absl::LogSeverity::kWarning:
      android_log_level = ANDROID_LOG_WARN;
      break;
    case absl::LogSeverity::kError:
      android_log_level = ANDROID_LOG_ERROR;
      break;
    case absl::LogSeverity::kFatal:
      android_log_level = ANDROID_LOG_FATAL;
      break;
    default:
      if (entry.log_severity() < absl::LogSeverity::kInfo) {
        android_log_level = ANDROID_LOG_VERBOSE;
      } else {
        android_log_level = ANDROID_LOG_ERROR;
      }
      break;
  }

  std::stringstream ss;
  const auto& fname = entry.FName();
  auto pos = fname.find("/");
  ss << (pos != std::string::npos ? fname.substr(pos + 1) : fname) << ":"
     << entry.Line() << " " << entry.ToString();
  __android_log_write(android_log_level, "native", ss.str().c_str());

  // Also log to stderr (for standalone Android apps).
  // Don't use 'std::cerr' since it crashes on Android.
  fprintf(stderr, "native : %s\n", ss.str().c_str());

  // Android logging at level FATAL does not terminate execution, so abort()
  // is still required to stop the program.
  if (entry.log_severity() == absl::LogSeverity::kFatal) {
    abort();
  }
#else   // PLATFORM_POSIX_ANDROID
  static const internal::VlogFileMgr vlog_file;
  static bool log_thread_id = internal::EmitThreadIdFromEnv();
  uint64_t now_micros = EnvTime::NowMicros();
  time_t now_seconds = static_cast<time_t>(now_micros / 1000000);
  int32_t micros_remainder = static_cast<int32_t>(now_micros % 1000000);
  const size_t time_buffer_size = 30;
  char time_buffer[time_buffer_size];
  struct tm* tp;
#if defined(__linux__) || defined(__APPLE__)
  struct tm now_tm;
  tp = localtime_r(&now_seconds, &now_tm);
#else
  tp = localtime(&now_seconds);  // NOLINT(runtime/threadsafe_fn)
#endif
  strftime(time_buffer, time_buffer_size, "%Y-%m-%d %H:%M:%S", tp);
  uint64_t tid = absl::base_internal::GetTID();
  constexpr size_t kTidBufferSize =
      (1 + std::numeric_limits<uint64_t>::digits10 + 1);
  char tid_buffer[kTidBufferSize] = "";
  if (log_thread_id) {
    absl::SNPrintF(tid_buffer, sizeof(tid_buffer), " %7u", tid);
  }

  char sev;
  switch (entry.log_severity()) {
    case absl::LogSeverity::kInfo:
      sev = 'I';
      break;

    case absl::LogSeverity::kWarning:
      sev = 'W';
      break;

    case absl::LogSeverity::kError:
      sev = 'E';
      break;

    case absl::LogSeverity::kFatal:
      sev = 'F';
      break;

    default:
      assert(false && "Unknown logging severity");
      sev = '?';
      break;
  }

  absl::FPrintF(vlog_file.FilePtr(), "%s.%06d: %c%s %s:%d] %s\n", time_buffer,
                micros_remainder, sev, tid_buffer, entry.FName().c_str(),
                entry.Line(), entry.ToString().c_str());
  fflush(vlog_file.FilePtr());  // Ensure logs are written immediately.
#endif  // PLATFORM_POSIX_ANDROID
}

void UpdateLogVerbosityIfDefined(const char* env_var) {}

}  // namespace tsl
