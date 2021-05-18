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

#include "tensorflow/core/platform/default/logging.h"

// TODO(b/142492876): Avoid depending on absl internal.
#include "absl/base/internal/cycleclock.h"
#include "absl/base/internal/sysinfo.h"
#include "tensorflow/core/platform/env_time.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"

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

namespace tensorflow {

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

  mutable tensorflow::mutex mutex_;
  std::vector<TFLogSink*> sinks_;
};

TFLogSinks::TFLogSinks() {
#ifndef NO_DEFAULT_LOGGER
  static TFDefaultLogSink* default_sink = new TFDefaultLogSink();
  sinks_.emplace_back(default_sink);
#endif
}

TFLogSinks& TFLogSinks::Instance() {
  static TFLogSinks* instance = new TFLogSinks();
  return *instance;
}

void TFLogSinks::Add(TFLogSink* sink) {
  assert(sink != nullptr && "The sink must not be a nullptr");

  tensorflow::mutex_lock lock(mutex_);
  sinks_.emplace_back(sink);

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

  tensorflow::mutex_lock lock(mutex_);
  auto it = std::find(sinks_.begin(), sinks_.end(), sink);
  if (it != sinks_.end()) sinks_.erase(it);
}

std::vector<TFLogSink*> TFLogSinks::GetSinks() const {
  tensorflow::mutex_lock lock(mutex_);
  return sinks_;
}

void TFLogSinks::Send(const TFLogEntry& entry) {
  tensorflow::mutex_lock lock(mutex_);

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

int ParseInteger(const char* str, size_t size) {
  // Ideally we would use env_var / safe_strto64, but it is
  // hard to use here without pulling in a lot of dependencies,
  // so we use std:istringstream instead
  string integer_str(str, size);
  std::istringstream ss(integer_str);
  int level = 0;
  ss >> level;
  return level;
}

// Parse log level (int64) from environment variable (char*)
int64 LogLevelStrToInt(const char* tf_env_var_val) {
  if (tf_env_var_val == nullptr) {
    return 0;
  }
  return ParseInteger(tf_env_var_val, strlen(tf_env_var_val));
}

// Using StringPiece breaks Windows build.
struct StringData {
  struct Hasher {
    size_t operator()(const StringData& sdata) const {
      // For dependency reasons, we cannot use hash.h here. Use DBJHash instead.
      size_t hash = 5381;
      const char* data = sdata.data;
      for (const char* top = data + sdata.size; data < top; ++data) {
        hash = ((hash << 5) + hash) + (*data);
      }
      return hash;
    }
  };

  StringData() = default;
  StringData(const char* data, size_t size) : data(data), size(size) {}

  bool operator==(const StringData& rhs) const {
    return size == rhs.size && memcmp(data, rhs.data, size) == 0;
  }

  const char* data = nullptr;
  size_t size = 0;
};

using VmoduleMap = std::unordered_map<StringData, int, StringData::Hasher>;

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
  VmoduleMap* result = new VmoduleMap();
  while (true) {
    const char* eq = strchr(env_data, '=');
    if (eq == nullptr) {
      break;
    }
    const char* after_eq = eq + 1;

    // Comma either points at the next comma delimiter, or at a null terminator.
    // We check that the integer we parse ends at this delimiter.
    const char* comma = strchr(after_eq, ',');
    const char* new_env_data;
    if (comma == nullptr) {
      comma = strchr(after_eq, '\0');
      new_env_data = comma;
    } else {
      new_env_data = comma + 1;
    }
    (*result)[StringData(env_data, eq - env_data)] =
        ParseInteger(after_eq, comma - after_eq);
    env_data = new_env_data;
  }
  return result;
}

bool EmitThreadIdFromEnv() {
  const char* tf_env_var_val = getenv("TF_CPP_LOG_THREAD_ID");
  return tf_env_var_val == nullptr
             ? false
             : ParseInteger(tf_env_var_val, strlen(tf_env_var_val)) != 0;
}

}  // namespace

int64 MinLogLevelFromEnv() {
  // We don't want to print logs during fuzzing as that would slow fuzzing down
  // by almost 2x. So, if we are in fuzzing mode (not just running a test), we
  // return a value so that nothing is actually printed. Since LOG uses >=
  // (see ~LogMessage in this file) to see if log messages need to be printed,
  // the value we're interested on to disable printing is the maximum severity.
  // See also http://llvm.org/docs/LibFuzzer.html#fuzzer-friendly-build-mode
#ifdef FUZZING_BUILD_MODE_UNSAFE_FOR_PRODUCTION
  return tensorflow::NUM_SEVERITIES;
#else
  const char* tf_env_var_val = getenv("TF_CPP_MIN_LOG_LEVEL");
  return LogLevelStrToInt(tf_env_var_val);
#endif
}

int64 MaxVLogLevelFromEnv() {
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

LogMessage::LogMessage(const char* fname, int line, int severity)
    : fname_(fname), line_(line), severity_(severity) {}

LogMessage& LogMessage::AtLocation(const char* fname, int line) {
  fname_ = fname;
  line_ = line;
  return *this;
}

LogMessage::~LogMessage() {
  // Read the min log level once during the first call to logging.
  static int64 min_log_level = MinLogLevelFromEnv();
  if (severity_ >= min_log_level) {
    GenerateLogMessage();
  }
}

void LogMessage::GenerateLogMessage() {
  TFLogSinks::Instance().Send(TFLogEntry(severity_, fname_, line_, str()));
}

int64 LogMessage::MaxVLogLevel() {
  static int64 max_vlog_level = MaxVLogLevelFromEnv();
  return max_vlog_level;
}

bool LogMessage::VmoduleActivated(const char* fname, int level) {
  if (level <= MaxVLogLevel()) {
    return true;
  }
  static VmoduleMap* vmodules = VmodulesMapFromEnv();
  if (TF_PREDICT_TRUE(vmodules == nullptr)) {
    return false;
  }
  const char* last_slash = strrchr(fname, '/');
  const char* module_start = last_slash == nullptr ? fname : last_slash + 1;
  const char* dot_after = strchr(module_start, '.');
  const char* module_limit =
      dot_after == nullptr ? strchr(fname, '\0') : dot_after;
  StringData module(module_start, module_limit - module_start);
  auto it = vmodules->find(module);
  return it != vmodules->end() && it->second >= level;
}

LogMessageFatal::LogMessageFatal(const char* file, int line)
    : LogMessage(file, line, FATAL) {}
LogMessageFatal::~LogMessageFatal() {
  // abort() ensures we don't return (we promised we would not via
  // ATTRIBUTE_NORETURN).
  GenerateLogMessage();
  abort();
}

void LogString(const char* fname, int line, int severity,
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

#if LANG_CXX11
template <>
void MakeCheckOpValueString(std::ostream* os, const std::nullptr_t& v) {
  (*os) << "nullptr";
}
#endif

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
  const int64 now_cycles = absl::base_internal::CycleClock::Now();
  int64 next_cycles = next_log_time_cycles_.load(std::memory_order_relaxed);
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
  static bool log_thread_id = internal::EmitThreadIdFromEnv();
  uint64 now_micros = EnvTime::NowMicros();
  time_t now_seconds = static_cast<time_t>(now_micros / 1000000);
  int32 micros_remainder = static_cast<int32>(now_micros % 1000000);
  const size_t time_buffer_size = 30;
  char time_buffer[time_buffer_size];
  strftime(time_buffer, time_buffer_size, "%Y-%m-%d %H:%M:%S",
           localtime(&now_seconds));
  const size_t tid_buffer_size = 10;
  char tid_buffer[tid_buffer_size] = "";
  if (log_thread_id) {
    snprintf(tid_buffer, sizeof(tid_buffer), " %7u",
             absl::base_internal::GetTID());
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

  // TODO(jeff,sanjay): Replace this with something that logs through the env.
  fprintf(stderr, "%s.%06d: %c%s %s:%d] %s\n", time_buffer, micros_remainder,
          sev, tid_buffer, entry.FName().c_str(), entry.Line(),
          entry.ToString().c_str());
#endif  // PLATFORM_POSIX_ANDROID
}

}  // namespace tensorflow
