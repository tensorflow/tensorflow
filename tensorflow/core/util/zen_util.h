/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_UTIL_ZEN_UTIL_H_
#define TENSORFLOW_CORE_UTIL_ZEN_UTIL_H_

#ifdef AMD_ZENDNN

#include <sys/types.h>

#include <array>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

namespace tensorflow {

/// Read an integer from the environment variable
/// Return default_value if the environment variable is not defined, otherwise
/// return actual value.
inline int zendnn_getenv_int(const char *name, int default_value = 0) {
  char *val = std::getenv(name);
  return val == NULL ? default_value : atoi(val);
}

#if !defined(LOG_LEVEL_DEFAULT)
#define LOG_LEVEL_DEFAULT LOG_LEVEL_ERROR
#endif

enum LogLevel {
  LOG_LEVEL_DISABLED = -1,
  LOG_LEVEL_ERROR = 0,
  LOG_LEVEL_WARNING = 1,
  LOG_LEVEL_INFO = 2,
  LOG_LEVEL_VERBOSE0 = 3
};

#define LOG_LEVEL_VERBOSE(n) (LOG_LEVEL_VERBOSE0 + n)

static inline const std::string logLevelToStr(int logLevel) {
  if (logLevel == LOG_LEVEL_ERROR) {
    return "E";
  } else if (logLevel == LOG_LEVEL_WARNING) {
    return "W";
  } else if (logLevel == LOG_LEVEL_INFO) {
    return "I";
  } else if (logLevel >= LOG_LEVEL_VERBOSE0) {
    std::stringstream ss;
    ss << "V" << logLevel - LOG_LEVEL_VERBOSE0;
    return ss.str();
  } else {
    return "?";
  }
}

static inline int zendnnGetLogLevel(const std::string &name) {
  static char *logCstr = getenv("ZENDNN_LOG_OPTS");
  if (!logCstr) {
    return LOG_LEVEL_DEFAULT;
  }
  std::string logStr(logCstr);

  size_t pos, epos;

  std::string namePlusColon(name + ":");
  pos = logStr.find(namePlusColon);
  if (pos == std::string::npos) {
    namePlusColon = "ALL:";
    pos = logStr.find(namePlusColon);
  }

  if (pos == std::string::npos) {
    return LOG_LEVEL_DEFAULT;
  }

  epos = pos + namePlusColon.size();
  long x;
  char *ep;
  if (epos >= logStr.size()) {
    assert(epos == logStr.size());
  } else {
    x = strtol(logStr.c_str() + epos, &ep, 0);
    size_t fpos = ep - logStr.c_str();
    if (fpos - epos > 0) {
      return x;
    }
  }

  return LOG_LEVEL_DEFAULT;
}

namespace cn = std::chrono;

enum ZendnnLogModule {
  ZENDNN_ALGOLOG,
  ZENDNN_CORELOG,
  ZENDNN_APILOG,
  ZENDNN_TESTLOG,
  ZENDNN_PROFLOG,
  ZENDNN_FWKLOG,
  ZENDNN_NUM_LOG_MODULES
};

struct ZendnnLogState {
  ZendnnLogState(cn::steady_clock::time_point startTime);
  cn::steady_clock::time_point startTime_;
  std::array<int, ZENDNN_NUM_LOG_MODULES> moduleLevels_;
  std::array<const char *, ZENDNN_NUM_LOG_MODULES> moduleNames_;
  std::ofstream logFIle;
  std::ostream *log;
  // std::ios iosDefaultState;
};

ZendnnLogState::ZendnnLogState(cn::steady_clock::time_point startTime)
    : startTime_(startTime) {
  moduleNames_[ZENDNN_ALGOLOG] = "ALGO";
  moduleNames_[ZENDNN_CORELOG] = "CORE";
  moduleNames_[ZENDNN_APILOG] = "API";
  moduleNames_[ZENDNN_TESTLOG] = "TEST";
  moduleNames_[ZENDNN_PROFLOG] = "PROF";
  moduleNames_[ZENDNN_FWKLOG] = "FWK";

  static_assert(ZENDNN_NUM_LOG_MODULES == 6,
                "Need to update moduleNames_ initialization");

  for (int mod = 0; mod < ZENDNN_NUM_LOG_MODULES; mod++) {
    auto name = moduleNames_.at(mod);
    int lvl = zendnnGetLogLevel(name);
    moduleLevels_.at(mod) = lvl;
  }

  log = &std::cout;
}

ZendnnLogState *_zendnnGetLogState(void) {
  static ZendnnLogState logState(cn::steady_clock::now());
  return &logState;
}

static inline void _zendnnLogMessageR(ZendnnLogState *logState) {
  *logState->log << "\n";
}

template <typename T, typename... Ts>
static inline void _zendnnLogMessageR(ZendnnLogState *logState, T arg0,
                                      Ts... arg1Misc) {
  *logState->log << arg0;
  _zendnnLogMessageR(logState, arg1Misc...);
}

template <typename... Ts>
static inline void _zendnnLogMessage(LogLevel level, ZendnnLogModule mod,
                                     Ts... vs) {
  auto logState = _zendnnGetLogState();
  auto now_t = cn::steady_clock::now();
  auto us =
      cn::duration_cast<cn::microseconds>(now_t - logState->startTime_).count();
  auto moduleName = logState->moduleNames_.at(mod);
  auto logLevelStr = logLevelToStr(level);
  float secs = (float)us / 1000000.0f;

  char logHdr[32];
  std::snprintf(logHdr, sizeof(logHdr), "[%s:%s][%.6f] ", moduleName,
                logLevelStr.c_str(), secs);

  _zendnnLogMessageR(logState, logHdr, vs...);
}

#define zendnnLogAtLevel(mod, level, ...)                       \
  do {                                                          \
    if (level <= _zendnnGetLogState()->moduleLevels_.at(mod)) { \
      _zendnnLogMessage(level, mod, ##__VA_ARGS__);             \
    }                                                           \
  } while (0)

#define zendnnInfo(mod, ...) \
  zendnnLogAtLevel(mod, LOG_LEVEL_INFO, ##__VA_ARGS__)
}  // namespace tensorflow

#endif  // AMD_ZENDNN

#endif
