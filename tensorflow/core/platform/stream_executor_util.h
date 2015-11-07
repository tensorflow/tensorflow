#ifndef TENSORFLOW_PLATFORM_STREAM_EXECUTOR_UTIL_H_
#define TENSORFLOW_PLATFORM_STREAM_EXECUTOR_UTIL_H_

#include "tensorflow/core/platform/port.h"

#if defined(PLATFORM_GOOGLE)
#include "tensorflow/core/platform/google/stream_executor_util.h"
#else
#include "tensorflow/core/platform/default/stream_executor_util.h"
#endif

#endif  // TENSORFLOW_PLATFORM_STREAM_EXECUTOR_UTIL_H_
