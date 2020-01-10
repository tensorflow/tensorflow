#ifndef TENSORFLOW_STREAM_EXECUTOR_PLATFORM_MUTEX_H_
#define TENSORFLOW_STREAM_EXECUTOR_PLATFORM_MUTEX_H_

#include "tensorflow/core/platform/port.h"

#if defined(PLATFORM_GOOGLE)
#include "tensorflow/stream_executor/platform/google/mutex.h"
#else
#include "tensorflow/stream_executor/platform/default/mutex.h"
#endif

#endif  // TENSORFLOW_STREAM_EXECUTOR_PLATFORM_MUTEX_H_
