#ifndef TENSORFLOW_STREAM_EXECUTOR_PLATFORM_LOGGING_H_
#define TENSORFLOW_STREAM_EXECUTOR_PLATFORM_LOGGING_H_

#include "tensorflow/core/platform/logging.h"
#include "tensorflow/stream_executor/platform/port.h"

#if !defined(PLATFORM_GOOGLE)

// A CHECK() macro that lets you assert the success of a function that
// returns -1 and sets errno in case of an error. E.g.
//
// CHECK_ERR(mkdir(path, 0700));
//
// or
//
// int fd = open(filename, flags); CHECK_ERR(fd) << ": open " << filename;
#define CHECK_ERR(invocation) CHECK((invocation) != -1) << #invocation

#endif

#endif  // TENSORFLOW_STREAM_EXECUTOR_PLATFORM_LOGGING_H_
