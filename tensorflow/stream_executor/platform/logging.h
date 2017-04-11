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
