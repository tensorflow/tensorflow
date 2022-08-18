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

// Exposes the family of BLAS routines as pre-canned high performance calls for
// use in conjunction with the StreamExecutor abstraction.
//
// Note that this interface is optionally supported by platforms; see
// StreamExecutor::SupportsBlas() for details.
//
// This abstraction makes it simple to entrain BLAS operations on GPU data into
// a Stream -- users typically will not use this API directly, but will use the
// Stream builder methods to entrain these operations "under the hood". For
// example:
//
//  DeviceMemory<float> x = stream_exec->AllocateArray<float>(1024);
//  DeviceMemory<float> y = stream_exec->AllocateArray<float>(1024);
//  // ... populate x and y ...
//  Stream stream{stream_exec};
//  stream
//    .Init()
//    .ThenBlasAxpy(1024, 5.5, x, 1, &y, 1);
//  SE_CHECK_OK(stream.BlockHostUntilDone());
//
// By using stream operations in this manner the user can easily intermix custom
// kernel launches (via StreamExecutor::ThenLaunch()) with these pre-canned BLAS
// routines.

#ifndef TENSORFLOW_STREAM_EXECUTOR_BLAS_H_
#define TENSORFLOW_STREAM_EXECUTOR_BLAS_H_

#include "tensorflow/compiler/xla/stream_executor/blas.h"

#endif  // TENSORFLOW_STREAM_EXECUTOR_BLAS_H_
