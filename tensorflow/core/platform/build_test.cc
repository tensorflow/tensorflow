/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/platform/abi.h"
#include "tensorflow/core/platform/byte_order.h"
#include "tensorflow/core/platform/cord.h"
#include "tensorflow/core/platform/cpu_feature_guard.h"
#include "tensorflow/core/platform/cpu_info.h"
#include "tensorflow/core/platform/demangle.h"
#include "tensorflow/core/platform/denormal.h"
#include "tensorflow/core/platform/dynamic_annotations.h"
#include "tensorflow/core/platform/env_time.h"
#include "tensorflow/core/platform/file_statistics.h"
#include "tensorflow/core/platform/fingerprint.h"
#include "tensorflow/core/platform/gif.h"
#include "tensorflow/core/platform/host_info.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/jpeg.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mem.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/net.h"
#include "tensorflow/core/platform/numa.h"
#include "tensorflow/core/platform/numbers.h"
#include "tensorflow/core/platform/platform.h"
#include "tensorflow/core/platform/platform_strings.h"
#include "tensorflow/core/platform/png.h"
#include "tensorflow/core/platform/prefetch.h"
#include "tensorflow/core/platform/protobuf.h"
#if !defined(__ANDROID__)
#include "tensorflow/core/platform/rocm_rocdl_path.h"
#endif
#include "tensorflow/core/platform/scanner.h"
#include "tensorflow/core/platform/setround.h"
#include "tensorflow/core/platform/snappy.h"
#include "tensorflow/core/platform/stacktrace.h"
#include "tensorflow/core/platform/stacktrace_handler.h"
#include "tensorflow/core/platform/str_util.h"
#include "tensorflow/core/platform/strcat.h"
#include "tensorflow/core/platform/stringpiece.h"
#include "tensorflow/core/platform/stringprintf.h"
#include "tensorflow/core/platform/subprocess.h"
#include "tensorflow/core/platform/tensor_coding.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/threadpool_interface.h"
#include "tensorflow/core/platform/threadpool_options.h"
#include "tensorflow/core/platform/tstring.h"
#include "tensorflow/core/platform/types.h"

int main(int argc, char *argv[]) { return 0; }
