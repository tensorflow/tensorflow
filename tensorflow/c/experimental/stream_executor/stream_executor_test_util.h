/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_C_EXPERIMENTAL_STREAM_EXECUTOR_STREAM_EXECUTOR_TEST_UTIL_H_
#define TENSORFLOW_C_EXPERIMENTAL_STREAM_EXECUTOR_STREAM_EXECUTOR_TEST_UTIL_H_

#include "tensorflow/c/experimental/stream_executor/stream_executor.h"

struct SP_Stream_st {
  explicit SP_Stream_st(int id) : stream_id(id) {}
  int stream_id;
};

struct SP_Event_st {
  explicit SP_Event_st(int id) : event_id(id) {}
  int event_id;
};

struct SP_Timer_st {
  explicit SP_Timer_st(int id) : timer_id(id) {}
  int timer_id;
};

namespace stream_executor {
namespace test_util {

constexpr int kDeviceCount = 2;
constexpr char kDeviceName[] = "MY_DEVICE";
constexpr char kDeviceType[] = "GPU";

void PopulateDefaultStreamExecutor(SP_StreamExecutor* se);
void PopulateDefaultDeviceFns(SP_DeviceFns* device_fns);
void PopulateDefaultTimerFns(SP_TimerFns* timer_fns);
void PopulateDefaultPlatform(SP_Platform* platform,
                             SP_PlatformFns* platform_fns);
void PopulateDefaultPlatformRegistrationParams(
    SE_PlatformRegistrationParams* params);

void DestroyPlatform(SP_Platform* platform);
void DestroyPlatformFns(SP_PlatformFns* platform_fns);

}  // namespace test_util
}  // namespace stream_executor

#endif  // TENSORFLOW_C_EXPERIMENTAL_STREAM_EXECUTOR_STREAM_EXECUTOR_TEST_UTIL_H_
