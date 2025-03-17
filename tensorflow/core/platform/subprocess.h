/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_PLATFORM_SUBPROCESS_H_
#define TENSORFLOW_CORE_PLATFORM_SUBPROCESS_H_

#include "xla/tsl/platform/subprocess.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
using tsl::ACTION_CLOSE;
using tsl::ACTION_DUPPARENT;
using tsl::ACTION_PIPE;
using tsl::CHAN_STDERR;
using tsl::CHAN_STDIN;
using tsl::CHAN_STDOUT;
using tsl::Channel;
using tsl::ChannelAction;
using tsl::CreateSubProcess;
using tsl::SubProcess;
}  // namespace tensorflow

#include "tensorflow/core/platform/platform.h"


#endif  // TENSORFLOW_CORE_PLATFORM_SUBPROCESS_H_
