/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_PLATFORM_CLOUD_FILE_BLOCK_CACHE_H_
#define TENSORFLOW_CORE_PLATFORM_CLOUD_FILE_BLOCK_CACHE_H_

#include <functional>
#include <list>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "xla/tsl/platform/cloud/file_block_cache.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/notification.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/stringpiece.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
// NOLINTBEGIN(misc-unused-using-decls)
using tsl::FileBlockCache;
using tsl::FileBlockCacheStatsInterface;
// NOLINTEND(misc-unused-using-decls)
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PLATFORM_CLOUD_FILE_BLOCK_CACHE_H_
