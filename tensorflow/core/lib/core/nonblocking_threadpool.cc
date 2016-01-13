// Copyright 2016 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//==============================================================================

#include "tensorflow/core/lib/core/nonblocking_threadpool.h"
#include "tensorflow/core/platform/mutex.h"

namespace tensorflow {
namespace thread {

thread_local std::unique_ptr<
    typename NonBlockingThreadPool::ThreadSpecificInstanceInfo::PerThreadInfo>
    NonBlockingThreadPool::ThreadSpecificInstanceInfo::static_info(
        new NonBlockingThreadPool::ThreadSpecificInstanceInfo::PerThreadInfo());

mutex NonBlockingThreadPool::ThreadSpecificInstanceInfo::free_index_lock;

std::deque<size_t>
    NonBlockingThreadPool::ThreadSpecificInstanceInfo::free_indexes;

uint32_t NonBlockingThreadPool::ThreadSpecificInstanceInfo::next_index = 0;

}  // namespace thread
}  // namespace tensorflow
