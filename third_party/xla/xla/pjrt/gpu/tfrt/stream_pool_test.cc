/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/pjrt/gpu/tfrt/stream_pool.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/service/backend.h"
#include "xla/stream_executor/platform_manager.h"

namespace xla {
namespace {

TEST(StreamPoolTest, Borrow) {
  auto platform = xla::se::PlatformManager::PlatformWithName("CUDA").value();
  xla::BackendOptions backend_options;
  backend_options.set_platform(platform);
  auto backend = xla::Backend::CreateBackend(backend_options).value();
  const int device_ordinal = 0;
  auto executor = backend->stream_executor(device_ordinal).value();
  BoundedStreamPool pool(executor, 1);
  ASSERT_OK_AND_ASSIGN(BoundedStreamPool::Handle handle, pool.Borrow());
  EXPECT_NE(handle.get(), nullptr);
  EXPECT_EQ(handle.get(), &*handle);
}

}  // namespace

}  // namespace xla
