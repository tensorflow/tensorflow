/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/service/llvm_ir/error_handler.h"

#include <cstdint>
#include <memory>
#include <utility>

#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/barrier.h"
#include "llvm/Support/ErrorHandling.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/platform/threadpool.h"

namespace xla {
namespace {

TEST(XlaScopedFatalErrorHandlerTest, MultiThreadedFatalError) {
  EXPECT_DEATH(
      {
        constexpr int32_t kNumThreads = 10;
        tsl::thread::ThreadPool pool(tsl::Env::Default(), "test", kNumThreads);
        absl::Barrier barrier(kNumThreads);

        for (int i = 0; i < kNumThreads; ++i) {
          pool.Schedule([i, &barrier]() {
            const auto handler = [i](absl::string_view reason) {
              LOG(ERROR) << "Handler called for thread " << i << " with reason "
                         << reason;
            };
            XlaScopedFatalErrorHandler guard(handler);
            // Ensure all threads have installed the handler before reporting
            // the error.
            barrier.Block();
            if (i == 4) {
              llvm::report_fatal_error("test error");
            }
          });
        }
      },
      "Handler called for thread 4 with reason test error");
}

TEST(XlaScopedFatalErrorHandlerTest, MultiThreadedFatalErrorDefaultHandler) {
  EXPECT_DEATH(
      {
        constexpr int32_t kNumThreads = 10;
        tsl::thread::ThreadPool pool(tsl::Env::Default(), "test", kNumThreads);
        absl::Barrier barrier(kNumThreads);

        for (int i = 0; i < kNumThreads; ++i) {
          pool.Schedule([i, &barrier]() {
            const auto handler = [i](absl::string_view reason) {
              LOG(ERROR) << "Handler called for thread " << i << " with reason "
                         << reason;
            };
            auto guard = i == 4 ? XlaScopedFatalErrorHandler(nullptr)
                                : XlaScopedFatalErrorHandler(handler);
            // Ensure all threads have installed the handler before reporting
            // the error.
            barrier.Block();
            if (i == 4) {
              llvm::report_fatal_error("test error");
            }
          });
        }
      },
      "LLVM ERROR: test error");
}

TEST(XlaScopedFatalErrorHandlerTest, MultiThreadedFatalErrorComplexObject) {
  EXPECT_DEATH(
      {
        constexpr int32_t kNumThreads = 10;
        tsl::thread::ThreadPool pool(tsl::Env::Default(), "test", kNumThreads);
        absl::Barrier barrier(kNumThreads);
        for (int i = 0; i < kNumThreads; ++i) {
          pool.Schedule([i, &barrier]() {
            auto i_ptr = std::make_unique<int32_t>(i);
            auto handler = [i_ptr =
                                std::move(i_ptr)](absl::string_view reason) {
              LOG(ERROR) << "Handler called for thread " << *i_ptr
                         << " with complex object and reason " << reason;
            };
            XlaScopedFatalErrorHandler guard(std::move(handler));
            // Ensure all threads have installed the handler before reporting
            // the error.
            barrier.Block();
            if (i == 4) {
              llvm::report_fatal_error("test error");
            }
          });
        }
      },
      "Handler called for thread 4 with complex object and reason test error");
}

TEST(XlaScopedFatalErrorHandlerTest, NestedFatalError) {
  EXPECT_DEATH(
      {
        const auto handler1 = [](absl::string_view reason) {
          LOG(ERROR) << "Handler1: " << reason;
        };
        XlaScopedFatalErrorHandler guard1(handler1);
        {
          const auto handler2 = [](absl::string_view reason) {
            LOG(ERROR) << "Handler2:" << reason;
          };
          XlaScopedFatalErrorHandler guard2(handler2);
        }  // guard2 is destroyed here
        llvm::report_fatal_error("test error");
      },
      "Handler1: test error");
}

TEST(XlaScopedFatalErrorHandlerTest, TestRestorationAfterScopeExit) {
  EXPECT_DEATH(
      {
        const auto handler1 = [](absl::string_view reason) {
          LOG(ERROR) << "Handler1: " << reason;
        };
        {
          XlaScopedFatalErrorHandler guard1(handler1);
        }  // guard1 is destroyed here
        llvm::report_fatal_error("test error");
      },
      "LLVM ERROR: test error");  // Expect default handler to be called.
}

}  // namespace
}  // namespace xla
