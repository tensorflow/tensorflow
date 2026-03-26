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
#include "xla/service/file_backed_metrics_hook.h"

#include <string>

#include "absl/status/status_matchers.h"
#include "absl/time/time.h"
#include "xla/service/metrics.pb.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/testing/temporary_directory.h"
#include "tsl/platform/path.h"

namespace xla {
namespace {

TEST(FileBackedMetricsHookTest, RecordsAndSerializes) {
  TF_ASSERT_OK_AND_ASSIGN(
      tsl::testing::TemporaryDirectory temp_dir,
      tsl::testing::TemporaryDirectory::CreateForCurrentTestcase());
  std::string filepath = tsl::io::JoinPath(temp_dir.path(), "metrics.txt");
  FileBackedMetricsHook hook(filepath);

  hook.RecordCompilationMetrics(CompilationLogEntry::HLO_PASSES,
                                absl::Milliseconds(1500), "test_module");
  EXPECT_OK(hook.DumpMetrics());

  CompilationLogs logs;
  EXPECT_THAT((tsl::ReadTextProto(tsl::Env::Default(), filepath, &logs)),
              absl_testing::IsOk());
  EXPECT_EQ(logs.entries_size(), 1);
  CompilationLogEntry entry = logs.entries(0);
  EXPECT_EQ(entry.stage(), CompilationLogEntry::HLO_PASSES);
  EXPECT_EQ(entry.duration().seconds(), 1);
  EXPECT_EQ(entry.hlo_module_name(), "test_module");
}

}  // namespace
}  // namespace xla
