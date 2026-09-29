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

#ifndef XLA_TOOLS_MULTIHOST_HLO_RUNNER_HLO_RUNNER_PROFILER_H_
#define XLA_TOOLS_MULTIHOST_HLO_RUNNER_HLO_RUNNER_PROFILER_H_

#include <memory>
#include <string>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/pjrt/distributed/key_value_store_interface.h"
#include "xla/tools/multihost_hlo_runner/profiler_interface.h"
#include "tsl/profiler/lib/profiler_session.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace xla {

// Interface that may optionally return an XSpace proto after UploadSession()
// is called. This can be used by callers to get a programmatic handle of the
// profile data.
class XSpaceProfilerInterface : public ProfilerInterface {
 public:
  virtual const tensorflow::profiler::XSpace* GetXSpace() = 0;
};

// HLORunnerProfiler is a profiler plugin that uses tsl::ProfilerSession to
// profile CPU/GPU execution and allows programmable control of
// profiling sessions for the MultihostHloRunner. It needs to be created after
// PJRT client is initialized. Example usage:
//
//   ABSL_ASSIGN_OR_RETURN(
//       env, xla::GetPjRtEnvironmentForGpu(...));
//   if (env.client != nullptr) {
//     ABSL_ASSIGN_OR_RETURN(auto profiler, HLORunnerProfiler::Create(...));
//   }
//   profiler->CreateSession();
//   ...
//   profiler->UploadSession();
class HLORunnerProfiler : public XSpaceProfilerInterface {
 public:
  // Factory method to create an HLORunnerProfiler with profile result dump
  // path.
  // If keep_xspace is true, the XSpace proto can be retrieved
  // by GetXSpace() after UploadSession() is called, which can be used by
  // callers to get a programmatic handle of the profile data and create XProf.
  static absl::StatusOr<std::unique_ptr<HLORunnerProfiler>> Create(
      absl::string_view dump_path, bool keep_xspace = false,
      bool enable_multipass_profiling = false,
      std::shared_ptr<KeyValueStoreInterface> kv_store = nullptr,
      int task_id = 0, int num_nodes = 1);
  ~HLORunnerProfiler() override;

  // Start a new profiling session.
  void CreateSession() override;

  // Stop the current profiling session.
  void UploadSession() override;

  // Returns the XSpace proto.
  const tensorflow::profiler::XSpace* GetXSpace() override;

 protected:
  explicit HLORunnerProfiler(absl::string_view dump_path, bool keep_xspace);

  void SaveXSpace(std::unique_ptr<tensorflow::profiler::XSpace> xspace);

  // The file path to dump the profiling result.
  std::string dump_path_;
  // Whether to keep the XSpace proto after UploadSession() is called.
  bool keep_xspace_;
  // The XSpace proto to be returned by GetXSpace().
  std::unique_ptr<tensorflow::profiler::XSpace> xspace_;

 private:
  // The profiler session.
  std::unique_ptr<tsl::ProfilerSession> session_;
  // Session counter to uniquely name dump paths when multiple sessions are
  // uploaded.
  int session_index_ = 0;
};

}  // namespace xla

#endif  // XLA_TOOLS_MULTIHOST_HLO_RUNNER_HLO_RUNNER_PROFILER_H_
