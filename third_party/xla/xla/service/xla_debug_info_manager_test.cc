/* Copyright 2019 The OpenXLA Authors.

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
#include "xla/service/xla_debug_info_manager.h"

#include <memory>
#include <string>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_set.h"
#include "absl/synchronization/mutex.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/hlo_module_config.h"

namespace xla {

class XlaDebugInfoManagerTestPeer {
 public:
  void RegisterModule(std::shared_ptr<const HloModule> hlo_module,
                      BufferAssignmentProto buffer_assignment) {
    return xla_debug_info_manager_.RegisterModule(hlo_module,
                                                  std::move(buffer_assignment));
  }

  void UnregisterModule(ModuleIdentifier module_id) {
    return xla_debug_info_manager_.UnregisterModule(module_id);
  }

  void StartTracing() { return xla_debug_info_manager_.StartTracing(); }

  absl::flat_hash_set<ModuleIdentifier> StopTracing() {
    std::vector<std::unique_ptr<HloProto>> module_debug_info;
    xla_debug_info_manager_.StopTracing(&module_debug_info);
    absl::flat_hash_set<ModuleIdentifier> module_ids;
    for (const auto& hlo_proto : module_debug_info) {
      module_ids.insert(hlo_proto->hlo_module().id());
    }
    return module_ids;
  }

  absl::flat_hash_set<ModuleIdentifier> GetModuleIds() {
    absl::flat_hash_set<ModuleIdentifier> module_ids;
    absl::MutexLock lock(&xla_debug_info_manager_.mutex_);
    for (const auto& it : xla_debug_info_manager_.modules_) {
      module_ids.insert(it.first);
    }
    return module_ids;
  }

 private:
  XlaDebugInfoManager xla_debug_info_manager_;
};

namespace {

using ::testing::IsEmpty;
using ::testing::UnorderedElementsAre;

class XlaDebugInfoManagerTest : public HloHardwareIndependentTestBase {
 protected:
  struct DebugMetadata {
    // We allow same id to be registered multiple times. we need unique id to
    // know which program is referenced (such as in UnregisterProgram).
    ModuleIdentifier unique_id;
    std::shared_ptr<HloModule> module;
  };

  // Return unique id of this module.
  ModuleIdentifier RegisterProgram(const std::string& module_name) {
    DebugMetadata debug_info;
    HloModuleConfig config;
    debug_info.module = std::make_shared<HloModule>(module_name, config);
    ModuleIdentifier unique_id = debug_info.module->unique_id();
    debug_info.unique_id = unique_id;
    xla_debug_info_manager_.RegisterModule(debug_info.module,
                                           BufferAssignmentProto());
    external_references_.push_back(std::move(debug_info));
    return unique_id;
  }

  void UnregisterProgram(ModuleIdentifier unique_id) {
    for (int i = 0; i < external_references_.size(); i++) {
      if (external_references_[i].unique_id == unique_id) {
        xla_debug_info_manager_.UnregisterModule(unique_id);
        external_references_.erase(external_references_.begin() + i);
        break;
      }
    }
  }

  absl::flat_hash_set<ModuleIdentifier> GetModuleIds() {
    return xla_debug_info_manager_.GetModuleIds();
  }

  void StartTrace() { xla_debug_info_manager_.StartTracing(); }

  absl::flat_hash_set<ModuleIdentifier> StopTrace() {
    return xla_debug_info_manager_.StopTracing();
  }

  // Simulation of compilation cache.
  std::vector<DebugMetadata> external_references_;

  // Use an instance per test instead of singleton to avoid interferences.
  XlaDebugInfoManagerTestPeer xla_debug_info_manager_;
};

// Test the cases where no trace session is involved.
TEST_F(XlaDebugInfoManagerTest, NoTraceBasic) {
  auto program0 = RegisterProgram("program0");
  EXPECT_THAT(GetModuleIds(), UnorderedElementsAre(program0));

  auto program1 = RegisterProgram("program1");
  EXPECT_THAT(GetModuleIds(), UnorderedElementsAre(program0, program1));

  UnregisterProgram(program0);
  EXPECT_THAT(GetModuleIds(), UnorderedElementsAre(program1));
  UnregisterProgram(program1);
  EXPECT_TRUE(GetModuleIds().empty());
}

TEST_F(XlaDebugInfoManagerTest, NoTraceDuplicateIds) {
  auto program0A = RegisterProgram("program0");
  auto program0B = RegisterProgram("program0");  // duplicates
  auto program1 = RegisterProgram("program1");
  EXPECT_THAT(GetModuleIds(),
              UnorderedElementsAre(program0A, program0B, program1));

  UnregisterProgram(program1);
  EXPECT_THAT(GetModuleIds(), UnorderedElementsAre(program0A, program0B));
  UnregisterProgram(program0A);
  EXPECT_THAT(GetModuleIds(), UnorderedElementsAre(program0B));
  UnregisterProgram(program0B);
  EXPECT_THAT(GetModuleIds(), IsEmpty());
}

// Test the cases where an active trace session is involved.
TEST_F(XlaDebugInfoManagerTest, ActiveTrace) {
  auto program0A = RegisterProgram("program0");
  auto program0B = RegisterProgram("program0");  // duplicates
  auto program1 = RegisterProgram("program1");

  StartTrace();
  auto program2 = RegisterProgram("program2");
  EXPECT_THAT(StopTrace(),
              UnorderedElementsAre(program0A, program0B, program1, program2));

  StartTrace();
  EXPECT_THAT(StopTrace(),
              UnorderedElementsAre(program0A, program0B, program1, program2));

  UnregisterProgram(program2);
  EXPECT_THAT(GetModuleIds(),
              UnorderedElementsAre(program0A, program0B, program1));
  UnregisterProgram(program0A);
  EXPECT_THAT(GetModuleIds(), UnorderedElementsAre(program0B, program1));
  UnregisterProgram(program0B);
  EXPECT_THAT(GetModuleIds(), UnorderedElementsAre(program1));
  UnregisterProgram(program1);
  EXPECT_THAT(GetModuleIds(), IsEmpty());
}

TEST_F(XlaDebugInfoManagerTest, UnregisterDuringTrace) {
  auto program0A = RegisterProgram("program0");
  auto program0B = RegisterProgram("program0");  // duplicates
  auto program1 = RegisterProgram("program1");

  StartTrace();
  UnregisterProgram(program1);
  UnregisterProgram(program0B);
  EXPECT_THAT(StopTrace(),
              UnorderedElementsAre(program0A, program0B, program1));
  EXPECT_THAT(GetModuleIds(), UnorderedElementsAre(program0A));

  UnregisterProgram(program0A);
}

}  // namespace
}  // namespace xla
