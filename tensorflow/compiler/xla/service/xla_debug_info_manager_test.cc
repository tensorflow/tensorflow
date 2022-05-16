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
#include "tensorflow/compiler/xla/service/xla_debug_info_manager.h"

#include <string>
#include <utility>

#include "tensorflow/compiler/xla/service/hlo.pb.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"

namespace xla {

using ::testing::UnorderedElementsAre;

class XlaDebugInfoManagerTest : public HloTestBase {
 protected:
  struct DebugMetadata {
    // We allow same id to be registered multiple times. we need unique id to
    // know which program is referenced (such as in UnregisterProgram).
    int unique_id;
    std::string id;
    std::shared_ptr<HloModule> module;
    std::shared_ptr<BufferAssignmentProto> buffer_assignment;
  };

  // Return unique id of this module.
  int RegisterProgram(const std::string& module_id) {
    DebugMetadata debug_info;
    HloModuleConfig config;
    debug_info.unique_id = ++serial_;
    debug_info.id = module_id;
    debug_info.module = std::make_shared<HloModule>(module_id, config);
    debug_info.buffer_assignment = nullptr;
    xla_debug_info_manager_.RegisterModule(module_id, debug_info.module,
                                           debug_info.buffer_assignment);
    external_references_.push_back(std::move(debug_info));
    return serial_;
  }

  void UnregisterProgram(int unique_id) {
    for (int i = 0; i < external_references_.size(); i++) {
      if (external_references_[i].unique_id == unique_id) {
        xla_debug_info_manager_.UnregisterModule(
            external_references_[i].id, external_references_[i].module,
            external_references_[i].buffer_assignment);
        external_references_.erase(external_references_.begin() + i);
        break;
      }
    }
  }

  std::set<ModuleIdentifier> GetActiveModule() {
    return xla_debug_info_manager_.GetActiveModules();
  }

  void StartTrace() { xla_debug_info_manager_.StartTracing(); }

  std::set<ModuleIdentifier> StopTrace() {
    std::vector<XlaModuleDebugInfo> module_debug_info;
    xla_debug_info_manager_.StopTracing(&module_debug_info);
    std::set<ModuleIdentifier> serialized;
    for (const auto& module : module_debug_info) {
      serialized.insert(module.module_id);
    }
    return serialized;
  }

  int serial_ = 0;

  // Simulation of compilation cache.
  std::vector<DebugMetadata> external_references_;

  // Use an instance per test instead of singleton to avoid interferences.
  XlaDebugInfoManager xla_debug_info_manager_;
};

// Test the cases where no trace session is involved.
TEST_F(XlaDebugInfoManagerTest, NoTraceBasic) {
  auto program0 = RegisterProgram("program0");
  EXPECT_THAT(GetActiveModule(), UnorderedElementsAre("program0"));

  auto program1 = RegisterProgram("program1");
  EXPECT_THAT(GetActiveModule(), UnorderedElementsAre("program0", "program1"));

  UnregisterProgram(program0);
  EXPECT_THAT(GetActiveModule(), UnorderedElementsAre("program1"));
  UnregisterProgram(program1);
  EXPECT_TRUE(GetActiveModule().empty());
}

TEST_F(XlaDebugInfoManagerTest, NoTraceDuplicateIds) {
  auto program0A = RegisterProgram("program0");
  auto program0B = RegisterProgram("program0");  // duplicates
  auto program1 = RegisterProgram("program1");
  EXPECT_THAT(GetActiveModule(), UnorderedElementsAre("program0", "program1"));

  UnregisterProgram(program1);
  EXPECT_THAT(GetActiveModule(), UnorderedElementsAre("program0"));
  UnregisterProgram(program0A);
  EXPECT_THAT(GetActiveModule(), UnorderedElementsAre("program0"));
  UnregisterProgram(program0B);
  EXPECT_TRUE(GetActiveModule().empty());
}

// Test the cases where an active trace session is involved.
TEST_F(XlaDebugInfoManagerTest, ActiveTrace) {
  auto program0A = RegisterProgram("program0");
  auto program0B = RegisterProgram("program0");  // duplicates
  auto program1 = RegisterProgram("program1");

  // Case 1: Trace starts when no program is running.
  StartTrace();
  auto program2 = RegisterProgram("program2");
  EXPECT_THAT(StopTrace(),
              UnorderedElementsAre("program0", "program1", "program2"));

  // Case 1: Trace starts during program is running.
  StartTrace();
  EXPECT_THAT(StopTrace(),
              UnorderedElementsAre("program0", "program1", "program2"));

  UnregisterProgram(program2);
  EXPECT_THAT(GetActiveModule(), UnorderedElementsAre("program0", "program1"));
  UnregisterProgram(program0A);
  EXPECT_THAT(GetActiveModule(), UnorderedElementsAre("program0", "program1"));
  UnregisterProgram(program0B);
  EXPECT_THAT(GetActiveModule(), UnorderedElementsAre("program1"));
  UnregisterProgram(program1);
  EXPECT_TRUE(GetActiveModule().empty());
}

TEST_F(XlaDebugInfoManagerTest, UnregisterDuringTrace) {
  auto program0A = RegisterProgram("program0");
  auto program0B = RegisterProgram("program0");  // duplicates
  auto program1 = RegisterProgram("program1");

  StartTrace();
  UnregisterProgram(program1);
  UnregisterProgram(program0B);
  EXPECT_THAT(StopTrace(), UnorderedElementsAre("program0", "program1"));
  EXPECT_THAT(GetActiveModule(), UnorderedElementsAre("program0"));

  UnregisterProgram(program0A);
}

}  // namespace xla
