/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/hlo/ir/hlo_module_metadata.h"

#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/test_helpers.h"

namespace xla {
namespace {

using ::testing::ElementsAre;
using ::testing::Property;
using ::testing::StrEq;

class TestEnv : public tsl::EnvWrapper {
 public:
  TestEnv() : EnvWrapper(Env::Default()) {}

  uint64_t NowMicros() const override { return current_micros_; }

  void SetCurrentMicros(uint64_t micros) { current_micros_ = micros; }

 private:
  uint64_t current_micros_ = 1;
};

TEST(HloModuleMetadata, RecordsPassStart) {
  TestEnv env;
  HloModuleMetadata module_metadata(&env);
  env.SetCurrentMicros(1234);
  module_metadata.RecordPassStart();
  EXPECT_THAT(
      module_metadata.proto().pass_metadata(),
      ElementsAre(Property(&HloPassMetadata::start_timestamp_usec, 1234)));
}

TEST(HloModuleMetadata, RecordsPassEnd) {
  TestEnv env;
  HloModuleMetadata module_metadata(&env);
  module_metadata.RecordPassStart();
  env.SetCurrentMicros(4321);
  EXPECT_IS_OK(module_metadata.RecordPassEnd());
  EXPECT_THAT(
      module_metadata.proto().pass_metadata(),
      ElementsAre(Property(&HloPassMetadata::end_timestamp_usec, 4321)));
}

TEST(HloModuleMetadata, RecordsPassEndInNestedMetadata) {
  TestEnv env;
  HloModuleMetadata module_metadata(&env);
  module_metadata.RecordPassStart();
  module_metadata.RecordPassStart();
  env.SetCurrentMicros(111);
  EXPECT_IS_OK(module_metadata.RecordPassEnd());
  EXPECT_THAT(module_metadata.proto().pass_metadata(),
              ElementsAre(Property(&HloPassMetadata::end_timestamp_usec, 0),
                          Property(&HloPassMetadata::end_timestamp_usec, 111)));

  env.SetCurrentMicros(222);
  EXPECT_IS_OK(module_metadata.RecordPassEnd());
  EXPECT_THAT(module_metadata.proto().pass_metadata(),
              ElementsAre(Property(&HloPassMetadata::end_timestamp_usec, 222),
                          Property(&HloPassMetadata::end_timestamp_usec, 111)));
}

TEST(HloModuleMetadata, RecordPassEndReturnsNotFound) {
  HloModuleMetadata module_metadata(tsl::Env::Default());
  EXPECT_EQ(module_metadata.RecordPassEnd().code(), tsl::error::NOT_FOUND);

  module_metadata.RecordPassStart();
  EXPECT_IS_OK(module_metadata.RecordPassEnd());
  EXPECT_EQ(module_metadata.RecordPassEnd().code(), tsl::error::NOT_FOUND);
}

TEST(HloModuleMetadata, SetsHloPassMetadataFields) {
  HloModuleMetadata module_metadata(tsl::Env::Default());
  module_metadata.RecordPassStart();
  EXPECT_IS_OK(module_metadata.set_current_pass_name("fake name"));
  EXPECT_THAT(
      module_metadata.proto().pass_metadata(),
      ElementsAre(Property(&HloPassMetadata::pass_name, StrEq("fake name"))));
}

TEST(HloModuleMetadata, SetsHloPassMetadataFieldsInNestedMetadata) {
  HloModuleMetadata module_metadata(tsl::Env::Default());
  module_metadata.RecordPassStart();
  module_metadata.RecordPassStart();
  EXPECT_IS_OK(module_metadata.set_current_pass_name("fake name"));
  EXPECT_THAT(
      module_metadata.proto().pass_metadata(),
      ElementsAre(Property(&HloPassMetadata::pass_name, StrEq("")),
                  Property(&HloPassMetadata::pass_name, StrEq("fake name"))));
}

TEST(HloModuleMetadata, SetterReturnsNotFound) {
  HloModuleMetadata module_metadata(tsl::Env::Default());
  EXPECT_EQ(module_metadata.set_current_pass_name("fake name").code(),
            tsl::error::NOT_FOUND);
}

TEST(HloModuleMetadata, CopiesRunningPrepartitioningPasses) {
  HloModuleMetadata old_module_metadata(tsl::Env::Default());
  old_module_metadata.RecordPassStart();
  EXPECT_IS_OK(old_module_metadata.set_current_pass_name("outer pass"));

  old_module_metadata.RecordPassStart();
  EXPECT_IS_OK(old_module_metadata.set_current_pass_name("finished pass"));
  EXPECT_IS_OK(old_module_metadata.RecordPassEnd());

  old_module_metadata.RecordPassStart();
  EXPECT_IS_OK(old_module_metadata.set_current_pass_name("inner pass"));

  HloModuleMetadata new_module_metadata(tsl::Env::Default());
  new_module_metadata.set_prepartitioning_metadata(old_module_metadata);

  // Passes that are still running go in the new module.
  EXPECT_THAT(
      new_module_metadata.proto().pass_metadata(),
      ElementsAre(Property(&HloPassMetadata::pass_name, StrEq("outer pass")),
                  Property(&HloPassMetadata::pass_name, StrEq("inner pass"))));

  // Passes that finished go in the prepartitioning metadata.
  EXPECT_THAT(new_module_metadata.prepartitioning_metadata()->pass_metadata(),
              ElementsAre(Property(&HloPassMetadata::pass_name,
                                   StrEq("finished pass"))));
}

}  // namespace
}  // namespace xla
