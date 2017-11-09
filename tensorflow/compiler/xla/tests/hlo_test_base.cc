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

#include "tensorflow/compiler/xla/tests/hlo_test_base.h"

#include <set>
#include <string>
#include <utility>

#include "tensorflow/compiler/xla/legacy_flags/debug_options_flags.h"
#include "tensorflow/compiler/xla/ptr_util.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/types.h"

namespace se = ::perftools::gputools;

namespace xla {

/* static */
std::unique_ptr<HloModule> HloTestBase::CreateNewModule() {
  HloModuleConfig config;

  auto debug_options = legacy_flags::GetDebugOptionsFromFlags();
  // TODO(b/38354253): Change tests to use Parameters instead of Constants.
  debug_options.add_xla_disable_hlo_passes("constant_folding");

  config.set_debug_options(debug_options);

  return MakeUnique<HloModule>(TestName(), VersionedComputationHandle(),
                               config);
}

StatusOr<perftools::gputools::DeviceMemoryBase> HloTestBase::Execute(
    std::unique_ptr<HloModule> module,
    tensorflow::gtl::ArraySlice<perftools::gputools::DeviceMemoryBase>
        arguments,
    Shape* result_shape) {
  return runner_.Execute(std::move(module), arguments, result_shape);
}

se::DeviceMemoryBase HloTestBase::TransferToDevice(const Literal& literal) {
  return runner_.TransferToDevice(literal).ValueOrDie();
}

std::unique_ptr<Literal> HloTestBase::TransferFromDevice(
    const Shape& shape, se::DeviceMemoryBase device_base) {
  return runner_.TransferFromDevice(shape, device_base).ValueOrDie();
}

std::unique_ptr<Literal> HloTestBase::ExecuteAndTransfer(
    std::unique_ptr<HloModule> module,
    tensorflow::gtl::ArraySlice<se::DeviceMemoryBase> arguments) {
  return runner_.ExecuteAndTransfer(std::move(module), arguments).ValueOrDie();
}

Backend& HloTestBase::backend() { return runner_.backend(); }

/* static */
string HloTestBase::TestName() {
  return ::testing::UnitTest::GetInstance()->current_test_info()->name();
}

}  // namespace xla
