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

#ifndef TENSORFLOW_COMPILER_XLA_TESTS_HLO_TEST_BASE_H_
#define TENSORFLOW_COMPILER_XLA_TESTS_HLO_TEST_BASE_H_

#include <memory>
#include <string>
#include <vector>

#include "tensorflow/compiler/xla/service/backend.h"
#include "tensorflow/compiler/xla/service/compiler.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_module_config.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"
#include "tensorflow/core/platform/test.h"

namespace xla {

// A base class for tests which build and run HLO code. This is a lower level of
// abstraction than using the client interface and enables, for one, explicitly
// building a graph of HLO instructions to run.
class HloTestBase : public ::testing::Test {
 protected:
  struct EigenThreadPoolWrapper;
  HloTestBase();

  ~HloTestBase() override;

  // Executes the given module and returns a global data handle.
  StatusOr<perftools::gputools::DeviceMemoryBase> Execute(
      std::unique_ptr<HloModule> module,
      tensorflow::gtl::ArraySlice<perftools::gputools::DeviceMemoryBase>
          arguments,
      Shape* result_shape);

  // Variation of Execute which takes a custom module_config instead of creating
  // a default one.
  StatusOr<perftools::gputools::DeviceMemoryBase> Execute(
      std::unique_ptr<HloModule> module,
      std::unique_ptr<HloModuleConfig> module_config,
      tensorflow::gtl::ArraySlice<perftools::gputools::DeviceMemoryBase>
          arguments,
      Shape* result_shape);

  // Transfers the given literal to the device and returns the data handle.
  perftools::gputools::DeviceMemoryBase TransferToDevice(
      const Literal& literal);

  // Transfers the array refered to by the given handle from the device and
  // returns as a Literal.
  std::unique_ptr<Literal> TransferFromDevice(
      const Shape& shape, perftools::gputools::DeviceMemoryBase device_base);

  // Executes the given module and return the result as a Literal.
  std::unique_ptr<Literal> ExecuteAndTransfer(
      std::unique_ptr<HloModule> module,
      tensorflow::gtl::ArraySlice<perftools::gputools::DeviceMemoryBase>
          arguments);

  // Variation of ExecuteAndTransfer which takes a custom module_config instead
  // of creating a default one.
  std::unique_ptr<Literal> ExecuteAndTransfer(
      std::unique_ptr<HloModule> module,
      std::unique_ptr<HloModuleConfig> module_config,
      tensorflow::gtl::ArraySlice<perftools::gputools::DeviceMemoryBase>
          arguments);

  // Helpers for comparing ordered and unordered equality of HloInstruction
  // containers.
  void ExpectEqOrdered(
      tensorflow::gtl::ArraySlice<const HloInstruction*> actual,
      tensorflow::gtl::ArraySlice<const HloInstruction*> expected) {
    std::vector<const HloInstruction*> expected_vec(expected.begin(),
                                                    expected.end());
    std::vector<const HloInstruction*> actual_vec(actual.begin(), actual.end());
    EXPECT_TRUE(testing::VectorMatcher<const HloInstruction*>(expected_vec)(
        actual_vec));
  }

  void ExpectEqUnordered(
      tensorflow::gtl::ArraySlice<const HloInstruction*> actual,
      tensorflow::gtl::ArraySlice<const HloInstruction*> expected) {
    std::vector<const HloInstruction*> expected_vec(expected.begin(),
                                                    expected.end());
    std::vector<const HloInstruction*> actual_vec(actual.begin(), actual.end());
    EXPECT_TRUE(testing::UnorderedElementsAre<const HloInstruction*>(
        expected_vec)(actual_vec));
  }

  string TestName() const;

  std::unique_ptr<Backend> backend_;

  Compiler::HloDumper test_hlo_dumper_;

  // This vector contains handles of all the device memory allocations performed
  // by the test. These are deallocated on destruction of the test object.
  std::vector<perftools::gputools::DeviceMemoryBase> allocations_;

  ErrorSpec error_spec_{0.0001};

  std::unique_ptr<EigenThreadPoolWrapper> thread_pool_wrapper_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_TESTS_HLO_TEST_BASE_H_
