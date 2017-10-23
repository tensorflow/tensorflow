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

#ifndef TENSORFLOW_COMPILER_XLA_TESTS_HLO_VERIFIED_TEST_BASE_H_
#define TENSORFLOW_COMPILER_XLA_TESTS_HLO_VERIFIED_TEST_BASE_H_

#include <functional>
#include <memory>
#include <utility>

#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"

namespace xla {

// A base class for HLO tests that stores a default HloModule, and automatically
// performs verification on that module on tear-down.
class HloVerifiedTestBase : public HloTestBase {
 public:
  // Returns the size in bytes of the given shape, using a default pointer size.
  static int64 DefaultShapeSize(const Shape& shape);

 protected:
  HloVerifiedTestBase();
  ~HloVerifiedTestBase() override;

  // Performs verification on the default HloModule returned by module().
  // Automatically called by the testing framework for each test.
  //
  // REQUIRED: subclasses that override TearDown() must call this explicitly.
  void TearDown() override;

  // Returns the default HloModule, lazily creating it if necessary via
  // HloTestBase::CreateNewModule().
  HloModule& module();

  // Sets the shape-size function used during hlo verification. If this isn't
  // called, DefaultShapeSize is used instead.
  void SetShapeSizeFn(std::function<int64(const Shape&)> shape_size_fn) {
    shape_size_fn_ = std::move(shape_size_fn);
  }

 private:
  std::unique_ptr<HloModule> module_;  // Lazily populated. Access via module().
  std::function<int64(const Shape&)> shape_size_fn_;
  bool tear_down_called_ = false;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_TESTS_HLO_VERIFIED_TEST_BASE_H_
