/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_TESTS_HLO_TEST_BASE_H_
#define XLA_TESTS_HLO_TEST_BASE_H_

#include <memory>

#include "xla/pjrt/pjrt_client.h"
#include "xla/tests/hlo_runner_agnostic_test_base.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {

struct HloTestBaseOptions {
  bool verifier_layout_sensitive = false;
  bool allow_mixed_precision_in_hlo_verifier = true;
  HloPredicate instruction_can_change_layout_func;
};

class HloTestBase : public HloRunnerAgnosticTestBase {
 protected:
  // This uses the PjRt interpreter backend for the reference backend and
  // automatically finds a PjRt backend for the test backend.
  explicit HloTestBase(HloTestBaseOptions options = {});

 private:
  HloTestBase(PjRtClient* client, HloTestBaseOptions options);
  HloTestBase(DeviceShapeRepresentationFn device_shape_representation_fn,
              DeviceShapeSizeFn device_shape_size_fn,
              std::unique_ptr<PjRtClient> client, HloTestBaseOptions options);
};

}  // namespace xla

#endif  // XLA_TESTS_HLO_TEST_BASE_H_
