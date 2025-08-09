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

#include "xla/service/gpu/model/sol_gpu_cost_model_stats_collection.h"

#include <cstdint>
#include <functional>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/log.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/statusor.h"

namespace xla::gpu {
namespace {

using ::testing::Gt;
using ::testing::Property;

using ShapeSizeFn = std::function<int64_t(const Shape&)>;

class SolGpuCostModelStatsCollectionTest
    : public HloHardwareIndependentTestBase {
 public:
  explicit SolGpuCostModelStatsCollectionTest() {
    ShapeSizeFn shape_size_bytes =
        [&shape_size_bytes](const Shape& shape) -> int64_t {
      int64_t shape_size = 0;
      if (shape.IsTuple()) {
        for (auto& sub_shape : shape.tuple_shapes()) {
          shape_size += shape_size_bytes(sub_shape);
        }
        return shape_size;
      }
      return ShapeUtil::ByteSizeOfElements(shape);
    };
    shape_size_fn_ = shape_size_bytes;
  }

 protected:
  se::DeviceDescription device_info_ =
      TestGpuDeviceInfo::RTXA6000DeviceInfo(se::CudaComputeCapability(9, 0));
  ShapeSizeFn shape_size_fn_;
  int pointer_size_ = 8;
};

TEST_F(SolGpuCostModelStatsCollectionTest,
       RecordsRuntimeInformationForCollectives) {
  constexpr absl::string_view kHloText = R"(
  HloModule m

  add {
    x = f32[] parameter(0)
    y = f32[] parameter(1)
    ROOT _ = f32[] add(x, y)
  }

  ENTRY ar {
    p0 = f32[8192,4096] parameter(0)

    ar-start = f32[8192,4096] all-reduce-start(p0), to_apply=add,
      replica_groups={{0,1,2,3,4,5,6,7}, {8,9,10,11,12,13,14,15}}
    ROOT ar-done = f32[8192,4096] all-reduce-done(ar-start)
  })";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHloText));

  TF_ASSERT_OK_AND_ASSIGN(
      bool changed, SolGpuCostModelStatsCollection(device_info_, shape_size_fn_,
                                                   pointer_size_)
                        .Run(module.get()));

  VLOG(1) << module->ToString();

  EXPECT_FALSE(changed);
  EXPECT_THAT(module->entry_computation()
                  ->root_instruction()
                  ->operand(0)
                  ->backend_config<GpuBackendConfig>()
                  ->reification_cost(),
              ElementsAre(Property(&ReificationCost::exec_time_us, Gt(0))));
}

}  // namespace
}  // namespace xla::gpu
