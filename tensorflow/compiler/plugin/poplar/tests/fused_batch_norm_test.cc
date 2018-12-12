// /* Copyright 2018 Graphcore Ltd

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ==============================================================================*/

#include <cmath>
#include <memory>
#include <vector>

#include "absl/strings/str_join.h"
#include "tensorflow/compiler/plugin/poplar/tests/test_utils.h"
#include "tensorflow/compiler/xla/array3d.h"
#include "tensorflow/compiler/xla/client/lib/arithmetic.h"
#include "tensorflow/compiler/xla/client/lib/math.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/reference_util.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/test_helpers.h"
#include "tensorflow/compiler/xla/tests/client_library_test_base.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/compiler/xla/tests/test_utils.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/math/math_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/types.h"

namespace xla {
namespace poplarplugin {
namespace {

/* Test based on tensorflow/compiler/xla/tests/batch_normalization_test.cc */

struct HloFusedBatchNorm3DTestParam {
  std::vector<int64> bounds;
  int64 feature_index;
  float random_value_mean;
  float random_value_var;

  friend ::std::ostream& operator<<(::std::ostream& os,
                                    const HloFusedBatchNorm3DTestParam& p) {
    os << "bounds={" << absl::StrJoin(p.bounds, ", ") << "}, ";
    os << "feature_index=" << p.feature_index << ", ";
    os << "random_value_mean=" << p.random_value_mean << ", ";
    os << "random_value_var=" << p.random_value_var;
    return os;
  }
};

// Tests to test the fused operation of BatchNorm.
class HloFusedBatchNorm3DTest
    : public ClientLibraryTestBase,
      public ::testing::WithParamInterface<HloFusedBatchNorm3DTestParam> {
 public:
  HloFusedBatchNorm3DTest() {}
};

std::vector<HloFusedBatchNorm3DTestParam> BuildHloFusedBatchNorm3DTestParams() {
  std::vector<HloFusedBatchNorm3DTestParam> params;

  auto add_testcase = [&](std::vector<int64> bounds, int64 feature_index,
                          float random_value_mean, float random_value_var) {
    HloFusedBatchNorm3DTestParam p{bounds, feature_index, random_value_mean,
                                   random_value_var};
    params.push_back(p);
  };
  std::vector<int64> shape = {2, 2, 2};
  for (auto dim : shape) {
    add_testcase(shape, dim, 100.2f, 200.0f);
  }

  return params;
}

INSTANTIATE_TEST_CASE_P(
    HloFusedBatchNorm3DTest_Instantiation, HloFusedBatchNorm3DTest,
    ::testing::ValuesIn(BuildHloFusedBatchNorm3DTestParams()));

POPLAR_TEST_P(HloFusedBatchNorm3DTest, RandomizedInferencingTests) {
  VLOG(1) << "Test case " << GetParam();
  float epsilon = 0.001;
  XlaBuilder builder(TestName());

  const std::vector<int64>& bounds = GetParam().bounds;
  Array3D<float> input_array(bounds[0], bounds[1], bounds[2]);

  input_array.FillRandom(GetParam().random_value_var,
                         GetParam().random_value_mean);

  const int64 feature_index = GetParam().feature_index;
  const int64 num_elements_per_feature =
      Product(bounds) / bounds[feature_index];
  const int64 feature_bound = bounds[feature_index];
  std::vector<float> offset(feature_bound, 1);
  std::vector<float> scale(feature_bound, 2);

  auto input_squared =
      reference_util::MapArray3D(input_array, [](float a) { return a * a; });
  std::vector<int64> reduce_dims;
  for (int64 i = 0; i < static_cast<int64>(bounds.size()); ++i) {
    if (i != feature_index) {
      reduce_dims.push_back(i);
    }
  }

  auto sum =
      reference_util::Reduce3DTo1D(input_array, /*init=*/0.0f, reduce_dims,
                                   [](float a, float b) { return a + b; });

  auto sum_squared =
      reference_util::Reduce3DTo1D(*input_squared, /*init=*/0.0f, reduce_dims,
                                   [](float a, float b) { return a + b; });

  std::vector<float> mean(feature_bound);

  for (int64 i = 0; i < feature_bound; ++i) {
    mean[i] = sum[i] / num_elements_per_feature;
  }

  std::vector<float> mean_square(feature_bound);
  for (int64 i = 0; i < feature_bound; ++i) {
    mean_square[i] = mean[i] * mean[i];
  }

  std::vector<float> square_mean(feature_bound);
  for (int64 i = 0; i < feature_bound; ++i) {
    square_mean[i] = sum_squared[i] / num_elements_per_feature;
  }

  std::vector<float> var(feature_bound);
  for (int64 i = 0; i < feature_bound; ++i) {
    var[i] = square_mean[i] - mean_square[i];
  }

  Array3D<float> mean3D =
      *reference_util::Broadcast1DTo3D(mean, bounds, feature_index);
  auto var3D = *reference_util::Broadcast1DTo3D(var, bounds, feature_index);
  auto scale3D = *reference_util::Broadcast1DTo3D(scale, bounds, feature_index);
  auto offset3D =
      *reference_util::Broadcast1DTo3D(offset, bounds, feature_index);

  auto normalized = *reference_util::BatchNorm3D(input_array, mean3D, var3D,
                                                 scale3D, offset3D, epsilon);

  auto offset_literal = LiteralUtil::CreateR1<float>(offset);
  auto scale_literal = LiteralUtil::CreateR1<float>(scale);
  auto mean_literal = LiteralUtil::CreateR1<float>(mean);
  auto var_literal = LiteralUtil::CreateR1<float>(var);
  auto input_literal = LiteralUtil::CreateR3FromArray3D<float>(input_array);

  auto input_activations =
      Parameter(&builder, 0, input_literal.shape(), "input");
  auto scale_activations =
      Parameter(&builder, 1, scale_literal.shape(), "offset");
  auto offset_activations =
      Parameter(&builder, 2, offset_literal.shape(), "scale");
  auto mean_activations = Parameter(&builder, 3, mean_literal.shape(), "mean");
  auto variance_activations =
      Parameter(&builder, 4, var_literal.shape(), "variance");

  Array3D<float> expected = normalized;

  std::unique_ptr<GlobalData> input_data =
      client_->TransferToServer(input_literal).ConsumeValueOrDie();
  std::unique_ptr<GlobalData> scale_data =
      client_->TransferToServer(scale_literal).ConsumeValueOrDie();
  std::unique_ptr<GlobalData> offset_data =
      client_->TransferToServer(offset_literal).ConsumeValueOrDie();
  std::unique_ptr<GlobalData> mean_data =
      client_->TransferToServer(mean_literal).ConsumeValueOrDie();
  std::unique_ptr<GlobalData> variance_data =
      client_->TransferToServer(var_literal).ConsumeValueOrDie();

  BatchNormInference(input_activations, scale_activations, offset_activations,
                     mean_activations, variance_activations, epsilon,
                     feature_index);

  ComputeAndCompareR3<float>(
      &builder, expected,
      {input_data.get(), scale_data.get(), offset_data.get(), mean_data.get(),
       variance_data.get()},
      ErrorSpec(0.01, 1));
}

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
