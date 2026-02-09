// Copyright 2025 TF.Text Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tensorflow_text/core/kernels/round_robin_trimmer_kernel.h"

#include <cstdint>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"

namespace tensorflow {
namespace text {

using RoundRobinGenerateMasksOpKernelInstance =
    RoundRobinGenerateMasksOpKernel<int32_t, int32_t>;

#define REGISTER_ROUND_ROBIN_GENERATE_MASKS_SPLITS(vals_type, splits_type) \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name(RoundRobinGenerateMasksOpKernelInstance::OpName())              \
          .Device(tensorflow::DEVICE_CPU)                                  \
          .TypeConstraint<vals_type>("T")                                  \
          .TypeConstraint<splits_type>("Tsplits"),                         \
      RoundRobinGenerateMasksOpKernel<vals_type, splits_type>);

#define REGISTER_ROUND_ROBIN_GENERATE_MASKS(vals_type)           \
  REGISTER_ROUND_ROBIN_GENERATE_MASKS_SPLITS(vals_type, int32_t) \
  REGISTER_ROUND_ROBIN_GENERATE_MASKS_SPLITS(vals_type, int64_t)

TF_CALL_tstring(REGISTER_ROUND_ROBIN_GENERATE_MASKS)
TF_CALL_bool(REGISTER_ROUND_ROBIN_GENERATE_MASKS)
TF_CALL_float(REGISTER_ROUND_ROBIN_GENERATE_MASKS)
TF_CALL_double(REGISTER_ROUND_ROBIN_GENERATE_MASKS)
TF_CALL_INTEGRAL_TYPES(REGISTER_ROUND_ROBIN_GENERATE_MASKS)

#undef REGISTER_ROUND_ROBIN_GENERATE_MASKS
#undef REGISTER_ROUND_ROBIN_GENERATE_MASKS_SPLITS

                using RoundRobinTrimOpKernelInstance =
                    RoundRobinTrimOpKernel<int32_t, int32_t>;

#define REGISTER_ROUND_ROBIN_TRIM_SPLITS(vals_type, splits_type)         \
  REGISTER_KERNEL_BUILDER(Name(RoundRobinTrimOpKernelInstance::OpName()) \
                              .Device(tensorflow::DEVICE_CPU)            \
                              .TypeConstraint<vals_type>("T")            \
                              .TypeConstraint<splits_type>("Tsplits"),   \
                          RoundRobinTrimOpKernel<vals_type, splits_type>);

#define REGISTER_ROUND_ROBIN_TRIM(vals_type)           \
  REGISTER_ROUND_ROBIN_TRIM_SPLITS(vals_type, int32_t) \
  REGISTER_ROUND_ROBIN_TRIM_SPLITS(vals_type, int64_t)

TF_CALL_tstring(REGISTER_ROUND_ROBIN_TRIM)
TF_CALL_bool(REGISTER_ROUND_ROBIN_TRIM)
TF_CALL_float(REGISTER_ROUND_ROBIN_TRIM)
TF_CALL_double(REGISTER_ROUND_ROBIN_TRIM)
TF_CALL_INTEGRAL_TYPES(REGISTER_ROUND_ROBIN_TRIM)

#undef REGISTER_ROUND_ROBIN_TRIM
#undef REGISTER_ROUND_ROBIN_TRIM_SPLITS

}  // namespace text
}  // namespace tensorflow
