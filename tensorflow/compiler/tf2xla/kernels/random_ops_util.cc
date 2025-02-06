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

#include "tensorflow/compiler/tf2xla/kernels/random_ops_util.h"

#include <functional>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "tensorflow/compiler/tf2xla/kernels/rng_converter_utils.h"
#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "xla/hlo/builder/lib/constants.h"
#include "xla/hlo/builder/lib/prng.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/literal.h"
#include "xla/primitive_util.h"
#include "xla/shape.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tensorflow/core/framework/rng_alg.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/kernels/stateless_random_ops_v2.h"
#include "tensorflow/core/platform/types.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace tensorflow {

namespace {

xla::XlaOp GetCounter(xla::RandomAlgorithm const& alg, xla::XlaOp state) {
  return xla::Slice(state, {RNG_KEY_SIZE},
                    {RNG_KEY_SIZE + xla::GetCounterSize(alg)}, {1});
}

absl::StatusOr<xla::RandomAlgorithm> ResolveAlg(
    int alg_id, absl::string_view device_type_string) {
  switch (alg_id) {
    case RNG_ALG_PHILOX:
      return xla::RandomAlgorithm::RNG_PHILOX;
    case RNG_ALG_THREEFRY:
      return xla::RandomAlgorithm::RNG_THREE_FRY;
    case RNG_ALG_AUTO_SELECT:
      return DefaultRngAlgForDeviceType(device_type_string);
    default:
      return absl::InvalidArgumentError(
          absl::StrCat("Unsupported algorithm id: ", alg_id));
  }
}

xla::RngOutput StatelessRngUniformV2(xla::RandomAlgorithm const& alg,
                                     xla::XlaOp key, xla::XlaOp counter,
                                     const xla::Shape& shape, xla::XlaOp minval,
                                     xla::XlaOp maxval) {
  xla::XlaBuilder* builder = key.builder();
  xla::PrimitiveType type = shape.element_type();
  using std::placeholders::_1;
  using std::placeholders::_2;
  using std::placeholders::_3;
  auto generator = std::bind(BitGenerator, alg, _1, _2, _3);
  switch (type) {
    case xla::F16:
    case xla::F32:
    case xla::F64:
      return xla::UniformFloatingPointDistribution(key, counter, generator,
                                                   minval, maxval, shape);
    case xla::S32:
    case xla::S64:
    case xla::U32:
    case xla::U64:
      return xla::UniformIntDistribution(key, counter, generator, minval,
                                         maxval, shape);
      break;
    default:
      return {builder->ReportError(xla::Unimplemented(
                  "Types other than F16, F32, S32, S64, U32 and U64 are not "
                  "implemented by "
                  "StatelessRngUniformV2; got %s",
                  xla::primitive_util::LowercasePrimitiveTypeName(type))),
              counter};
  }
}

}  // namespace

xla::RngOutput BitGenerator(xla::RandomAlgorithm const& alg, xla::XlaOp key,
                            xla::XlaOp counter, const xla::Shape& shape) {
  key = BitcastConvertType(key, xla::U64);
  counter = BitcastConvertType(counter, xla::U64);
  xla::XlaOp state = xla::ConcatInDim(key.builder(), {key, counter}, 0);
  xla::XlaOp result = xla::RngBitGenerator(alg, state, shape);
  auto new_counter = GetCounter(alg, xla::GetTupleElement(result, 0));
  new_counter = BitcastConvertType(new_counter, xla::S64);
  return xla::RngOutput{/*value=*/xla::GetTupleElement(result, 1),
                        /*state=*/new_counter};
}
xla::XlaOp GetU64FromS32Seeds(xla::XlaOp seed0, xla::XlaOp seed1) {
  // Here, the seeds are cast to unsigned type of the same width to have leading
  // zeros in the 64 bit representation.
  xla::XlaOp u64_seed0 =
      ConvertElementType(ConvertElementType(seed0, xla::U32), xla::U64);
  xla::XlaOp u64_seed1 =
      ConvertElementType(ConvertElementType(seed1, xla::U32), xla::U64);
  return u64_seed0 |
         (u64_seed1 << ConstantR0WithType(seed0.builder(), xla::U64, 32));
}

absl::StatusOr<int> GetAlgId(XlaOpKernelContext* ctx, int alg_input_idx) {
  TF_ASSIGN_OR_RETURN(auto alg_shape, ctx->InputXlaShape(alg_input_idx));
  if (alg_shape.rank() != 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("The algorithm argument must be of shape [], not ",
                     alg_shape.DebugString()));
  }
  auto alg_dtype = ctx->input_type(alg_input_idx);
  if (alg_dtype != DT_INT32 && alg_dtype != DT_INT64) {
    return absl::InvalidArgumentError(absl::StrCat(
        "The algorithm argument must have dtype int32 or int64, not ",
        DataTypeString(alg_dtype)));
  }
  xla::Literal alg_literal;
  TF_RETURN_IF_ERROR(ctx->ConstantInput(alg_input_idx, &alg_literal));
  if (alg_dtype == DT_INT32) {
    return alg_literal.Get<int>({});
  } else {
    return alg_literal.Get<int64>({});
  }
}

absl::StatusOr<xla::RandomAlgorithm> AlgorithmFromInput(
    XlaOpKernelContext* ctx, int alg_input_idx,
    absl::string_view device_type_string) {
  TF_ASSIGN_OR_RETURN(auto alg_id, GetAlgId(ctx, alg_input_idx));
  return ResolveAlg(alg_id, device_type_string);
}

xla::XlaOp MaybeSliceCounter(xla::RandomAlgorithm const& alg,
                             TensorShape const& counter_shape,
                             xla::XlaOp counter) {
  auto input_counter_size = counter_shape.dim_size(0);
  auto real_counter_size = xla::GetCounterSize(alg);
  if (input_counter_size > real_counter_size) {
    counter = xla::Slice(counter, {0}, {real_counter_size}, {1});
  }
  return counter;
}

DataType MaybeConvertBF16ToF32(DataType const& dtype) {
  if (dtype == DT_BFLOAT16) {
    // We'll go through F32 to generate BF16.
    // TODO(b/256243456): Generate BF16 directly from U16.
    return DT_FLOAT;
  }
  return dtype;
}

absl::StatusOr<xla::XlaOp> BuildUniformRandoms(
    XlaOpKernelContext* ctx, DataType dtype, string device_type_string,
    TensorShape shape,
    std::function<xla::XlaOp(xla::XlaBuilder*, xla::PrimitiveType)> lo_fn,
    std::function<xla::XlaOp(xla::XlaBuilder*, xla::PrimitiveType)> hi_fn) {
  xla::XlaBuilder* builder = ctx->builder();
  auto rng_dtype = MaybeConvertBF16ToF32(dtype);
  xla::Shape xla_shape;
  TF_RETURN_IF_ERROR(TensorShapeToXLAShape(rng_dtype, shape, &xla_shape));

  xla::PrimitiveType rng_primitive_type = xla_shape.element_type();
  xla::XlaOp lo = lo_fn(builder, rng_primitive_type);
  xla::XlaOp hi = hi_fn(builder, rng_primitive_type);

  return BuildUniformRandoms(ctx, dtype, device_type_string, xla_shape, lo, hi);
}

absl::StatusOr<xla::XlaOp> BuildUniformRandoms(XlaOpKernelContext* ctx,
                                               DataType dtype,
                                               string device_type_string,
                                               xla::Shape xla_shape,
                                               xla::XlaOp lo, xla::XlaOp hi) {
  xla::XlaOp key = ctx->Input(kRandomKeyInputIdx);
  xla::XlaOp counter = ctx->Input(kRandomCounterInputIdx);

  TF_ASSIGN_OR_RETURN(
      xla::RandomAlgorithm alg,
      AlgorithmFromInput(ctx, kRandomAlgInputIdx, device_type_string));
  TensorShape counter_shape = ctx->InputShape(kRandomCounterInputIdx);
  TF_RETURN_IF_ERROR(CheckKeyCounterShape(
      GetCounterSize(alg), ctx->InputShape(kRandomKeyInputIdx), counter_shape));
  counter = MaybeSliceCounter(alg, counter_shape, counter);

  xla::RngOutput result =
      StatelessRngUniformV2(alg, key, counter, xla_shape, lo, hi);
  return result.value;
}
}  // namespace tensorflow

namespace xla {

int GetCounterSize(RandomAlgorithm const& alg) {
  switch (alg) {
    case RandomAlgorithm::RNG_PHILOX:
      return 2;
    case RandomAlgorithm::RNG_THREE_FRY:  // fall through
    case RandomAlgorithm::RNG_DEFAULT:    // fall through
    default:
      return 1;
  }
}

}  // namespace xla
