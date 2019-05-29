#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_FLOORDIV_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_FLOORDIV_H_

#include "public/gemmlowp.h"
#include "tensorflow/lite/kernels/internal/common.h"

namespace tflite {
namespace reference_integer_ops {

template <typename T>
inline void DivElementwise(int size, const ArithmeticParams& params,
                           const T* input1_data, const T* input2_data,
                           T* output_data, int q_one, int pot) {
  // The ide is to get the division result and then
  // use quantized value of one multiplied by the multipler to get the
  // result in the range
  for (int i = 0; i < size; ++i) {
    const int32 input1_val = params.input1_offset + input1_data[i];
    int32 input2_val = params.input2_offset + input2_data[i];
    using FixedPoint0 = gemmlowp::FixedPoint<int32, 0>;
    int32 sign_multiplier = 1;
    if (input2_val < 0) {
      sign_multiplier = -1;
      input2_val = sign_multiplier * input2_val;
    }

    const int32 input2_diff_rescaled = MultiplyByQuantizedMultiplier(
        input2_val * (1 << params.left_shift), params.input2_multiplier,
        params.input2_shift);

    int num_bits_over_unit;
    FixedPoint0 shifted_scale = FixedPoint0::FromRaw(
        GetReciprocal(input2_diff_rescaled, 0, &num_bits_over_unit));

    const int32 unsat_output = gemmlowp::RoundingDivideByPOT(
        shifted_scale.raw(), num_bits_over_unit + 31 - 9);

    const int32 input2_scaled = MultiplyByQuantizedMultiplier(
        unsat_output, params.input1_multiplier, params.input1_shift);

    int32 unclamped_result = params.output_offset +
                             MultiplyByQuantizedMultiplier(
                                 sign_multiplier * input2_scaled * input1_val,
                                 params.output_multiplier, params.output_shift);

    shifted_scale = FixedPoint0::FromRaw(unclamped_result);
    int32 unclaimed_pot =
        gemmlowp::RoundingDivideByPOT(shifted_scale.raw(), pot);

    int res = 0;
    // Adjusting the Range below
    if (unclaimed_pot > 0) {
      if (q_one * unclaimed_pot <= unclamped_result) {
        res = q_one * unclaimed_pot;
      } else {
        res = q_one * (unclaimed_pot - 1);
      }
    } else if (q_one * unclaimed_pot >= unclamped_result) {
      res = q_one * (unclaimed_pot - 1);
    } else {
      res = q_one * unclaimed_pot;
    }

    if (res >= std::numeric_limits<T>::max()) {
      res = std::numeric_limits<T>::max();
    } else if (unclamped_result <= std::numeric_limits<T>::min()) {
      res = std::numeric_limits<T>::min();
    }

    output_data[i] = static_cast<T>(res);
  }
}

template <typename T>
inline void FloorDiv(const ArithmeticParams& params,
                     const RuntimeShape& input1_shape, const T* input1_data,
                     const RuntimeShape& input2_shape, const T* input2_data,
                     const RuntimeShape& output_shape, T* output_data,
                     int q_one, int pot) {
  gemmlowp::ScopedProfilingLabel label("FloorDiv/8bit");
  const int flat_size =
      MatchingFlatSize(input1_shape, input2_shape, output_shape);

  DivElementwise(flat_size, params, input1_data, input2_data, output_data,
                 q_one, pot);
}

template <typename T>
inline void BroadcastFloorDiv4DSlow(const ArithmeticParams& params,
                                    const RuntimeShape& input1_shape,
                                    const T* input1_data,
                                    const RuntimeShape& input2_shape,
                                    const T* input2_data,
                                    const RuntimeShape& output_shape,
                                    T* output_data, int q_one, int pot) {
  gemmlowp::ScopedProfilingLabel label("BroadcastDiv4DSlow/8bit");

  // The ide is to get the division result and then
  // use quantized value of one multiplied by the multipler to get the
  // result in the range

  NdArrayDesc<4> desc1;
  NdArrayDesc<4> desc2;
  // The input shapes are extended as part of NdArrayDesc initialization.
  NdArrayDescsForElementwiseBroadcast(input1_shape, input2_shape, &desc1,
                                      &desc2);
  const RuntimeShape extended_output_shape =
      RuntimeShape::ExtendedShape(4, output_shape);

  for (int b = 0; b < extended_output_shape.Dims(0); ++b) {
    for (int y = 0; y < extended_output_shape.Dims(1); ++y) {
      for (int x = 0; x < extended_output_shape.Dims(2); ++x) {
        for (int c = 0; c < extended_output_shape.Dims(3); ++c) {
          const int32 input1_val =
              params.input1_offset +
              input1_data[SubscriptToIndex(desc1, b, y, x, c)];
          int32 input2_val = params.input2_offset +
                             input2_data[SubscriptToIndex(desc2, b, y, x, c)];
          using FixedPoint0 = gemmlowp::FixedPoint<int32, 0>;

          int32 sign_multiplier = 1;
          if (input2_val < 0) {
            sign_multiplier = -1;
            input2_val = sign_multiplier * input2_val;
          }

          const int32 input2_diff_rescaled = MultiplyByQuantizedMultiplier(
              input2_val * (1 << params.left_shift), params.input2_multiplier,
              params.input2_shift);
          int num_bits_over_unit;
          FixedPoint0 shifted_scale = FixedPoint0::FromRaw(
              GetReciprocal(input2_diff_rescaled, 0, &num_bits_over_unit));

          int32 unsat_output = gemmlowp::RoundingDivideByPOT(
              shifted_scale.raw(), num_bits_over_unit + 31 - 9);

          const int32 input2_scaled = MultiplyByQuantizedMultiplier(
              unsat_output, params.input1_multiplier, params.input1_shift);

          int32 unclamped_result =
              params.output_offset +
              MultiplyByQuantizedMultiplier(
                  sign_multiplier * input2_scaled * input1_val,
                  params.output_multiplier, params.output_shift);

          unclamped_result = unclamped_result - params.output_offset;

          shifted_scale = FixedPoint0::FromRaw(unclamped_result);
          int32 unclaimed_pot =
              gemmlowp::RoundingDivideByPOT(shifted_scale.raw(), pot);

          int res = 0;

          // Adjusting the Range below
          if (unclaimed_pot > 0) {
            if (q_one * unclaimed_pot <= unclamped_result) {
              res = q_one * unclaimed_pot;
            } else {
              res = q_one * (unclaimed_pot - 1);
            }
          } else if (q_one * unclaimed_pot >= unclamped_result) {
            res = q_one * (unclaimed_pot - 1);
          } else {
            res = q_one * unclaimed_pot;
          }

          res = res + params.output_offset;

          if (res >= std::numeric_limits<T>::max()) {
            res = std::numeric_limits<T>::max();
          } else if (res <= std::numeric_limits<T>::min()) {
            res = std::numeric_limits<T>::min();
          }

          output_data[Offset(extended_output_shape, b, y, x, c)] =
              static_cast<T>(res);
        }
      }
    }
  }
}

}  // namespace reference_integer_ops
}  // namespace tflite
#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_FLOORDIV_H_
