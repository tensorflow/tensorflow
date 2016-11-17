/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#ifndef THIRD_PARTY_TENSORFLOW_CORE_KERNELS_FAKE_QUANT_FUNCTOR_H_
#define THIRD_PARTY_TENSORFLOW_CORE_KERNELS_FAKE_QUANT_FUNCTOR_H_

#include <tuple>

#define EIGEN_STACK_ALLOCATION_LIMIT 0
#define EIGEN_USE_THREADS
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"

EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE float StdRound(float input) {
// On Android, std::round() isn't present, just round().
#if defined(__ANDROID__)
  return round(input);
#else
  return std::round(input);
#endif
}

namespace tensorflow {

static constexpr int kSteps = 255;
static constexpr float kStepsFloat = static_cast<float>(kSteps);

// Gymnastics with nudged zero point is to ensure that real zero maps to
// an integer, which is required for e.g. zero-padding in convolutional layers.
// Returns (nudged_min, nudged_max, nudged_scale).
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE void Nudge(const float min,
                                                 const float max,
                                                 float* nudged_min,
                                                 float* nudged_max,
                                                 float* scale) {
  *scale = (max - min) / (kStepsFloat - 0.0f);
  const float zero_point_from_min = 0.0f - min / *scale;
  const uint8 nudged_zero_point = [zero_point_from_min] {
    if (zero_point_from_min < 0.0f) {
      return static_cast<uint8>(0);
    } else if (zero_point_from_min > kStepsFloat) {
      return static_cast<uint8>(kSteps);
    } else {
      return static_cast<uint8>(StdRound(zero_point_from_min));
    }
  }();

  *nudged_min = (0.0f - nudged_zero_point) * (*scale);
  *nudged_max = (kStepsFloat - nudged_zero_point) * (*scale);
}

template <typename T>
using ConstScalar = typename tensorflow::TTypes<T>::ConstScalar;
template <typename T>
using Scalar = typename tensorflow::TTypes<T>::Scalar;
template <typename T>
using ConstVec = typename tensorflow::TTypes<T>::ConstVec;
template <typename T>
using Vec = typename tensorflow::TTypes<T>::Vec;
template <typename T>
using ConstFlat = typename tensorflow::TTypes<T>::ConstFlat;
template <typename T>
using Flat = typename tensorflow::TTypes<T>::Flat;

// Functor called by FakeQuantWithMinMaxArgsOp to do the work.  Compiles both
// for CPU and GPU.
template <typename Device>
struct FakeQuantWithMinMaxArgsFunctor {
  void operator()(const Device& d, ConstFlat<float> inputs, const float min,
                  const float max, Flat<float> outputs) {
    eigen_assert(min <= 0.0f && "min should be <= 0.0");
    eigen_assert(max >= 0.0f && "max should be >= 0.0");
    eigen_assert(min < max && "min should be < max");

    float nudged_min, nudged_max, nudged_scale;
    Nudge(min, max, &nudged_min, &nudged_max, &nudged_scale);
    const float inv_nudged_scale = 1.0f / nudged_scale;

    auto clamped = inputs.cwiseMin(nudged_max).cwiseMax(nudged_min);
    auto clamped_shifted = clamped - nudged_min;
    outputs.device(d) =
        (clamped_shifted * inv_nudged_scale + 0.5f).floor() * nudged_scale +
        nudged_min;
  }
};

// Functor called by FakeQuantWithMinMaxArgsGradientOp to do the work.  Compiles
// both for CPU and GPU.
template <typename Device>
struct FakeQuantWithMinMaxArgsGradientFunctor {
  void operator()(const Device& d, ConstFlat<float> gradients,
                  ConstFlat<float> inputs, const float min, const float max,
                  Flat<float> backprops) {
    eigen_assert(min <= 0.0f && "min should be <= 0.0");
    eigen_assert(max >= 0.0f && "max should be >= 0.0");
    eigen_assert(min < max && "min should be < max");

    float nudged_min, nudged_max, nudged_scale;
    Nudge(min, max, &nudged_min, &nudged_max, &nudged_scale);

    auto between_nudged_min_max =
        (inputs >= nudged_min && inputs <= nudged_max)
            .select(inputs.constant(1.0f), inputs.constant(0.0f));
    backprops.device(d) = gradients * between_nudged_min_max;
  }
};

// Functor called by FakeQuantWithMinMaxVarsOp to do the work.  Compiles both
// for CPU and GPU.
template <typename Device>
struct FakeQuantWithMinMaxVarsFunctor {
  void operator()(const Device& d, ConstFlat<float> inputs,
                  ConstScalar<float> min, ConstScalar<float> max,
#ifndef FAKE_QUANT_NO_DEBUG
                  Scalar<bool> check_min_max,
#endif
                  Flat<float> outputs) {
#ifndef FAKE_QUANT_NO_DEBUG
    check_min_max.device(d) = (min <= 0.0f).all();
    eigen_assert(check_min_max() && "min should be <= 0.0 coeff-wise");
    check_min_max.device(d) = (max >= 0.0f).all();
    eigen_assert(check_min_max() >= 0.0f && "max should be >= 0.0 coeff-wise");
    check_min_max.device(d) = (min < max).all();
    eigen_assert(check_min_max() && "min should be < max coeff-wise");
#endif

    float nudged_min, nudged_max, nudged_scale;
    Nudge(min(), max(), &nudged_min, &nudged_max, &nudged_scale);
    const auto nudged_scale_repl = inputs.constant(nudged_scale);

    const auto clamped = inputs.cwiseMin(nudged_max).cwiseMax(nudged_min);
    const auto clamped_shifted = clamped - nudged_min;
    outputs.device(d) = (clamped_shifted / nudged_scale_repl + 0.5f).floor() *
                            nudged_scale_repl +
                        nudged_min;
  }
};

// Functor called by FakeQuantWithMinMaxVarsGradientOp to do the work.  Compiles
// both for CPU and GPU.
template <typename Device>
struct FakeQuantWithMinMaxVarsGradientFunctor {
  void operator()(const Device& d, ConstFlat<float> gradients,
                  ConstFlat<float> inputs, ConstScalar<float> min,
                  ConstScalar<float> max,
#ifndef FAKE_QUANT_NO_DEBUG
                  Scalar<bool> check_min_max,
#endif
                  Flat<float> backprops_wrt_input,
                  Scalar<float> backprop_wrt_min,
                  Scalar<float> backprop_wrt_max) {
#ifndef FAKE_QUANT_NO_DEBUG
    check_min_max.device(d) = (min <= 0.0f).all();
    eigen_assert(check_min_max() && "min should be <= 0.0 coeff-wise");
    check_min_max.device(d) = (max >= 0.0f).all();
    eigen_assert(check_min_max() >= 0.0f && "max should be >= 0.0 coeff-wise");
    check_min_max.device(d) = (min < max).all();
    eigen_assert(check_min_max() && "min should be < max coeff-wise");
#endif

    float nudged_min, nudged_max, nudged_scale;
    Nudge(min(), max(), &nudged_min, &nudged_max, &nudged_scale);

    const auto between_min_max =
        (inputs >= nudged_min && inputs <= nudged_max)
            .select(inputs.constant(1.0f), inputs.constant(0.0f));
    backprops_wrt_input.device(d) = gradients * between_min_max;

    const auto below_min =
        (inputs < nudged_min)
            .select(inputs.constant(1.0f), inputs.constant(0.0f));
    backprop_wrt_min.device(d) = (gradients * below_min).sum();

    const auto above_max =
        (inputs > nudged_max)
            .select(inputs.constant(1.0f), inputs.constant(0.0f));
    backprop_wrt_max.device(d) = (gradients * above_max).sum();
  }
};

using Index = typename tensorflow::TTypes<float>::ConstTensor::Index;

// Functor called by FakeQuantWithMinMaxVarsPerChannelOp to do the work.
// Compiles both for CPU and GPU.
//
// Already verified: inputs, outputs, min, max are of shape [d].
template <typename Device>
struct FakeQuant1WithMinMaxVarsPerChannelFunctor {
  void operator()(const Device& d, ConstVec<float> inputs, ConstVec<float> min,
                  ConstVec<float> max,
#ifndef FAKE_QUANT_NO_DEBUG
                  Scalar<bool> check_min_max,
#endif
                  Vec<float> outputs) {
#ifndef FAKE_QUANT_NO_DEBUG
    check_min_max.device(d) = (min <= 0.0f).all();
    eigen_assert(check_min_max() && "min should be <= 0.0 coeff-wise");
    check_min_max.device(d) = (max >= 0.0f).all();
    eigen_assert(check_min_max() >= 0.0f && "max should be >= 0.0 coeff-wise");
    check_min_max.device(d) = (min < max).all();
    eigen_assert(check_min_max() && "min should be < max coeff-wise");
#endif

    for (Index i = 0; i < min.size(); ++i) {
      float nudged_min, nudged_max, nudged_scale;
      Nudge(min(i), max(i), &nudged_min, &nudged_max, &nudged_scale);
      const float clamped =
          std::max(std::min(inputs(i), nudged_max), nudged_min);
      const float clamped_shifted = clamped - nudged_min;

      outputs(i) =
          StdRound(clamped_shifted / nudged_scale) * nudged_scale + nudged_min;
    }
  }
};

// Already verified: inputs, outputs are of shape [b, d], min, max are of shape
// [d].
template <typename Device>
struct FakeQuant2WithMinMaxVarsPerChannelFunctor {
  void operator()(const Device& d, const Index batch_size, const Index depth,
                  ConstFlat<float> inputs, ConstVec<float> min,
                  ConstVec<float> max,
#ifndef FAKE_QUANT_NO_DEBUG
                  Scalar<bool> check_min_max,
#endif
                  Flat<float> outputs) {
#ifndef FAKE_QUANT_NO_DEBUG
    check_min_max.device(d) = (min <= 0.0f).all();
    eigen_assert(check_min_max() && "min should be <= 0.0 coeff-wise");
    check_min_max.device(d) = (max >= 0.0f).all();
    eigen_assert(check_min_max() >= 0.0f && "max should be >= 0.0 coeff-wise");
    check_min_max.device(d) = (min < max).all();
    eigen_assert(check_min_max() && "min should be < max coeff-wise");
#endif

    Eigen::DSizes<Index, 2> restored(batch_size, depth);
    const auto inputs_restored = inputs.reshape(restored);
    for (Index i = 0; i < min.size(); ++i) {
      float nudged_min, nudged_max, nudged_scale;
      Nudge(min(i), max(i), &nudged_min, &nudged_max, &nudged_scale);
      const auto clamped =
          inputs_restored.chip<1>(i).cwiseMin(nudged_max).cwiseMax(nudged_min);
      const auto clamped_shifted = clamped - nudged_min;

      outputs.reshape(restored).chip<1>(i).device(d) =
          (clamped_shifted / nudged_scale + 0.5f).floor() * nudged_scale +
          nudged_min;
    }
  }
};

// Already verified: inputs, outputs are of shape [b, h, w, d], min, max are
// of shape [d].
template <typename Device>
struct FakeQuant4WithMinMaxVarsPerChannelFunctor {
  void operator()(const Device& d, const Index batch_size, const Index height,
                  const Index width, const Index depth, ConstFlat<float> inputs,
                  ConstVec<float> min, ConstVec<float> max,
#ifndef FAKE_QUANT_NO_DEBUG
                  Scalar<bool> check_min_max,
#endif
                  Flat<float> outputs) {
#ifndef FAKE_QUANT_NO_DEBUG
    check_min_max.device(d) = (min <= 0.0f).all();
    eigen_assert(check_min_max() && "min should be <= 0.0 coeff-wise");
    check_min_max.device(d) = (max >= 0.0f).all();
    eigen_assert(check_min_max() >= 0.0f && "max should be >= 0.0 coeff-wise");
    check_min_max.device(d) = (min < max).all();
    eigen_assert(check_min_max() && "min should be < max coeff-wise");
#endif

    Eigen::DSizes<Index, 4> restored(batch_size, height, width, depth);
    const auto inputs_restored = inputs.reshape(restored);
    for (Index i = 0; i < min.size(); ++i) {
      float nudged_min, nudged_max, nudged_scale;
      Nudge(min(i), max(i), &nudged_min, &nudged_max, &nudged_scale);
      const auto clamped =
          inputs_restored.chip<3>(i).cwiseMin(nudged_max).cwiseMax(nudged_min);
      const auto clamped_shifted = clamped - nudged_min;

      outputs.reshape(restored).chip<3>(i).device(d) =
          (clamped_shifted / nudged_scale + 0.5f).floor() * nudged_scale +
          nudged_min;
    }
  }
};

// Functor called by FakeQuantWithMinMaxVarsPerChannelGradientOp to do the work.
// Compiles both for CPU and GPU.
//
// Already verified: gradients, inputs, outputs, min, max, backprops_wrt_input,
// backprop_wrt_min, backprop_wrt_max are of shape [d].
template <typename Device>
struct FakeQuant1WithMinMaxVarsPerChannelGradientFunctor {
  void operator()(const Device& d, ConstVec<float> gradients,
                  ConstVec<float> inputs, ConstVec<float> min,
                  ConstVec<float> max,
#ifndef FAKE_QUANT_NO_DEBUG
                  Scalar<bool> check_min_max,
#endif
                  Vec<float> backprops_wrt_input, Vec<float> backprop_wrt_min,
                  Vec<float> backprop_wrt_max) {
#ifndef FAKE_QUANT_NO_DEBUG
    check_min_max.device(d) = (min <= 0.0f).all();
    eigen_assert(check_min_max() && "min should be <= 0.0 coeff-wise");
    check_min_max.device(d) = (max >= 0.0f).all();
    eigen_assert(check_min_max() >= 0.0f && "max should be >= 0.0 coeff-wise");
    check_min_max.device(d) = (min < max).all();
    eigen_assert(check_min_max() && "min should be < max coeff-wise");
#endif

    for (Index i = 0; i < min.size(); ++i) {
      float nudged_min, nudged_max, nudged_scale;
      Nudge(min(i), max(i), &nudged_min, &nudged_max, &nudged_scale);

      const bool between_min_max =
          inputs(i) >= nudged_min && inputs(i) <= nudged_max;
      backprops_wrt_input(i) = between_min_max ? gradients(i) : 0.0f;

      const bool below_min = inputs(i) < nudged_min;
      backprop_wrt_min(i) = below_min ? gradients(i) : 0.0f;

      const bool above_max = inputs(i) > nudged_max;
      backprop_wrt_max(i) = above_max ? gradients(i) : 0.0f;
    }
  }
};

// Already verified: gradients, inputs, backprops_wrt_input are of shape [b, d],
// min, max, backprop_wrt_min, backprop_wrt_max are of shape [d].
template <typename Device>
struct FakeQuant2WithMinMaxVarsPerChannelGradientFunctor {
  void operator()(const Device& d, const Index batch_size, const Index depth,
                  ConstFlat<float> gradients, ConstFlat<float> inputs,
                  ConstVec<float> min, ConstVec<float> max,
#ifndef FAKE_QUANT_NO_DEBUG
                  Scalar<bool> check_min_max,
#endif
                  Flat<float> backprops_wrt_input, Vec<float> backprop_wrt_min,
                  Vec<float> backprop_wrt_max) {
#ifndef FAKE_QUANT_NO_DEBUG
    check_min_max.device(d) = (min <= 0.0f).all();
    eigen_assert(check_min_max() && "min should be <= 0.0 coeff-wise");
    check_min_max.device(d) = (max >= 0.0f).all();
    eigen_assert(check_min_max() >= 0.0f && "max should be >= 0.0 coeff-wise");
    check_min_max.device(d) = (min < max).all();
    eigen_assert(check_min_max() && "min should be < max coeff-wise");
#endif

    Eigen::DSizes<Index, 2> restored(batch_size, depth);
    const auto gradients_restored = gradients.reshape(restored);
    const auto inputs_restored = inputs.reshape(restored);
    for (Index i = 0; i < min.size(); ++i) {
      float nudged_min, nudged_max, nudged_scale;
      Nudge(min(i), max(i), &nudged_min, &nudged_max, &nudged_scale);
      const auto gradients_chip = gradients_restored.chip<1>(i);
      const auto inputs_chip = inputs_restored.chip<1>(i);

      const auto between_min_max =
          (inputs_chip >= nudged_min && inputs_chip <= nudged_max)
              .select(inputs_chip.constant(1.0f), inputs_chip.constant(0.0f));
      backprops_wrt_input.reshape(restored).chip<1>(i).device(d) =
          gradients_chip * between_min_max;

      const auto below_min =
          (inputs_chip < nudged_min)
              .select(inputs_chip.constant(1.0f), inputs_chip.constant(0.0f));
      Eigen::DSizes<Index, 1> reduce(0);
      backprop_wrt_min.chip<0>(i).device(d) =
          (gradients_chip * below_min).sum(reduce);

      const auto above_max =
          (inputs_chip > nudged_max)
              .select(inputs_chip.constant(1.0f), inputs_chip.constant(0.0f));
      backprop_wrt_max.chip<0>(i).device(d) =
          (gradients_chip * above_max).sum(reduce);
    }
  }
};

// Already verified: gradients, inputs, backprops_wrt_input are of shape
// [b, h, w, d], min, max, backprop_wrt_min, backprop_wrt_max are of shape [d].
template <typename Device>
struct FakeQuant4WithMinMaxVarsPerChannelGradientFunctor {
  void operator()(const Device& d, const Index batch_size, const Index height,
                  const Index width, const Index depth,
                  ConstFlat<float> gradients, ConstFlat<float> inputs,
                  ConstVec<float> min, ConstVec<float> max,
#ifndef FAKE_QUANT_NO_DEBUG
                  Scalar<bool> check_min_max,
#endif
                  Flat<float> backprops_wrt_input, Vec<float> backprop_wrt_min,
                  Vec<float> backprop_wrt_max) {
#ifndef FAKE_QUANT_NO_DEBUG
    check_min_max.device(d) = (min <= 0.0f).all();
    eigen_assert(check_min_max() && "min should be <= 0.0 coeff-wise");
    check_min_max.device(d) = (max >= 0.0f).all();
    eigen_assert(check_min_max() >= 0.0f && "max should be >= 0.0 coeff-wise");
    check_min_max.device(d) = (min < max).all();
    eigen_assert(check_min_max() && "min should be < max coeff-wise");
#endif

    Eigen::DSizes<Index, 4> restored(batch_size, height, width, depth);
    const auto gradients_restored = gradients.reshape(restored);
    const auto inputs_restored = inputs.reshape(restored);
    for (Index i = 0; i < min.size(); ++i) {
      float nudged_min, nudged_max, nudged_scale;
      Nudge(min(i), max(i), &nudged_min, &nudged_max, &nudged_scale);
      const auto gradients_chip = gradients_restored.chip<3>(i);
      const auto inputs_chip = inputs_restored.chip<3>(i);

      const auto between_min_max =
          (inputs_chip >= nudged_min && inputs_chip <= nudged_max)
              .select(inputs_chip.constant(1.0f), inputs_chip.constant(0.0f));
      backprops_wrt_input.reshape(restored).chip<3>(i).device(d) =
          gradients_chip * between_min_max;

      const auto below_min =
          (inputs_chip < nudged_min)
              .select(inputs_chip.constant(1.0f), inputs_chip.constant(0.0f));
      Eigen::DSizes<Index, 3> reduce(0, 1, 2);
      backprop_wrt_min.chip<0>(i).device(d) =
          (gradients_chip * below_min).sum(reduce);

      const auto above_max =
          (inputs_chip > nudged_max)
              .select(inputs_chip.constant(1.0f), inputs_chip.constant(0.0f));
      backprop_wrt_max.chip<0>(i).device(d) =
          (gradients_chip * above_max).sum(reduce);
    }
  }
};

}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_CORE_KERNELS_FAKE_QUANT_FUNCTOR_H_
