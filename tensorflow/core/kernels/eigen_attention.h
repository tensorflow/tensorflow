/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_KERNELS_EIGEN_ATTENTION_H_
#define TENSORFLOW_CORE_KERNELS_EIGEN_ATTENTION_H_

#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive

namespace Eigen {

// Noise mode used when padding.
enum ExtractGlimpsesNoiseMode {
  UNIFORM = 0,
  GAUSSIAN = 1,
  ZERO = 2,
};

/** ExtractGlimpses
 * \ingroup CXX11_NeuralNetworks_Module
 *
 * \brief Extract glimpses from an input tensor.
 *
 * The input parameter is expected to be a col-major tensor with a rank of 4
 * (depth, x, y, and batch). The width and height parameters specify the
 * extension of the returned glimpses. The offsets parameter specifies the x, y
 * locations of the center of the glimpses relative to the center of the input
 * image. The vector is expected to contain one IndexPair for each image in the
 * batch dimension. The normalized boolean indicates if incoming coordinates are
 * normalized so that 0.0 and 1.0 correspond to the minimum and maximum of each
 * height and width dimension. The centered boolean indicates if incoming
 * coordinates are centered relative to the image, in which case -1.0 and 1.0
 * correspond to minimum and maximum of each dimension while 0.0 corresponds to
 * the center.
 *
 * The result can be assigned to a tensor of rank equal to that of the input.
 * The result will be laid out in col-major order (depth, x, y, batch). The
 * dimensions of the result will be equal to the dimensions of the input except
 * for width and height which will be equal to the requested glimpse size.
 */
namespace {

template <typename Index>
struct GlimpseExtractionOp {
  GlimpseExtractionOp(const Index width, const Index height,
                      const std::vector<IndexPair<float> >& offsets,
                      const bool normalized, const bool centered,
                      const ExtractGlimpsesNoiseMode noise, const int version)
      : width_(width),
        height_(height),
        offsets_(offsets),
        normalized_(normalized),
        centered_(centered),
        noise_(noise),
        version_(version) {}

  template <typename Input>
  DSizes<Index, 4> dimensions(const Input& input) const {
    typedef typename internal::traits<Input>::Index IndexType;
    typedef TensorRef<const Tensor<typename internal::traits<Input>::Scalar, 4,
                                   internal::traits<Input>::Layout, IndexType> >
        Ref;
    Ref in(input);

    DSizes<Index, 4> dims = in.dimensions();

    dims[0] = in.dimension(0);
    dims[1] = width_;
    dims[2] = height_;
    dims[3] = in.dimension(3);
    return dims;
  }

  template <typename Input, typename Output, typename Device>
  EIGEN_DEVICE_FUNC void eval(const Input& input, Output& output,
                              const Device& device) const {
    typedef typename internal::traits<Input>::Index IndexType;
    typedef TensorRef<const Tensor<typename internal::traits<Input>::Scalar, 4,
                                   internal::traits<Input>::Layout, IndexType> >
        Ref;
    Ref in(input);
    const Index num_channels = in.dimension(0);
    const Index input_width = in.dimension(1);
    const Index input_height = in.dimension(2);
    const Index batch_size = in.dimension(3);
    eigen_assert(input_width > 0);
    eigen_assert(input_height > 0);
    internal::NormalRandomGenerator<float> gen;
    internal::UniformRandomGenerator<float> unigen;

    for (Index i = 0; i < batch_size; ++i) {
      float x = offsets_[i].first, y = offsets_[i].second;

      if (version_ == 1) {
        // Un-normalize coordinates back to pixel space if normalized.
        if (normalized_) {
          x *= input_width;
          y *= input_height;
        }
        // Un-center if coordinates are centered on the image center.
        if (centered_) {
          x /= 2.0f;
          y /= 2.0f;
          x += input_width / 2.0f;
          y += input_height / 2.0f;
        }
        // Remove half of the glimpse window.
        x -= width_ / 2.0f;
        y -= height_ / 2.0f;
      } else {
        if (normalized_) {
          // Un-normalize coordinates back to pixel space if normalized.
          x *= input_width;
          y *= input_height;
          if (centered_) {
            // Un-center if coordinates are centered on the image center.
            x /= 2.0f;
            y /= 2.0f;
            x += input_width / 2.0f;
            y += input_height / 2.0f;
            // Remove half of the glimpse window.
            x -= width_ / 2.0f;
            y -= height_ / 2.0f;
          }
        } else {
          if (centered_) {
            x += input_width / 2.0f;
            y += input_height / 2.0f;
          }
        }
      }

      const Index offset_x = (Index)x;
      const Index offset_y = (Index)y;
      Index glimpse_width = width_;
      Index glimpse_height = height_;
      bool partial_overlap = false;
      DSizes<Index, 3> slice_offset(0, offset_x, offset_y);
      DSizes<Index, 3> slice_extent(num_channels, width_, height_);
      DSizes<Index, 3> base_offset(0, 0, 0);

      if (offset_x < 0) {
        slice_offset[1] = 0;
        glimpse_width = (std::max<Index>)(0, width_ + offset_x);
        slice_extent[1] = glimpse_width;
        base_offset[1] = width_ - glimpse_width;
        partial_overlap = true;
      } else if (offset_x + width_ >= input_width) {
        glimpse_width = (std::max<Index>)(0, input_width - offset_x);
        slice_extent[1] = glimpse_width;
        partial_overlap = true;
      }
      if (offset_y < 0) {
        slice_offset[2] = 0;
        glimpse_height = (std::max<Index>)(0, height_ + offset_y);
        slice_extent[2] = glimpse_height;
        base_offset[2] = height_ - glimpse_height;
        partial_overlap = true;
      } else if (offset_y + height_ >= input_height) {
        glimpse_height = (std::max<Index>)(0, input_height - offset_y);
        slice_extent[2] = glimpse_height;
        partial_overlap = true;
      }
      slice_extent[1] = std::min<Index>(input_width, slice_extent[1]);
      slice_extent[2] = std::min<Index>(input_height, slice_extent[2]);

      if (partial_overlap) {
        switch (noise_) {
          case ZERO: {
            // Initialize the glimpse with zero noise.
            output.template chip<3>(i).device(device) =
                output.template chip<3>(i).constant(0);
          } break;
          case UNIFORM: {
            // Initialize the glimpse with uniform noise.
            typedef std::remove_const_t<
                typename internal::traits<Input>::Scalar>
                Scalar;
            TensorFixedSize<Scalar, Sizes<> > mini;
            mini.device(device) = input.template chip<3>(i).minimum();
            TensorFixedSize<float, Sizes<> > range;
            range.device(device) = (input.template chip<3>(i).maximum() - mini)
                                       .template cast<float>();

            DSizes<Index, 3> glimpse_size(num_channels, width_, height_);
            TensorMap<Tensor<float, 3> > tmp(nullptr, glimpse_size);
            output.template chip<3>(i).device(device) =
                mini.reshape(Sizes<1, 1, 1>()).broadcast(glimpse_size) +
                (tmp.random(unigen) *
                 range.reshape(Sizes<1, 1, 1>()).broadcast(glimpse_size))
                    .template cast<Scalar>();
          } break;
          case GAUSSIAN: {
            // Initialize the glimpse with white noise: compute the mean and
            // sigma
            // of each channel, and use them to shape the gaussian.
            DSizes<Index, 2> glimpse_size(width_, height_);
            DSizes<Index, 2> input_size(input_width, input_height);
            typedef std::remove_const_t<
                typename internal::traits<Input>::Scalar>
                Scalar;

            for (int j = 0; j < num_channels; ++j) {
              TensorFixedSize<Scalar, Sizes<> > mean;
              mean.device(device) = input.template chip<3>(i)
                                        .template chip<0>(j)
                                        .template cast<float>()
                                        .mean();
              TensorFixedSize<float, Sizes<> > sigma;
              sigma.device(device) =
                  (input.template chip<3>(i)
                       .template chip<0>(j)
                       .template cast<float>() -
                   mean.reshape(Sizes<1, 1>()).broadcast(input_size))
                      .square()
                      .mean()
                      .sqrt();
              TensorFixedSize<Scalar, Sizes<> > mini;
              mini.device(device) =
                  input.template chip<3>(i).template chip<0>(j).minimum();
              TensorFixedSize<float, Sizes<> > maxi;
              maxi.device(device) =
                  input.template chip<3>(i).template chip<0>(j).maximum();

              TensorMap<Tensor<float, 2> > tmp(nullptr, glimpse_size);
              output.template chip<3>(i).template chip<0>(j).device(device) =
                  (mean.reshape(Sizes<1, 1>()).broadcast(glimpse_size) +
                   (tmp.random(gen) *
                    sigma.reshape(Sizes<1, 1>()).broadcast(glimpse_size))
                       .template cast<Scalar>())
                      .cwiseMin(
                          maxi.reshape(Sizes<1, 1>()).broadcast(glimpse_size))
                      .cwiseMax(
                          mini.reshape(Sizes<1, 1>()).broadcast(glimpse_size));
            }
          } break;
        }

        // Copy the part of the glimpse that cover the input image if any.
        if (glimpse_width == 0 || glimpse_height == 0) {
          continue;
        }
        output.template chip<3>(i)
            .slice(base_offset, slice_extent)
            .device(device) =
            input.template chip<3>(i).slice(slice_offset, slice_extent);
      } else {
        output.template chip<3>(i).device(device) =
            input.template chip<3>(i).slice(slice_offset, slice_extent);
      }
    }
  }

 private:
  const Index width_;
  const Index height_;
  const std::vector<IndexPair<float> > offsets_;
  const bool normalized_;
  const bool centered_;
  const ExtractGlimpsesNoiseMode noise_;
  const int version_;
};
}  // namespace

template <typename Input>
EIGEN_ALWAYS_INLINE static const TensorCustomUnaryOp<
    const GlimpseExtractionOp<typename internal::traits<Input>::Index>,
    const Input>
ExtractGlimpses(
    const Input& input, const typename internal::traits<Input>::Index width,
    const typename internal::traits<Input>::Index height,
    const std::vector<IndexPair<float> >& offsets, const bool normalized = true,
    const bool centered = true,
    const ExtractGlimpsesNoiseMode noise = ExtractGlimpsesNoiseMode::UNIFORM,
    const int version = 2) {
  EIGEN_STATIC_ASSERT(internal::traits<Input>::Layout == ColMajor,
                      YOU_MADE_A_PROGRAMMING_MISTAKE);
  EIGEN_STATIC_ASSERT(internal::traits<Input>::NumDimensions == 4,
                      YOU_MADE_A_PROGRAMMING_MISTAKE);

  typedef typename internal::traits<Input>::Index Index;
  const GlimpseExtractionOp<Index> op(width, height, offsets, normalized,
                                      centered, noise, version);
  return input.customOp(op);
}

}  // end namespace Eigen

#endif  // TENSORFLOW_CORE_KERNELS_EIGEN_ATTENTION_H_
