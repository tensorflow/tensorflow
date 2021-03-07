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

// This is a helper struct to package up the input and output
// parameters of an image resizer (the height, widths, etc.).  To
// reduce code duplication and ensure consistency across the different
// resizers, it performs the input validation.

#ifndef TENSORFLOW_CORE_KERNELS_UTIL_IMAGE_RESIZER_STATE_H_
#define TENSORFLOW_CORE_KERNELS_UTIL_IMAGE_RESIZER_STATE_H_

#define EIGEN_USE_THREADS
#include <math.h>

#include <algorithm>
#include <array>

#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace tensorflow {

// CalculateResizeScale determines the float scaling factor.
inline float CalculateResizeScale(int64 in_size, int64 out_size,
                                  bool align_corners) {
  return (align_corners && out_size > 1)
             ? (in_size - 1) / static_cast<float>(out_size - 1)
             : in_size / static_cast<float>(out_size);
}

// Half pixel scaler scales assuming that the pixel centers are at 0.5, i.e. the
// floating point coordinates of the top,left pixel is 0.5,0.5.
struct HalfPixelScaler {
  HalfPixelScaler(){};
  inline float operator()(const int x, const float scale) const {
    // Note that we subtract 0.5 from the return value, as the existing bilinear
    // sampling code etc assumes pixels are in the old coordinate system.
    return (static_cast<float>(x) + 0.5f) * scale - 0.5f;
  }
};

// Older incorrect scaling method that causes all resizes to have a slight
// translation leading to inconsistent results. For example, a flip then a
// resize gives different results then a resize then a flip.
struct LegacyScaler {
  LegacyScaler(){};
  inline float operator()(const int x, const float scale) const {
    return static_cast<float>(x) * scale;
  }
};

// The struct storing the data related to image resizing.
struct ImageResizerStateData {
  int64 batch_size;
  int64 out_height;
  int64 out_width;
  int64 in_height;
  int64 in_width;
  int64 channels;
  float height_scale;
  float width_scale;

  Tensor* output;
};

class ImageResizerStateImpl;
class ImageResizerStateBase {
 public:
  ImageResizerStateBase(bool align_corners, bool half_pixel_center,
                        bool need_swap);
  ImageResizerStateBase(const ImageResizerStateBase&) = delete;
  ImageResizerStateBase& operator=(const ImageResizerStateBase&) = delete;
  virtual ~ImageResizerStateBase();

  // Calculates all the required variables, and allocates the output.
  void ValidateAndCreateOutput(OpKernelContext* context,
                               const TensorShape& input_shape,
                               const TensorShape& output_shape);

  // Returns the data related to image resizing.
  const ImageResizerStateData& GetData() const;

 private:
  std::unique_ptr<ImageResizerStateImpl> impl_;
};

class ImageResizerState : public ImageResizerStateBase {
 public:
  ImageResizerState(bool align_corners, bool half_pixel_centers);
  ~ImageResizerState() override;
};

class ImageResizerGradientState : public ImageResizerStateBase {
 public:
  explicit ImageResizerGradientState(bool align_corners,
                                     bool half_pixel_centers);
  ~ImageResizerGradientState() override;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_UTIL_IMAGE_RESIZER_STATE_H_
