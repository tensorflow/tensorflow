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

#ifndef THIRD_PARTY_TENSORFLOW_EXAMPLES_ANDROID_JNI_OBJECT_TRACKING_IMAGE_DATA_H_
#define THIRD_PARTY_TENSORFLOW_EXAMPLES_ANDROID_JNI_OBJECT_TRACKING_IMAGE_DATA_H_

#include <memory>

#include "tensorflow/core/platform/types.h"

#include "tensorflow/examples/android/jni/object_tracking/image-inl.h"
#include "tensorflow/examples/android/jni/object_tracking/image.h"
#include "tensorflow/examples/android/jni/object_tracking/image_utils.h"
#include "tensorflow/examples/android/jni/object_tracking/integral_image.h"
#include "tensorflow/examples/android/jni/object_tracking/time_log.h"
#include "tensorflow/examples/android/jni/object_tracking/utils.h"

#include "tensorflow/examples/android/jni/object_tracking/config.h"

using namespace tensorflow;

namespace tf_tracking {

// Class that encapsulates all bulky processed data for a frame.
class ImageData {
 public:
  explicit ImageData(const int width, const int height)
      : uv_frame_width_(width << 1),
        uv_frame_height_(height << 1),
        timestamp_(0),
        image_(width, height) {
    InitPyramid(width, height);
    ResetComputationCache();
  }

 private:
  void ResetComputationCache() {
    uv_data_computed_ = false;
    integral_image_computed_ = false;
    for (int i = 0; i < kNumPyramidLevels; ++i) {
      spatial_x_computed_[i] = false;
      spatial_y_computed_[i] = false;
      pyramid_sqrt2_computed_[i * 2] = false;
      pyramid_sqrt2_computed_[i * 2 + 1] = false;
    }
  }

  void InitPyramid(const int width, const int height) {
    int level_width = width;
    int level_height = height;

    for (int i = 0; i < kNumPyramidLevels; ++i) {
      pyramid_sqrt2_[i * 2] = NULL;
      pyramid_sqrt2_[i * 2 + 1] = NULL;
      spatial_x_[i] = NULL;
      spatial_y_[i] = NULL;

      level_width /= 2;
      level_height /= 2;
    }

    // Alias the first pyramid level to image_.
    pyramid_sqrt2_[0] = &image_;
  }

 public:
  ~ImageData() {
    // The first pyramid level is actually an alias to image_,
    // so make sure it doesn't get deleted here.
    pyramid_sqrt2_[0] = NULL;

    for (int i = 0; i < kNumPyramidLevels; ++i) {
      SAFE_DELETE(pyramid_sqrt2_[i * 2]);
      SAFE_DELETE(pyramid_sqrt2_[i * 2 + 1]);
      SAFE_DELETE(spatial_x_[i]);
      SAFE_DELETE(spatial_y_[i]);
    }
  }

  void SetData(const uint8* const new_frame, const int stride,
               const int64 timestamp, const int downsample_factor) {
    SetData(new_frame, NULL, stride, timestamp, downsample_factor);
  }

  void SetData(const uint8* const new_frame,
               const uint8* const uv_frame,
               const int stride,
               const int64 timestamp, const int downsample_factor) {
    ResetComputationCache();

    timestamp_ = timestamp;

    TimeLog("SetData!");

    pyramid_sqrt2_[0]->FromArray(new_frame, stride, downsample_factor);
    pyramid_sqrt2_computed_[0] = true;
    TimeLog("Downsampled image");

    if (uv_frame != NULL) {
      if (u_data_.get() == NULL) {
        u_data_.reset(new Image<uint8>(uv_frame_width_, uv_frame_height_));
        v_data_.reset(new Image<uint8>(uv_frame_width_, uv_frame_height_));
      }

      GetUV(uv_frame, u_data_.get(), v_data_.get());
      uv_data_computed_ = true;
      TimeLog("Copied UV data");
    } else {
      LOGV("No uv data!");
    }

#ifdef LOG_TIME
    // If profiling is enabled, precompute here to make it easier to distinguish
    // total costs.
    Precompute();
#endif
  }

  inline const uint64 GetTimestamp() const {
    return timestamp_;
  }

  inline const Image<uint8>* GetImage() const {
    SCHECK(pyramid_sqrt2_computed_[0], "image not set!");
    return pyramid_sqrt2_[0];
  }

  const Image<uint8>* GetPyramidSqrt2Level(const int level) const {
    if (!pyramid_sqrt2_computed_[level]) {
      SCHECK(level != 0, "Level equals 0!");
      if (level == 1) {
        const Image<uint8>& upper_level = *GetPyramidSqrt2Level(0);
        if (pyramid_sqrt2_[level] == NULL) {
          const int new_width =
              (static_cast<int>(upper_level.GetWidth() / sqrtf(2)) + 1) / 2 * 2;
          const int new_height =
              (static_cast<int>(upper_level.GetHeight() / sqrtf(2)) + 1) / 2 *
              2;

          pyramid_sqrt2_[level] = new Image<uint8>(new_width, new_height);
        }
        pyramid_sqrt2_[level]->DownsampleInterpolateLinear(upper_level);
      } else {
        const Image<uint8>& upper_level = *GetPyramidSqrt2Level(level - 2);
        if (pyramid_sqrt2_[level] == NULL) {
          pyramid_sqrt2_[level] = new Image<uint8>(
              upper_level.GetWidth() / 2, upper_level.GetHeight() / 2);
        }
        pyramid_sqrt2_[level]->DownsampleAveraged(
            upper_level.data(), upper_level.stride(), 2);
      }
      pyramid_sqrt2_computed_[level] = true;
    }
    return pyramid_sqrt2_[level];
  }

  inline const Image<int32>* GetSpatialX(const int level) const {
    if (!spatial_x_computed_[level]) {
      const Image<uint8>& src = *GetPyramidSqrt2Level(level * 2);
      if (spatial_x_[level] == NULL) {
        spatial_x_[level] = new Image<int32>(src.GetWidth(), src.GetHeight());
      }
      spatial_x_[level]->DerivativeX(src);
      spatial_x_computed_[level] = true;
    }
    return spatial_x_[level];
  }

  inline const Image<int32>* GetSpatialY(const int level) const {
    if (!spatial_y_computed_[level]) {
      const Image<uint8>& src = *GetPyramidSqrt2Level(level * 2);
      if (spatial_y_[level] == NULL) {
        spatial_y_[level] = new Image<int32>(src.GetWidth(), src.GetHeight());
      }
      spatial_y_[level]->DerivativeY(src);
      spatial_y_computed_[level] = true;
    }
    return spatial_y_[level];
  }

  // The integral image is currently only used for object detection, so lazily
  // initialize it on request.
  inline const IntegralImage* GetIntegralImage() const {
    if (integral_image_.get() == NULL) {
      integral_image_.reset(new IntegralImage(image_));
    } else if (!integral_image_computed_) {
      integral_image_->Recompute(image_);
    }
    integral_image_computed_ = true;
    return integral_image_.get();
  }

  inline const Image<uint8>* GetU() const {
    SCHECK(uv_data_computed_, "UV data not provided!");
    return u_data_.get();
  }

  inline const Image<uint8>* GetV() const {
    SCHECK(uv_data_computed_, "UV data not provided!");
    return v_data_.get();
  }

 private:
  void Precompute() {
    // Create the smoothed pyramids.
    for (int i = 0; i < kNumPyramidLevels * 2; i += 2) {
      (void) GetPyramidSqrt2Level(i);
    }
    TimeLog("Created smoothed pyramids");

    // Create the smoothed pyramids.
    for (int i = 1; i < kNumPyramidLevels * 2; i += 2) {
      (void) GetPyramidSqrt2Level(i);
    }
    TimeLog("Created smoothed sqrt pyramids");

    // Create the spatial derivatives for frame 1.
    for (int i = 0; i < kNumPyramidLevels; ++i) {
      (void) GetSpatialX(i);
      (void) GetSpatialY(i);
    }
    TimeLog("Created spatial derivatives");

    (void) GetIntegralImage();
    TimeLog("Got integral image!");
  }

  const int uv_frame_width_;
  const int uv_frame_height_;

  int64 timestamp_;

  Image<uint8> image_;

  bool uv_data_computed_;
  std::unique_ptr<Image<uint8> > u_data_;
  std::unique_ptr<Image<uint8> > v_data_;

  mutable bool spatial_x_computed_[kNumPyramidLevels];
  mutable Image<int32>* spatial_x_[kNumPyramidLevels];

  mutable bool spatial_y_computed_[kNumPyramidLevels];
  mutable Image<int32>* spatial_y_[kNumPyramidLevels];

  // Mutable so the lazy initialization can work when this class is const.
  // Whether or not the integral image has been computed for the current image.
  mutable bool integral_image_computed_;
  mutable std::unique_ptr<IntegralImage> integral_image_;

  mutable bool pyramid_sqrt2_computed_[kNumPyramidLevels * 2];
  mutable Image<uint8>* pyramid_sqrt2_[kNumPyramidLevels * 2];

  TF_DISALLOW_COPY_AND_ASSIGN(ImageData);
};

}  // namespace tf_tracking

#endif  // THIRD_PARTY_TENSORFLOW_EXAMPLES_ANDROID_JNI_OBJECT_TRACKING_IMAGE_DATA_H_
