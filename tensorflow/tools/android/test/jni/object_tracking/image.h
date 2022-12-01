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

#ifndef TENSORFLOW_TOOLS_ANDROID_TEST_JNI_OBJECT_TRACKING_IMAGE_H_
#define TENSORFLOW_TOOLS_ANDROID_TEST_JNI_OBJECT_TRACKING_IMAGE_H_

#include <stdint.h>

#include "tensorflow/tools/android/test/jni/object_tracking/geom.h"
#include "tensorflow/tools/android/test/jni/object_tracking/utils.h"

// TODO(andrewharp): Make this a cast to uint32_t if/when we go unsigned for
// operations.
#define ZERO 0

#ifdef SANITY_CHECKS
  #define CHECK_PIXEL(IMAGE, X, Y) {\
    SCHECK((IMAGE)->ValidPixel((X), (Y)), \
          "CHECK_PIXEL(%d,%d) in %dx%d image.", \
          static_cast<int>(X), static_cast<int>(Y), \
          (IMAGE)->GetWidth(), (IMAGE)->GetHeight());\
  }

  #define CHECK_PIXEL_INTERP(IMAGE, X, Y) {\
    SCHECK((IMAGE)->validInterpPixel((X), (Y)), \
          "CHECK_PIXEL_INTERP(%.2f, %.2f) in %dx%d image.", \
          static_cast<float>(X), static_cast<float>(Y), \
          (IMAGE)->GetWidth(), (IMAGE)->GetHeight());\
  }
#else
  #define CHECK_PIXEL(image, x, y) {}
  #define CHECK_PIXEL_INTERP(IMAGE, X, Y) {}
#endif

namespace tf_tracking {

#ifdef SANITY_CHECKS
// Class which exists solely to provide bounds checking for array-style image
// data access.
template <typename T>
class RowData {
 public:
  RowData(T* const row_data, const int max_col)
      : row_data_(row_data), max_col_(max_col) {}

  inline T& operator[](const int col) const {
    SCHECK(InRange(col, 0, max_col_),
          "Column out of range: %d (%d max)", col, max_col_);
    return row_data_[col];
  }

  inline operator T*() const {
    return row_data_;
  }

 private:
  T* const row_data_;
  const int max_col_;
};
#endif

// Naive templated sorting function.
template <typename T>
int Comp(const void* a, const void* b) {
  const T val1 = *reinterpret_cast<const T*>(a);
  const T val2 = *reinterpret_cast<const T*>(b);

  if (val1 == val2) {
    return 0;
  } else if (val1 < val2) {
    return -1;
  } else {
    return 1;
  }
}

// TODO(andrewharp): Make explicit which operations support negative numbers or
// struct/class types in image data (possibly create fast multi-dim array class
// for data where pixel arithmetic does not make sense).

// Image class optimized for working on numeric arrays as grayscale image data.
// Supports other data types as a 2D array class, so long as no pixel math
// operations are called (convolution, downsampling, etc).
template <typename T>
class Image {
 public:
  Image(const int width, const int height);
  explicit Image(const Size& size);

  // Constructor that creates an image from preallocated data.
  // Note: The image takes ownership of the data lifecycle, unless own_data is
  // set to false.
  Image(const int width, const int height, T* const image_data,
        const bool own_data = true);

  ~Image();

  // Extract a pixel patch from this image, starting at a subpixel location.
  // Uses 16:16 fixed point format for representing real values and doing the
  // bilinear interpolation.
  //
  // Arguments fp_x and fp_y tell the subpixel position in fixed point format,
  // patchwidth/patchheight give the size of the patch in pixels and
  // to_data must be a valid pointer to a *contiguous* destination data array.
  template<class DstType>
  bool ExtractPatchAtSubpixelFixed1616(const int fp_x,
                                       const int fp_y,
                                       const int patchwidth,
                                       const int patchheight,
                                       DstType* to_data) const;

  Image<T>* Crop(
      const int left, const int top, const int right, const int bottom) const;

  inline int GetWidth() const { return width_; }
  inline int GetHeight() const { return height_; }

  // Bilinearly sample a value between pixels.  Values must be within the image.
  inline float GetPixelInterp(const float x, const float y) const;

  // Bilinearly sample a pixels at a subpixel position using fixed point
  // arithmetic.
  // Avoids float<->int conversions.
  // Values must be within the image.
  // Arguments fp_x and fp_y tell the subpixel position in
  // 16:16 fixed point format.
  //
  // Important: This function only makes sense for integer-valued images, such
  // as Image<uint8_t> or Image<int> etc.
  inline T GetPixelInterpFixed1616(const int fp_x_whole,
                                   const int fp_y_whole) const;

  // Returns true iff the pixel is in the image's boundaries.
  inline bool ValidPixel(const int x, const int y) const;

  inline BoundingBox GetContainingBox() const;

  inline bool Contains(const BoundingBox& bounding_box) const;

  inline T GetMedianValue() {
    qsort(image_data_, data_size_, sizeof(image_data_[0]), Comp<T>);
    return image_data_[data_size_ >> 1];
  }

  // Returns true iff the pixel is in the image's boundaries for interpolation
  // purposes.
  // TODO(andrewharp): check in interpolation follow-up change.
  inline bool ValidInterpPixel(const float x, const float y) const;

  // Safe lookup with boundary enforcement.
  inline T GetPixelClipped(const int x, const int y) const {
    return (*this)[Clip(y, ZERO, height_less_one_)]
                  [Clip(x, ZERO, width_less_one_)];
  }

#ifdef SANITY_CHECKS
  inline RowData<T> operator[](const int row) {
    SCHECK(InRange(row, 0, height_less_one_),
          "Row out of range: %d (%d max)", row, height_less_one_);
    return RowData<T>(image_data_ + row * stride_, width_less_one_);
  }

  inline const RowData<T> operator[](const int row) const {
    SCHECK(InRange(row, 0, height_less_one_),
          "Row out of range: %d (%d max)", row, height_less_one_);
    return RowData<T>(image_data_ + row * stride_, width_less_one_);
  }
#else
  inline T* operator[](const int row) {
    return image_data_ + row * stride_;
  }

  inline const T* operator[](const int row) const {
    return image_data_ + row * stride_;
  }
#endif

  const T* data() const { return image_data_; }

  inline int stride() const { return stride_; }

  // Clears image to a single value.
  inline void Clear(const T& val) {
    memset(image_data_, val, sizeof(*image_data_) * data_size_);
  }

#ifdef __ARM_NEON
  void Downsample2x32ColumnsNeon(const uint8_t* const original,
                                 const int stride, const int orig_x);

  void Downsample4x32ColumnsNeon(const uint8_t* const original,
                                 const int stride, const int orig_x);

  void DownsampleAveragedNeon(const uint8_t* const original, const int stride,
                              const int factor);
#endif

  // Naive downsampler that reduces image size by factor by averaging pixels in
  // blocks of size factor x factor.
  void DownsampleAveraged(const T* const original, const int stride,
                          const int factor);

  // Naive downsampler that reduces image size by factor by averaging pixels in
  // blocks of size factor x factor.
  inline void DownsampleAveraged(const Image<T>& original, const int factor) {
    DownsampleAveraged(original.data(), original.GetWidth(), factor);
  }

  // Native downsampler that reduces image size using nearest interpolation
  void DownsampleInterpolateNearest(const Image<T>& original);

  // Native downsampler that reduces image size using fixed-point bilinear
  // interpolation
  void DownsampleInterpolateLinear(const Image<T>& original);

  // Relatively efficient downsampling of an image by a factor of two with a
  // low-pass 3x3 smoothing operation thrown in.
  void DownsampleSmoothed3x3(const Image<T>& original);

  // Relatively efficient downsampling of an image by a factor of two with a
  // low-pass 5x5 smoothing operation thrown in.
  void DownsampleSmoothed5x5(const Image<T>& original);

  // Optimized Scharr filter on a single pixel in the X direction.
  // Scharr filters are like central-difference operators, but have more
  // rotational symmetry in their response because they also consider the
  // diagonal neighbors.
  template <typename U>
  inline T ScharrPixelX(const Image<U>& original,
                        const int center_x, const int center_y) const;

  // Optimized Scharr filter on a single pixel in the X direction.
  // Scharr filters are like central-difference operators, but have more
  // rotational symmetry in their response because they also consider the
  // diagonal neighbors.
  template <typename U>
  inline T ScharrPixelY(const Image<U>& original,
                        const int center_x, const int center_y) const;

  // Convolve the image with a Scharr filter in the X direction.
  // Much faster than an equivalent generic convolution.
  template <typename U>
  inline void ScharrX(const Image<U>& original);

  // Convolve the image with a Scharr filter in the Y direction.
  // Much faster than an equivalent generic convolution.
  template <typename U>
  inline void ScharrY(const Image<U>& original);

  static inline T HalfDiff(int32_t first, int32_t second) {
    return (second - first) / 2;
  }

  template <typename U>
  void DerivativeX(const Image<U>& original);

  template <typename U>
  void DerivativeY(const Image<U>& original);

  // Generic function for convolving pixel with 3x3 filter.
  // Filter pixels should be in row major order.
  template <typename U>
  inline T ConvolvePixel3x3(const Image<U>& original,
                            const int* const filter,
                            const int center_x, const int center_y,
                            const int total) const;

  // Generic function for convolving an image with a 3x3 filter.
  // TODO(andrewharp): Generalize this for any size filter.
  template <typename U>
  inline void Convolve3x3(const Image<U>& original,
                          const int32_t* const filter);

  // Load this image's data from a data array. The data at pixels is assumed to
  // have dimensions equivalent to this image's dimensions * factor.
  inline void FromArray(const T* const pixels, const int stride,
                        const int factor = 1);

  // Copy the image back out to an appropriately sized data array.
  inline void ToArray(T* const pixels) const {
    // If not subsampling, memcpy should be faster.
    memcpy(pixels, this->image_data_, data_size_ * sizeof(T));
  }

  // Precompute these for efficiency's sake as they're used by a lot of
  // clipping code and loop code.
  // TODO(andrewharp): make these only accessible by other Images.
  const int width_less_one_;
  const int height_less_one_;

  // The raw size of the allocated data.
  const int data_size_;

 private:
  inline void Allocate() {
    image_data_ = new T[data_size_];
    if (image_data_ == NULL) {
      LOGE("Couldn't allocate image data!");
    }
  }

  T* image_data_;

  bool own_data_;

  const int width_;
  const int height_;

  // The image stride (offset to next row).
  // TODO(andrewharp): Make sure that stride is honored in all code.
  const int stride_;

  TF_DISALLOW_COPY_AND_ASSIGN(Image);
};

template <typename t>
inline std::ostream& operator<<(std::ostream& stream, const Image<t>& image) {
  for (int y = 0; y < image.GetHeight(); ++y) {
    for (int x = 0; x < image.GetWidth(); ++x) {
      stream << image[y][x] << " ";
    }
    stream << std::endl;
  }
  return stream;
}

}  // namespace tf_tracking

#endif  // TENSORFLOW_TOOLS_ANDROID_TEST_JNI_OBJECT_TRACKING_IMAGE_H_
