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
// See docs in ../ops/image_ops.cc.
#include <math.h>

#include <cmath>

#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/stateless_random_ops.h"
#include "tensorflow/core/lib/random/philox_random.h"
#include "tensorflow/core/lib/random/simple_philox.h"
#include "tensorflow/core/util/guarded_philox_random.h"

using tensorflow::random::SimplePhilox;

namespace tensorflow {
namespace {

// A simple Rectangle class that supplies intersection.
class Rectangle {
 public:
  Rectangle() { Set(0, 0, 0, 0); }
  Rectangle(int xmin, int ymin, int xmax, int ymax) {
    Set(xmin, ymin, xmax, ymax);
  }

  void Set(int xmin, int ymin, int xmax, int ymax) {
    min_x_ = xmin;
    min_y_ = ymin;
    max_x_ = xmax;
    max_y_ = ymax;
  }

  bool IsEmpty() const { return min_x_ > max_x_ || min_y_ > max_y_; }
  float Area() const {
    return static_cast<float>((max_x_ - min_x_) * (max_y_ - min_y_));
  }

  Rectangle Intersect(const Rectangle& r) const {
    const int pmin_x = std::max(min_x_, r.min_x_);
    const int pmin_y = std::max(min_y_, r.min_y_);
    const int pmax_x = std::min(max_x_, r.max_x_);
    const int pmax_y = std::min(max_y_, r.max_y_);

    if (pmin_x > pmax_x || pmin_y > pmax_y) {
      return Rectangle();
    } else {
      return Rectangle(pmin_x, pmin_y, pmax_x, pmax_y);
    }
  }

  int min_x_;
  int min_y_;
  int max_x_;
  int max_y_;
};

// Determine if the supplied cropping box covers a sufficient fraction of the
// the supplied bounding boxes.
bool SatisfiesOverlapConstraints(const Rectangle& crop,
                                 float minimum_object_covered,
                                 const std::vector<Rectangle>& bounding_boxes) {
  // Reject any bounding box which contains no pixels.
  const float kMinArea = 1.0;
  if (crop.Area() < kMinArea) {
    return false;
  }

  // Loop through all objects and determine if the proposed cropping box covers
  // a sufficient fraction of one of the supplied bounding boxes.
  bool is_object_covered = false;
  for (const auto& bbox : bounding_boxes) {
    const float object_area = bbox.Area();
    if (object_area < kMinArea) {
      continue;
    }

    const float object_covered = crop.Intersect(bbox).Area() / object_area;

    if (object_covered >= minimum_object_covered) {
      is_object_covered = true;
      break;
    }
  }
  return is_object_covered;
}

// Generate a random crop within the rectangle
// (0, 0, original_width, original_height).
// The minimum area of the crop will be between
//   min_relative_crop_area * orig_width * orig_height
// and
//   max_relative_crop_area * orig_width * orig_height
// such that its width = round(aspect_ratio * height).
// The diameter of the generated rectangle will be uniformly distributed between
// its minimum and maximum size. The center of the rectangle will be distributed
// uniformly within the source rectangle. The function returns false if the
// rectangle could not be generated with the given constraints.
bool GenerateRandomCrop(int original_width, int original_height,
                        float min_relative_crop_area,
                        float max_relative_crop_area, float aspect_ratio,
                        SimplePhilox* random, Rectangle* crop_rect) {
  if (max_relative_crop_area <= 0.0 || aspect_ratio <= 0.0 ||
      original_width <= 0 || original_height <= 0 ||
      min_relative_crop_area > max_relative_crop_area) {
    return false;
  }

  const float min_area =
      min_relative_crop_area * original_width * original_height;
  const float max_area =
      max_relative_crop_area * original_width * original_height;

  int height = static_cast<int>(lrintf(std::sqrt(min_area / aspect_ratio)));
  int max_height = static_cast<int>(lrintf(std::sqrt(max_area / aspect_ratio)));

  // TODO(b/140767341): Rewrite the generation logic to be more tolerant
  // of floating point behavior.
  if (lrintf(max_height * aspect_ratio) > original_width) {
    // We must find the smallest max_height satisfying
    // round(max_height * aspect_ratio) <= original_width:
    const float kEps = 0.0000001;
    max_height = static_cast<int>((original_width + 0.5 - kEps) / aspect_ratio);
    // If due some precision issues, we still cannot guarantee
    // round(max_height * aspect_ratio) <= original_width, subtract 1 from
    // max height.
    if (lrintf(max_height * aspect_ratio) > original_width) {
      max_height -= 1;
    }
  }

  if (max_height > original_height) {
    max_height = original_height;
  }

  if (height >= max_height) {
    height = max_height;
  }

  if (height < max_height) {
    // We need to generate a random number in the closed range
    // [0, max_height - height].
    height += random->Uniform(max_height - height + 1);
  }
  int width = static_cast<int>(lrintf(height * aspect_ratio));
  DCHECK_LE(width, original_width);

  // Let us not fail if rounding error causes the area to be
  // outside the constraints.
  // Try first with a slightly bigger rectangle first.
  float area = static_cast<float>(width * height);
  if (area < min_area) {
    height += 1;
    width = static_cast<int>(lrintf(height * aspect_ratio));
    area = width * height;
  }

  // Let us not fail if rounding error causes the area to be
  // outside the constraints.
  // Try first with a slightly smaller rectangle first.
  if (area > max_area) {
    height -= 1;
    width = static_cast<int>(lrintf(height * aspect_ratio));
    area = width * height;
  }

  // Now, we explored all options to rectify small rounding errors.
  // It seems the constraints can't be satisfied: return false.
  if (area < min_area || area > max_area || width > original_width ||
      height > original_height || width <= 0 || height <= 0) {
    return false;
  }

  int y = 0;
  if (height < original_height) {
    y = random->Uniform(original_height - height);
  }
  int x = 0;
  if (width < original_width) {
    x = random->Uniform(original_width - width);
  }

  crop_rect->min_x_ = x;
  crop_rect->min_y_ = y;
  crop_rect->max_x_ = x + width;
  crop_rect->max_y_ = y + height;
  return true;
}
}  // namespace

template <typename T>
class SampleDistortedBoundingBoxBaseOp : public OpKernel {
 public:
  explicit SampleDistortedBoundingBoxBaseOp(OpKernelConstruction* context)
      : OpKernel(context) {
    if (context->num_inputs() == 2) {
      OP_REQUIRES_OK(context, context->GetAttr("min_object_covered",
                                               &min_object_covered_));
      OP_REQUIRES(
          context, min_object_covered_ >= 0,
          errors::InvalidArgument("Min object covered must be non-negative: ",
                                  min_object_covered_));
    }

    OP_REQUIRES_OK(context, context->GetAttr("use_image_if_no_bounding_boxes",
                                             &use_image_if_no_bounding_boxes_));

    OP_REQUIRES_OK(
        context, context->GetAttr("aspect_ratio_range", &aspect_ratio_range_));
    OP_REQUIRES(context, aspect_ratio_range_.size() == 2,
                errors::InvalidArgument(
                    "Aspect ratio range field must specify 2 dimensions"));

    OP_REQUIRES(
        context, aspect_ratio_range_[0] > 0 && aspect_ratio_range_[1] > 0,
        errors::InvalidArgument("Aspect ratio range must be non-negative: [",
                                aspect_ratio_range_[0], ", ",
                                aspect_ratio_range_[1], "]"));

    OP_REQUIRES_OK(context, context->GetAttr("area_range", &area_range_));
    OP_REQUIRES(
        context, area_range_.size() == 2,
        errors::InvalidArgument("Area range field must specify 2 dimensions"));

    OP_REQUIRES(
        context, area_range_[0] > 0 && area_range_[1] > 0,
        errors::InvalidArgument("Area range must be non-negative: [",
                                area_range_[0], ", ", area_range_[1], "]"));

    OP_REQUIRES(context, area_range_[0] <= 1 && area_range_[1] <= 1,
                errors::InvalidArgument(
                    "Area range must be less then or equal to 1.0: [",
                    area_range_[0], ", ", area_range_[1], "]"));

    OP_REQUIRES_OK(context, context->GetAttr("max_attempts", &max_attempts_));
    OP_REQUIRES(context, max_attempts_ > 0,
                errors::InvalidArgument("Max attempts must be non-negative: ",
                                        max_attempts_));
  }

  void DoCompute(OpKernelContext* context, const random::PhiloxRandom& rng) {
    const Tensor& image_size = context->input(0);

    OP_REQUIRES(context, image_size.dims() == 1,
                errors::InvalidArgument("image_size must be 1-dimensional",
                                        image_size.shape().DebugString()));
    OP_REQUIRES(context, image_size.dim_size(0) == 3,
                errors::InvalidArgument("image_size must contain 3 elements",
                                        image_size.shape().DebugString()));

    // Note image_size_data(2) is the depth and unused.
    const uint64 height_raw = internal::SubtleMustCopy(image_size.flat<T>()(0));
    const uint64 width_raw = internal::SubtleMustCopy(image_size.flat<T>()(1));
    OP_REQUIRES(context,
                FastBoundsCheck(height_raw, std::numeric_limits<int32>::max()),
                errors::InvalidArgument("image height cannot be >= int32 max"));
    OP_REQUIRES(context,
                FastBoundsCheck(width_raw, std::numeric_limits<int32>::max()),
                errors::InvalidArgument("image width cannot be >= int32 max"));
    const int32_t height = static_cast<int32>(height_raw);
    const int32_t width = static_cast<int32>(width_raw);

    // Ensure that the supplied bounding boxes are sane and convert them to
    // Rectangles.
    const Tensor& input_boxes = context->input(1);
    OP_REQUIRES(context, input_boxes.dims() == 3,
                errors::InvalidArgument("input boxes must be 3-dimensional "
                                        "[batch, num_boxes, coords]: ",
                                        input_boxes.shape().DebugString()));
    OP_REQUIRES(context, input_boxes.dim_size(input_boxes.dims() - 1) == 4,
                errors::InvalidArgument(
                    "bounding boxes must have shape [4] or [*, 4], got ",
                    input_boxes.shape().DebugString()));

    float min_object_covered_val = 0.0;
    // `SampleDistortedBoundingBox` op accepts 2 inputs and has
    // `min_object_covered` as an attribute (handled in the constructor).
    // `SampleDistortedBoundingBoxV2` and `StatelessSampleDistortedBoundingBox`
    // ops accept 3+ inputs, including `min_object_covered`.
    if (context->num_inputs() >= 3) {
      const Tensor& min_object_covered = context->input(2);

      OP_REQUIRES(
          context, TensorShapeUtils::IsScalar(min_object_covered.shape()),
          errors::InvalidArgument("min_object_covered must be 0-D, got shape ",
                                  min_object_covered.shape().DebugString()));

      min_object_covered_val = min_object_covered.scalar<float>()();

      OP_REQUIRES(
          context, min_object_covered_val >= 0,
          errors::InvalidArgument("Min object covered must be non-negative: ",
                                  min_object_covered_val));
    } else {
      min_object_covered_val = min_object_covered_;
    }

    std::vector<Rectangle> bounding_boxes;
    if (input_boxes.NumElements() > 0) {
      TTypes<float>::ConstMatrix boxes = input_boxes.flat_inner_dims<float>();
      for (int b = 0; b < boxes.dimension(0); ++b) {
        for (int i = 0; i < 4; ++i) {
          OP_REQUIRES(
              context, boxes(b, i) >= 0.0 && boxes(b, i) <= 1.0,
              errors::InvalidArgument("All bounding box coordinates must "
                                      "be in [0.0, 1.0]: ",
                                      boxes(b, i)));
        }

        const int32_t x_min = static_cast<int32>(boxes(b, 1) * width);
        const int32_t y_min = static_cast<int32>(boxes(b, 0) * height);
        const int32_t x_max = static_cast<int32>(boxes(b, 3) * width);
        const int32_t y_max = static_cast<int32>(boxes(b, 2) * height);

        bounding_boxes.push_back(Rectangle(x_min, y_min, x_max, y_max));
      }
    }

    // Insert the entire image if no bounding boxes are supplied.
    const Rectangle image_rect(0, 0, width, height);
    if (bounding_boxes.empty()) {
      OP_REQUIRES(context, use_image_if_no_bounding_boxes_,
                  errors::InvalidArgument(
                      "No bounding boxes provided as input. One must "
                      "enable use_image_if_no_bounding_boxes if you wish "
                      "to not provide any bounding boxes."));
      bounding_boxes.push_back(image_rect);
    }

    const float min_sample_area = area_range_[0];
    const float max_sample_area = area_range_[1];
    const float min_sample_aspect_ratio = aspect_ratio_range_[0];
    const float max_sample_aspect_ratio = aspect_ratio_range_[1];

    auto local_rng = rng;
    random::SimplePhilox random(&local_rng);

    Rectangle crop_rect;
    bool sample_generated = false;
    for (int i = 0; i < max_attempts_; ++i) {
      const float sample_aspect_ratio =
          random.RandFloat() *
              (max_sample_aspect_ratio - min_sample_aspect_ratio) +
          min_sample_aspect_ratio;

      if (GenerateRandomCrop(width, height, min_sample_area, max_sample_area,
                             sample_aspect_ratio, &random, &crop_rect)) {
        if (SatisfiesOverlapConstraints(crop_rect, min_object_covered_val,
                                        bounding_boxes)) {
          sample_generated = true;
          break;
        }
      }
    }

    if (!sample_generated) {
      crop_rect = image_rect;
    }

    // Determine the cropping parameters from the bounding box.
    const int target_width = crop_rect.max_x_ - crop_rect.min_x_;
    const int target_height = crop_rect.max_y_ - crop_rect.min_y_;

    const int offset_width = crop_rect.min_x_;
    const int offset_height = crop_rect.min_y_;

    // Ensure that the bounding box fits in the image dimensions.
    OP_REQUIRES(context, width >= target_width + offset_width,
                errors::FailedPrecondition(
                    "width must be > target_width + offset_width: ", width,
                    "vs ", target_width, " + ", offset_width));
    OP_REQUIRES(context, height >= target_height + offset_height,
                errors::FailedPrecondition(
                    "height must be >= target_height: height = ", height, "vs ",
                    target_height, " + ", offset_height));

    // Create two vectors, each 3 elements, to provide as arguments to Slice.
    // See Slice() operation for details.
    Tensor* begin = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, TensorShape({3}), &begin));
    Tensor* size = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(1, TensorShape({3}), &size));
    Tensor* bboxes = nullptr;
    OP_REQUIRES_OK(
        context, context->allocate_output(2, TensorShape({1, 1, 4}), &bboxes));

    typename TTypes<T, 1>::Tensor begin_data(begin->tensor<T, 1>());
    typename TTypes<T, 1>::Tensor size_data(size->tensor<T, 1>());
    TTypes<float, 3>::Tensor bboxes_data = bboxes->tensor<float, 3>();

    begin_data(0) = T(offset_height);
    size_data(0) = T(target_height);

    begin_data(1) = T(offset_width);
    size_data(1) = T(target_width);

    bboxes_data(0, 0, 0) =
        static_cast<float>(crop_rect.min_y_) / static_cast<float>(height);
    bboxes_data(0, 0, 1) =
        static_cast<float>(crop_rect.min_x_) / static_cast<float>(width);
    bboxes_data(0, 0, 2) =
        static_cast<float>(crop_rect.max_y_) / static_cast<float>(height);
    bboxes_data(0, 0, 3) =
        static_cast<float>(crop_rect.max_x_) / static_cast<float>(width);

    // Retain all of the channels.
    begin_data(2) = T(0);
    size_data(2) = T(-1);
  }

 protected:
  int32 max_attempts_;
  std::vector<float> area_range_;
  std::vector<float> aspect_ratio_range_;
  float min_object_covered_;
  bool use_image_if_no_bounding_boxes_;
};

template <typename T>
class StatefulSampleDistortedBoundingBoxOp
    : public SampleDistortedBoundingBoxBaseOp<T> {
 public:
  explicit StatefulSampleDistortedBoundingBoxOp(OpKernelConstruction* context)
      : SampleDistortedBoundingBoxBaseOp<T>(context) {
    OP_REQUIRES_OK(context, generator_.Init(context));
  }

  void Compute(OpKernelContext* context) override {
    // Need to reserve samples since `generator_` is shared.
    this->DoCompute(context,
                    generator_.ReserveSamples32(4 * this->max_attempts_));
  }

 private:
  GuardedPhiloxRandom generator_;
};

template <typename T>
class StatelessSampleDistortedBoundingBoxOp
    : public SampleDistortedBoundingBoxBaseOp<T> {
 public:
  explicit StatelessSampleDistortedBoundingBoxOp(OpKernelConstruction* context)
      : SampleDistortedBoundingBoxBaseOp<T>(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& seed_t = context->input(3);
    OP_REQUIRES(context, seed_t.dims() == 1 && seed_t.dim_size(0) == 2,
                errors::InvalidArgument("seed must have shape [2], not ",
                                        seed_t.shape().DebugString()));

    // Create and initialize stateless random number generator (rng).
    // There is no need to `Skip` (or reserve) samples since the scope of this
    // rng is local.
    random::PhiloxRandom::Key key;
    random::PhiloxRandom::ResultType counter;
    OP_REQUIRES_OK(context, GenerateKey(seed_t, &key, &counter));

    this->DoCompute(context, random::PhiloxRandom(counter, key));
  }
};

#define REGISTER_KERNELS(type)                                        \
  REGISTER_KERNEL_BUILDER(Name("SampleDistortedBoundingBox")          \
                              .Device(DEVICE_CPU)                     \
                              .TypeConstraint<type>("T"),             \
                          StatefulSampleDistortedBoundingBoxOp<type>) \
  REGISTER_KERNEL_BUILDER(Name("SampleDistortedBoundingBoxV2")        \
                              .Device(DEVICE_CPU)                     \
                              .TypeConstraint<type>("T"),             \
                          StatefulSampleDistortedBoundingBoxOp<type>) \
  REGISTER_KERNEL_BUILDER(Name("StatelessSampleDistortedBoundingBox") \
                              .Device(DEVICE_CPU)                     \
                              .TypeConstraint<type>("T"),             \
                          StatelessSampleDistortedBoundingBoxOp<type>)

TF_CALL_INTEGRAL_TYPES(REGISTER_KERNELS);
#undef REGISTER_KERNELS

}  // namespace tensorflow
