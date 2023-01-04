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

// NOTE: no native object detectors are currently provided or used by the code
// in this directory. This class remains mainly for historical reasons.
// Detection in the TF demo is done through TensorFlowMultiBoxDetector.java.

// Defines the ObjectDetector class that is the main interface for detecting
// ObjectModelBases in frames.

#ifndef TENSORFLOW_TOOLS_ANDROID_TEST_JNI_OBJECT_TRACKING_OBJECT_DETECTOR_H_
#define TENSORFLOW_TOOLS_ANDROID_TEST_JNI_OBJECT_TRACKING_OBJECT_DETECTOR_H_

#include <float.h>

#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "tensorflow/tools/android/test/jni/object_tracking/geom.h"
#include "tensorflow/tools/android/test/jni/object_tracking/image-inl.h"
#include "tensorflow/tools/android/test/jni/object_tracking/image.h"
#include "tensorflow/tools/android/test/jni/object_tracking/integral_image.h"
#ifdef __RENDER_OPENGL__
#include "tensorflow/tools/android/test/jni/object_tracking/sprite.h"
#endif
#include "tensorflow/tools/android/test/jni/object_tracking/config.h"
#include "tensorflow/tools/android/test/jni/object_tracking/image_data.h"
#include "tensorflow/tools/android/test/jni/object_tracking/object_model.h"
#include "tensorflow/tools/android/test/jni/object_tracking/utils.h"

namespace tf_tracking {

// Adds BoundingSquares to a vector such that the first square added is centered
// in the position given and of square_size, and the remaining squares are added
// concentrentically, scaling down by scale_factor until the minimum threshold
// size is passed.
// Squares that do not fall completely within image_bounds will not be added.
static inline void FillWithSquares(
    const BoundingBox& image_bounds,
    const BoundingBox& position,
    const float starting_square_size,
    const float smallest_square_size,
    const float scale_factor,
    std::vector<BoundingSquare>* const squares) {
  BoundingSquare descriptor_area =
      GetCenteredSquare(position, starting_square_size);

  SCHECK(scale_factor < 1.0f, "Scale factor too large at %.2f!", scale_factor);

  // Use a do/while loop to ensure that at least one descriptor is created.
  do {
    if (image_bounds.Contains(descriptor_area.ToBoundingBox())) {
      squares->push_back(descriptor_area);
    }
    descriptor_area.Scale(scale_factor);
  } while (descriptor_area.size_ >= smallest_square_size - EPSILON);
  LOGV("Created %zu squares starting from size %.2f to min size %.2f "
       "using scale factor: %.2f",
       squares->size(), starting_square_size, smallest_square_size,
       scale_factor);
}


// Represents a potential detection of a specific ObjectExemplar and Descriptor
// at a specific position in the image.
class Detection {
 public:
  explicit Detection(const ObjectModelBase* const object_model,
                     const MatchScore match_score,
                     const BoundingBox& bounding_box)
      : object_model_(object_model),
        match_score_(match_score),
        bounding_box_(bounding_box) {}

  Detection(const Detection& other)
      : object_model_(other.object_model_),
        match_score_(other.match_score_),
        bounding_box_(other.bounding_box_) {}

  virtual ~Detection() {}

  inline BoundingBox GetObjectBoundingBox() const {
    return bounding_box_;
  }

  inline MatchScore GetMatchScore() const {
    return match_score_;
  }

  inline const ObjectModelBase* GetObjectModel() const {
    return object_model_;
  }

  inline bool Intersects(const Detection& other) {
    // Check if any of the four axes separates us, there must be at least one.
    return bounding_box_.Intersects(other.bounding_box_);
  }

  struct Comp {
    inline bool operator()(const Detection& a, const Detection& b) const {
      return a.match_score_ > b.match_score_;
    }
  };

  // TODO(andrewharp): add accessors to update these instead.
  const ObjectModelBase* object_model_;
  MatchScore match_score_;
  BoundingBox bounding_box_;
};

inline std::ostream& operator<<(std::ostream& stream,
                                const Detection& detection) {
  const BoundingBox actual_area = detection.GetObjectBoundingBox();
  stream << actual_area;
  return stream;
}

class ObjectDetectorBase {
 public:
  explicit ObjectDetectorBase(const ObjectDetectorConfig* const config)
      : config_(config),
        image_data_(NULL) {}

  virtual ~ObjectDetectorBase();

  // Sets the current image data. All calls to ObjectDetector other than
  // FillDescriptors use the image data last set.
  inline void SetImageData(const ImageData* const image_data) {
    image_data_ = image_data;
  }

  // Main entry point into the detection algorithm.
  // Scans the frame for candidates, tweaks them, and fills in the
  // given std::vector of Detection objects with acceptable matches.
  virtual void Detect(const std::vector<BoundingSquare>& positions,
                      std::vector<Detection>* const detections) const = 0;

  virtual ObjectModelBase* CreateObjectModel(const std::string& name) = 0;

  virtual void DeleteObjectModel(const std::string& name) = 0;

  virtual void GetObjectModels(
      std::vector<const ObjectModelBase*>* models) const = 0;

  // Creates a new ObjectExemplar from the given position in the context of
  // the last frame passed to NextFrame.
  // Will return null in the case that there's no room for a descriptor to be
  // created in the example area, or the example area is not completely
  // contained within the frame.
  virtual void UpdateModel(const Image<uint8_t>& base_image,
                           const IntegralImage& integral_image,
                           const BoundingBox& bounding_box, const bool locked,
                           ObjectModelBase* model) const = 0;

  virtual void Draw() const = 0;

  virtual bool AllowSpontaneousDetections() = 0;

 protected:
  const std::unique_ptr<const ObjectDetectorConfig> config_;

  // The latest frame data, upon which all detections will be performed.
  // Not owned by this object, just provided for reference by ObjectTracker
  // via SetImageData().
  const ImageData* image_data_;

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(ObjectDetectorBase);
};

template <typename ModelType>
class ObjectDetector : public ObjectDetectorBase {
 public:
  explicit ObjectDetector(const ObjectDetectorConfig* const config)
      : ObjectDetectorBase(config) {}

  virtual ~ObjectDetector() {
    typename std::map<std::string, ModelType*>::const_iterator it =
        object_models_.begin();
    for (; it != object_models_.end(); ++it) {
      ModelType* model = it->second;
      delete model;
    }
  }

  virtual void DeleteObjectModel(const std::string& name) {
    ModelType* model = object_models_[name];
    CHECK_ALWAYS(model != NULL, "Model was null!");
    object_models_.erase(name);
    SAFE_DELETE(model);
  }

  virtual void GetObjectModels(
      std::vector<const ObjectModelBase*>* models) const {
    typename std::map<std::string, ModelType*>::const_iterator it =
        object_models_.begin();
    for (; it != object_models_.end(); ++it) {
      models->push_back(it->second);
    }
  }

  virtual bool AllowSpontaneousDetections() {
    return false;
  }

 protected:
  std::map<std::string, ModelType*> object_models_;

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(ObjectDetector);
};

}  // namespace tf_tracking

#endif  // TENSORFLOW_TOOLS_ANDROID_TEST_JNI_OBJECT_TRACKING_OBJECT_DETECTOR_H_
