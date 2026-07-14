/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_TOOLS_EVALUATION_STAGES_UTILS_IMAGE_METRICS_H_
#define TENSORFLOW_LITE_TOOLS_EVALUATION_STAGES_UTILS_IMAGE_METRICS_H_

#include <stdint.h>

#include <vector>

namespace tflite {
namespace evaluation {
namespace image {

struct Box2D {
  struct Interval {
    float min = 0;
    float max = 0;
    Interval(float x, float y) {
      min = x;
      max = y;
    }
    Interval() {}
  };

  Interval x;
  Interval y;
  static float Length(const Interval& a);
  static float Intersection(const Interval& a, const Interval& b);
  float Area() const;
  float Intersection(const Box2D& other) const;
  float Union(const Box2D& other) const;
  // Intersection of this box and the given box normalized over the union of
  // this box and the given box.
  float IoU(const Box2D& other) const;
  // Intersection of this box and the given box normalized over the area of
  // this box.
  float Overlap(const Box2D& other) const;
};

// If the value is:
//   - kDontIgnore: The object is included in this evaluation.
//   - kIgnoreOneMatch: the first matched prediction bbox will be ignored. This
//      is useful when this groundtruth object is not intended to be evaluated.
//   - kIgnoreAllMatches: all matched prediction bbox will be ignored. Typically
//      it is used to mark an area that has not been labeled.
enum IgnoreType {
  kDontIgnore = 0,
  kIgnoreOneMatch = 1,
  kIgnoreAllMatches = 2,
};

struct Detection {
 public:
  bool difficult = false;
  int64_t imgid = 0;
  float score = 0;
  Box2D box;
  IgnoreType ignore = IgnoreType::kDontIgnore;

  Detection() {}
  Detection(bool d, int64_t id, float s, Box2D b)
      : difficult(d), imgid(id), score(s), box(b) {}
  Detection(bool d, int64_t id, float s, Box2D b, IgnoreType i)
      : difficult(d), imgid(id), score(s), box(b), ignore(i) {}
};

// Precision and recall.
struct PR {
  float p = 0;
  float r = 0;
  PR(const float p_, const float r_) : p(p_), r(r_) {}
};

class AveragePrecision {
 public:
  // iou_threshold: A predicted box matches a ground truth box if and only if
  //   IoU between these two are larger than this iou_threshold. Default: 0.5.
  // num_recall_points: AP is computed as the average of maximum precision at (1
  //   + num_recall_points) recall levels. E.g., if num_recall_points is 10,
  //   recall levels are 0., 0.1, 0.2, ..., 0.9, 1.0.
  // Default: 100. If num_recall_points < 0, AveragePrecision of 0 is returned.
  struct Options {
    float iou_threshold = 0.5;
    int num_recall_points = 100;
  };
  AveragePrecision() : AveragePrecision(Options()) {}
  explicit AveragePrecision(const Options& opts) : opts_(opts) {}

  // Given a sequence of precision-recall points ordered by the recall in
  // non-increasing order, returns the average of maximum precisions at
  // different recall values (0.0, 0.1, 0.2, ..., 0.9, 1.0).
  // The p-r pairs at these fixed recall points will be written to pr_out, if
  // it is not null_ptr.
  float FromPRCurve(const std::vector<PR>& pr,
                    std::vector<PR>* pr_out = nullptr);

  // An axis aligned bounding box for an image with id 'imageid'.  Score
  // indicates its confidence.
  //
  // 'difficult' is a special bit specific to Pascal VOC dataset and tasks using
  // the data. If 'difficult' is true, by convention, the box is often ignored
  // during the AP calculation. I.e., if a predicted box matches a 'difficult'
  // ground box, this predicted box is ignored as if the model does not make
  // such a prediction.

  // Given the set of ground truth boxes and a set of predicted boxes, returns
  // the average of the maximum precisions at different recall values.
  float FromBoxes(const std::vector<Detection>& groundtruth,
                  const std::vector<Detection>& prediction,
                  std::vector<PR>* pr_out = nullptr);

 private:
  Options opts_;
};

}  // namespace image
}  // namespace evaluation
}  // namespace tflite

#endif  // TENSORFLOW_LITE_TOOLS_EVALUATION_STAGES_UTILS_IMAGE_METRICS_H_
