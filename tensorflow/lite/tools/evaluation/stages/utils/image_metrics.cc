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
#include "tensorflow/lite/tools/evaluation/stages/utils/image_metrics.h"

#include <algorithm>
#include <cmath>

#include "absl/container/flat_hash_map.h"
#include "tensorflow/core/platform/logging.h"

namespace tflite {
namespace evaluation {
namespace image {

float Box2D::Length(const Box2D::Interval& a) {
  return std::max(0.f, a.max - a.min);
}

float Box2D::Intersection(const Box2D::Interval& a, const Box2D::Interval& b) {
  return Length(Interval{std::max(a.min, b.min), std::min(a.max, b.max)});
}

float Box2D::Area() const { return Length(x) * Length(y); }

float Box2D::Intersection(const Box2D& other) const {
  return Intersection(x, other.x) * Intersection(y, other.y);
}

float Box2D::Union(const Box2D& other) const {
  return Area() + other.Area() - Intersection(other);
}

float Box2D::IoU(const Box2D& other) const {
  const float total = Union(other);
  if (total > 0) {
    return Intersection(other) / total;
  } else {
    return 0.0;
  }
}

float Box2D::Overlap(const Box2D& other) const {
  const float intersection = Intersection(other);
  return intersection > 0 ? intersection / Area() : 0.0;
}

float AveragePrecision::FromPRCurve(const std::vector<PR>& pr,
                                    std::vector<PR>* pr_out) {
  // Because pr[...] are ordered by recall, iterate backward to compute max
  // precision. p(r) = max_{r' >= r} p(r') for r in 0.0, 0.1, 0.2, ..., 0.9,
  // 1.0. Then, take the average of (num_recal_points) quantities.
  float p = 0;
  float sum = 0;
  int r_level = opts_.num_recall_points;
  for (int i = pr.size() - 1; i >= 0; --i) {
    const PR& item = pr[i];
    if (i > 0) {
      if (item.r < pr[i - 1].r) {
        LOG(ERROR) << "recall points are not in order: " << pr[i - 1].r << ", "
                   << item.r;
        return 0;
      }
    }

    // Because r takes values opts_.num_recall_points, opts_.num_recall_points -
    // 1, ..., 0, the following condition is checking whether item.r crosses r /
    // opts_.num_recall_points. I.e., 1.0, 0.90, ..., 0.01, 0.0.  We don't use
    // float to represent r because 0.01 is not representable precisely.
    while (item.r * opts_.num_recall_points < r_level) {
      const float recall =
          static_cast<float>(r_level) / opts_.num_recall_points;
      if (r_level < 0) {
        LOG(ERROR) << "Number of recall points should be > 0";
        return 0;
      }
      sum += p;
      r_level -= 1;
      if (pr_out != nullptr) {
        pr_out->emplace_back(p, recall);
      }
    }
    p = std::max(p, item.p);
  }
  for (; r_level >= 0; --r_level) {
    const float recall = static_cast<float>(r_level) / opts_.num_recall_points;
    sum += p;
    if (pr_out != nullptr) {
      pr_out->emplace_back(p, recall);
    }
  }
  return sum / (1 + opts_.num_recall_points);
}

float AveragePrecision::FromBoxes(const std::vector<Detection>& groundtruth,
                                  const std::vector<Detection>& prediction,
                                  std::vector<PR>* pr_out) {
  // Index ground truth boxes based on imageid.
  absl::flat_hash_map<int64_t, std::list<Detection>> gt;
  int num_gt = 0;
  for (auto& box : groundtruth) {
    gt[box.imgid].push_back(box);
    if (!box.difficult && box.ignore == kDontIgnore) {
      ++num_gt;
    }
  }

  if (num_gt == 0) {
    return NAN;
  }

  // Sort all predicted boxes by their scores in a non-ascending order.
  std::vector<Detection> pd = prediction;
  std::sort(pd.begin(), pd.end(), [](const Detection& a, const Detection& b) {
    return a.score > b.score;
  });

  // Computes p-r for every prediction.
  std::vector<PR> pr;
  int correct = 0;
  int num_pd = 0;
  for (int i = 0; i < pd.size(); ++i) {
    const Detection& b = pd[i];
    auto* g = &gt[b.imgid];
    auto best = g->end();
    float best_iou = -INFINITY;
    for (auto it = g->begin(); it != g->end(); ++it) {
      const auto iou = b.box.IoU(it->box);
      if (iou > best_iou) {
        best = it;
        best_iou = iou;
      }
    }
    if ((best != g->end()) && (best_iou >= opts_.iou_threshold)) {
      if (best->difficult) {
        continue;
      }
      switch (best->ignore) {
        case kDontIgnore: {
          ++correct;
          ++num_pd;
          g->erase(best);
          pr.push_back({static_cast<float>(correct) / num_pd,
                        static_cast<float>(correct) / num_gt});
          break;
        }
        case kIgnoreOneMatch: {
          g->erase(best);
          break;
        }
        case kIgnoreAllMatches: {
          break;
        }
      }
    } else {
      ++num_pd;
      pr.push_back({static_cast<float>(correct) / num_pd,
                    static_cast<float>(correct) / num_gt});
    }
  }
  return FromPRCurve(pr, pr_out);
}

}  // namespace image
}  // namespace evaluation
}  // namespace tflite
