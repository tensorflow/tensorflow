/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/contrib/lite/tools/accuracy/ilsvrc/imagenet_topk_eval.h"

#include <numeric>

namespace {
constexpr int kNumCategories = 1001;
std::vector<int> GetTopK(const std::vector<float>& values, int k) {
  CHECK_LE(k, values.size());
  std::vector<int> indices(values.size());

  std::iota(indices.begin(), indices.end(), 0);
  std::sort(indices.begin(), indices.end(),
            [&values](int a, int b) { return values[a] > values[b]; });

  indices.resize(k);
  return indices;
}
}  // namespace

namespace tensorflow {
namespace metrics {
ImagenetTopKAccuracy::ImagenetTopKAccuracy(
    const std::vector<string>& ground_truth_labels, int k)
    : ground_truth_labels_(ground_truth_labels),
      k_(k),
      accuracy_counts_(k_, 0),
      num_samples_(0) {
  CHECK_EQ(kNumCategories, ground_truth_labels.size());
}

Status ImagenetTopKAccuracy::ComputeEval(
    const std::vector<Tensor>& model_outputs, const Tensor& ground_truth) {
  if (model_outputs.size() != 1) {
    return errors::InvalidArgument("Invalid model output: ",
                                   model_outputs.size());
  }
  const Tensor& output = model_outputs[0];
  if (!output.shape().IsSameSize({1, kNumCategories})) {
    return errors::InvalidArgument("Invalid shape of model output: ",
                                   output.shape().DebugString());
  }
  if (ground_truth.dtype() != DT_STRING && ground_truth.dims() != 0) {
    return errors::InvalidArgument("Invalid ground truth type: ",
                                   ground_truth.DebugString());
  }
  string ground_truth_label = ground_truth.scalar<string>()();

  std::vector<float> probabilities;
  probabilities.reserve(kNumCategories);
  if (output.dtype() == DT_FLOAT) {
    auto probs = output.flat<float>();
    for (size_t i = 0; i < probs.size(); i++) {
      probabilities.push_back(probs(i));
    }
  } else {
    auto probs = output.flat<uint8>();
    for (size_t i = 0; i < probs.size(); i++) {
      probabilities.push_back(probs(i));
    }
  }

  CHECK_EQ(kNumCategories, probabilities.size());
  std::vector<int> topK = GetTopK(probabilities, k_);
  int ground_truth_index = GroundTruthIndex(ground_truth_label);
  UpdateSamples(topK, ground_truth_index);
  return Status::OK();
}

const ImagenetTopKAccuracy::AccuracyStats
ImagenetTopKAccuracy::GetTopKAccuracySoFar() const {
  mutex_lock lock(mu_);
  AccuracyStats stats;
  stats.number_of_images = num_samples_;
  stats.topk_counts = accuracy_counts_;
  return stats;
}

void ImagenetTopKAccuracy::UpdateSamples(const std::vector<int>& counts,
                                         int ground_truth_index) {
  mutex_lock lock(mu_);
  for (size_t i = 0; i < counts.size(); ++i) {
    if (ground_truth_index == counts[i]) {
      for (size_t j = i; j < counts.size(); j++) {
        accuracy_counts_[j] += 1;
      }
      break;
    }
  }
  num_samples_++;
}

int ImagenetTopKAccuracy::GroundTruthIndex(const string& label) const {
  auto index = std::find(ground_truth_labels_.cbegin(),
                         ground_truth_labels_.cend(), label);
  CHECK(index != ground_truth_labels_.end()) << "Invalid label: " << label;
  return std::distance(ground_truth_labels_.cbegin(), index);
}
}  //  namespace metrics
}  //  namespace tensorflow
