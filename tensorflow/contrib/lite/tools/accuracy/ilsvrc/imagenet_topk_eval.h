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

#ifndef TENSORFLOW_CONTRIB_LITE_TOOLS_ACCURACY_IMAGENET_TOPK_EVAL_H_
#define TENSORFLOW_CONTRIB_LITE_TOOLS_ACCURACY_IMAGENET_TOPK_EVAL_H_

#include <string>
#include <vector>

#include "tensorflow/contrib/lite/tools/accuracy/accuracy_eval_stage.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/mutex.h"

namespace tensorflow {
namespace metrics {
// An |AccuracyEval| stage that calculates the top K error rate for model
// evaluations on imagenet like datasets.
// Inputs: A {1, 1001} shaped tensor that contains the probabilities for objects
// predicted by the model.
// Ground truth: A |string| label for the image.
// From the input object probabilities, the stage computes the predicted labels
// and finds the top K error rates by comparing the predictions with ground
// truths.
class ImagenetTopKAccuracy : public AccuracyEval {
 public:
  // Accuracy statistics.
  struct AccuracyStats {
    // Number of images evaluated.
    int number_of_images;
    // A vector of size |k| that contains the number of images
    // that have correct labels in top K.
    // E.g. topk_counts[0] contains number of images for which
    // model returned the correct label as the first result.
    // Similarly topk_counts[4] contains the number of images for which
    // model returned the correct label in top 5 results.
    // This can be used to compute the top K error-rate for the model.
    std::vector<int> topk_counts;
  };

  // Creates a new instance of |ImagenetTopKAccuracy| with the given
  // |ground_truth_labels| and |k|.
  // Args:
  // |ground_truth_labels| : an ordered vector of labels for images. This is
  // used to compute the index for the predicted labels and ground_truth label.
  ImagenetTopKAccuracy(const std::vector<string>& ground_truth_labels, int k);

  // Computes accuracy for a given  image. The |model_outputs| should
  // be a vector containing exactly one Tensor of shape: {1, 1001} where each
  // item is a probability of the predicted object representing the image as
  // output by the model.
  // Uses |ground_truth_labels| to compute the index of |model_outputs| and
  // |ground_truth| and computes the top K error rate.
  Status ComputeEval(const std::vector<Tensor>& model_outputs,
                     const Tensor& ground_truth) override;

  // Gets the topK accuracy for images that have been evaluated till now.
  const AccuracyStats GetTopKAccuracySoFar() const;

 private:
  int GroundTruthIndex(const string& label) const;
  void UpdateSamples(const std::vector<int>& counts, int ground_truth_index);
  const std::vector<string> ground_truth_labels_;
  const int k_;
  std::vector<int> accuracy_counts_ GUARDED_BY(mu_);
  int num_samples_ GUARDED_BY(mu_);
  mutable mutex mu_;
};
}  //  namespace metrics
}  //  namespace tensorflow

#endif  // TENSORFLOW_CONTRIB_LITE_TOOLS_ACCURACY_ILSVRC_IMAGENET_TOPK_EVAL_H_
