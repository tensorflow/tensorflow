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
#ifndef TENSORFLOW_LITE_TOOLS_EVALUATION_STAGES_OBJECT_DETECTION_STAGE_H_
#define TENSORFLOW_LITE_TOOLS_EVALUATION_STAGES_OBJECT_DETECTION_STAGE_H_

#include <memory>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/tools/evaluation/evaluation_delegate_provider.h"
#include "tensorflow/lite/tools/evaluation/evaluation_stage.h"
#include "tensorflow/lite/tools/evaluation/proto/evaluation_config.pb.h"
#include "tensorflow/lite/tools/evaluation/proto/evaluation_stages.pb.h"
#include "tensorflow/lite/tools/evaluation/stages/image_preprocessing_stage.h"
#include "tensorflow/lite/tools/evaluation/stages/object_detection_average_precision_stage.h"
#include "tensorflow/lite/tools/evaluation/stages/tflite_inference_stage.h"

namespace tflite {
namespace evaluation {

// An EvaluationStage to encapsulate the complete Object Detection task.
// Assumes that the object detection model's signature (number of
// inputs/outputs, ordering of outputs & what they denote) is same as the
// MobileNet SSD model:
// https://www.tensorflow.org/lite/examples/object_detection/overview#output_signature.
// Input size/type & number of detections could be different.
//
// This class will be extended to support other types of detection models, if
// required in the future.
class ObjectDetectionStage : public EvaluationStage {
 public:
  explicit ObjectDetectionStage(const EvaluationStageConfig& config)
      : EvaluationStage(config) {}

  TfLiteStatus Init() override { return Init(nullptr); }
  TfLiteStatus Init(const DelegateProviders* delegate_providers);

  TfLiteStatus Run() override;

  EvaluationStageMetrics LatestMetrics() override;

  // Call before Init(). all_labels should contain all possible object labels
  // that can be detected by the model, in the correct order. all_labels should
  // outlive the call to Init().
  void SetAllLabels(const std::vector<std::string>& all_labels) {
    all_labels_ = &all_labels;
  }

  // Call before Run().
  // ground_truth_objects instance should outlive the call to Run().
  void SetInputs(const std::string& image_path,
                 const ObjectDetectionResult& ground_truth_objects) {
    image_path_ = image_path;
    ground_truth_objects_ = &ground_truth_objects;
  }

  // Provides a pointer to the underlying TfLiteInferenceStage.
  // Returns non-null value only if this stage has been initialized.
  TfliteInferenceStage* const GetInferenceStage() {
    return inference_stage_.get();
  }

  // Returns a const pointer to the latest inference output.
  const ObjectDetectionResult* GetLatestPrediction() {
    return &predicted_objects_;
  }

 private:
  const std::vector<std::string>* all_labels_ = nullptr;
  std::unique_ptr<ImagePreprocessingStage> preprocessing_stage_;
  std::unique_ptr<TfliteInferenceStage> inference_stage_;
  std::unique_ptr<ObjectDetectionAveragePrecisionStage> eval_stage_;
  std::string image_path_;

  // Obtained from SetInputs(...).
  const ObjectDetectionResult* ground_truth_objects_;
  // Reflects the outputs generated from the latest call to Run().
  ObjectDetectionResult predicted_objects_;
};

// Reads a tflite::evaluation::ObjectDetectionGroundTruth instance from a
// textproto file and populates a mapping of image name to
// ObjectDetectionResult.
// File with ObjectDetectionGroundTruth can be generated using the
// preprocess_coco_minival.py script in evaluation/tasks/coco_object_detection.
// Useful for wrappers/scripts that use ObjectDetectionStage.
TfLiteStatus PopulateGroundTruth(
    const std::string& grouth_truth_proto_file,
    absl::flat_hash_map<std::string, ObjectDetectionResult>*
        ground_truth_mapping);

}  // namespace evaluation
}  // namespace tflite

#endif  // TENSORFLOW_LITE_TOOLS_EVALUATION_STAGES_OBJECT_DETECTION_STAGE_H_
