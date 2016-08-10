// Copyright 2015 Google Inc. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef TENSORFLOW_CONTRIB_IOS_EXAMPLES_CAMERA_TENSORFLOW_UTILS_H_
#define TENSORFLOW_CONTRIB_IOS_EXAMPLES_CAMERA_TENSORFLOW_UTILS_H_

#include <memory>
#include <vector>

#include "tensorflow/core/public/session.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

tensorflow::Status LoadModel(NSString* file_name, NSString* file_type,
                             std::unique_ptr<tensorflow::Session>* session);
tensorflow::Status LoadLabels(NSString* file_name, NSString* file_type,
                              std::vector<std::string>* label_strings);
void GetTopN(const Eigen::TensorMap<Eigen::Tensor<float, 1, Eigen::RowMajor>,
             Eigen::Aligned>& prediction, const int num_results,
             const float threshold,
             std::vector<std::pair<float, int> >* top_results);

#endif  // TENSORFLOW_CONTRIB_IOS_EXAMPLES_CAMERA_TENSORFLOW_UTILS_H_
