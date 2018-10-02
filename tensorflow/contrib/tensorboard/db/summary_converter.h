/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CONTRIB_TENSORBOARD_DB_SUMMARY_CONVERTER_H_
#define TENSORFLOW_CONTRIB_TENSORBOARD_DB_SUMMARY_CONVERTER_H_

#include "tensorflow/core/framework/summary.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {

// TODO(jart): Delete these methods in favor of new Python implementation.
Status AddTensorAsScalarToSummary(const Tensor& t, const string& tag,
                                  Summary* s);
Status AddTensorAsHistogramToSummary(const Tensor& t, const string& tag,
                                     Summary* s);
Status AddTensorAsImageToSummary(const Tensor& tensor, const string& tag,
                                 int max_images, const Tensor& bad_color,
                                 Summary* s);
Status AddTensorAsAudioToSummary(const Tensor& tensor, const string& tag,
                                 int max_outputs, float sample_rate,
                                 Summary* s);

}  // namespace tensorflow

#endif  // TENSORFLOW_CONTRIB_TENSORBOARD_DB_SUMMARY_CONVERTER_H_
