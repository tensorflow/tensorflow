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

#include "tensorflow/compiler/tf2tensorrt/utils/calibration_resource.h"

#if GOOGLE_CUDA
#if GOOGLE_TENSORRT

namespace tensorflow {
namespace tensorrt {

const absl::string_view kCalibrationContainerName = "TF-TRT-Calibration";

TRTCalibrationResource::~TRTCalibrationResource() {
  VLOG(0) << "Destroying Calibration Resource " << std::endl << DebugString();
}

string TRTCalibrationResource::DebugString() const {
  std::stringstream oss;
  using std::dec;
  using std::endl;
  using std::hex;
  oss << " Calibrator = " << hex << calibrator_.get() << dec << endl
      << " Builder    = " << hex << builder_.get() << dec << endl
      << " Engine     = " << hex << engine_.get() << dec << endl
      << " Logger     = " << hex << &logger_ << dec << endl
      << " Thread     = " << hex << thr_.get() << dec << endl;
  return oss.str();
}

void TRTCalibrationResource::SetCalibrationTable() {
  calibration_table_ = calibrator_->getCalibrationTableAsString();
}

Status TRTCalibrationResource::SerializeToString(string* serialized) {
  calibrator_->waitAndSetDone();
  thr_->join();
  *serialized = calibration_table_;
  if (serialized->empty()) {
    return errors::Unknown("Calibration table is empty.");
  }
  return Status::OK();
}

}  // namespace tensorrt
}  // namespace tensorflow

#endif  // GOOGLE_TENSORRT
#endif  // GOOGLE_CUDA
