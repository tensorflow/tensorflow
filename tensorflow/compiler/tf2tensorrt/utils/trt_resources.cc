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

#include "tensorflow/compiler/tf2tensorrt/utils/trt_resources.h"

#if GOOGLE_CUDA
#if GOOGLE_TENSORRT

namespace tensorflow {
namespace tensorrt {

TRTCalibrationResource::~TRTCalibrationResource() {
  VLOG(0) << "Destroying Calibration Resource " << std::endl << DebugString();
  builder_.reset();
  engine_.reset();
  // We need to manually destroy the builder and engine before the allocator
  // is destroyed.
  allocator_.reset();
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
      << " Allocator  = " << hex << allocator_.get() << dec << endl
      << " Thread     = " << hex << thr_.get() << dec << endl;
  return oss.str();
}

Status TRTCalibrationResource::SerializeToString(string* serialized) {
  calibrator_->waitAndSetDone();
  thr_->join();
  *serialized = calibrator_->getCalibrationTableAsString();
  if (serialized->empty()) {
    return errors::Unknown("Calibration table is empty.");
  }
  return Status::OK();
}

}  // namespace tensorrt
}  // namespace tensorflow

#endif  // GOOGLE_TENSORRT
#endif  // GOOGLE_CUDA
