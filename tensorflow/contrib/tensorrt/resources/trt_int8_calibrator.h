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

#ifndef TENSORFLOW_CONTRIB_TENSORRT_RESOURCES_TRT_INT8_CALIBRATOR_H_
#define TENSORFLOW_CONTRIB_TENSORRT_RESOURCES_TRT_INT8_CALIBRATOR_H_

#include <atomic>
#include <string>
#include <unordered_map>
#include <utility>
#include "tensorflow/core/platform/mutex.h"

#if GOOGLE_CUDA
#if GOOGLE_TENSORRT

#include "cuda/include/cuda_runtime_api.h"
#include "tensorrt/NvInfer.h"

namespace tensorflow {
namespace tensorrt {
// This class provides a 1 element queue to match TFs push model to
// TRTs pull model for calibration. When TRT implements a means for
// a push calibration This class should be updated accordingly

struct TRTInt8Calibrator : public nvinfer1::IInt8EntropyCalibrator {
 public:
  // Construct a calibrator for future calibration.
  TRTInt8Calibrator(
      const std::unordered_map<string, std::pair<void*, size_t>>& dev_buffers,
      int batch_size, string engine_name);

  // Construct a finalized calibrator where we don't need to run calibration any
  // more, as the calibration data is provided.
  TRTInt8Calibrator(const string& calibration_data);

  ~TRTInt8Calibrator();

  int getBatchSize() const override;

  bool getBatch(void* bindings[], const char* names[],
                int num_bindings) override;

  bool setBatch(const std::unordered_map<string, void*>& data,
                const cudaStream_t stream);

  // Wait until the last batch is consumed by the calibrator and set done.
  void waitAndSetDone();

  // Notify that calibration is done and future batches provided by setBatch()
  // will be ignored.
  void setDone();

  // If not null, calibration is skipped.
  const void* readCalibrationCache(std::size_t& length) override;

  void writeCalibrationCache(const void* ptr, std::size_t length) override;

  const string& getCalibrationTableAsString() { return calibration_table_; }

 private:
  const int batch_size_;

  // mutex for condition_variable
  tensorflow::mutex cond_mtx_;

  // condition variable to implement producer-consumer queue for calibration
  tensorflow::condition_variable cond_;

  // Is calibration finished?
  bool done_;

  // Map to keep tensorrt input buffers and sizes keyed with buffer names
  const std::unordered_map<string, std::pair<void*, size_t>> dev_buffers_;

  bool calib_running_;
  bool batch_is_set_;

  string engine_name_;
  string calibration_table_;
};

}  // namespace tensorrt
}  // namespace tensorflow

#endif
#endif
#endif  // TENSORFLOW_CONTRIB_TENSORRT_RESOURCES_TRT_INT8_CALIBRATOR_H_
