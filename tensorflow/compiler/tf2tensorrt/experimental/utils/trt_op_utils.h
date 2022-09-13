/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_TF2TENSORRT_EXPERIMENTAL_UTILS_TRT_OP_UTILS_H_
#define TENSORFLOW_COMPILER_TF2TENSORRT_EXPERIMENTAL_UTILS_TRT_OP_UTILS_H_

#include <string>

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/public/session.h"

#if GOOGLE_CUDA && GOOGLE_TENSORRT

namespace tensorflow {
namespace tensorrt {

std::string GetCanonicalEngineName(const std::string& node_name);

Status SetupGetCalibrationDataOp(Session* session, const std::string& prefix);

Status GetCalibrationData(Session* session, const NodeDef& node,
                          const std::string& prefix, std::string* calib_data);

Status SetupSerializeTRTResourceOp(Session* session,
                                   bool save_gpu_specific_engines,
                                   const std::string& prefix);

Status SerializeTRTResource(Session* session, const NodeDef& node,
                            const std::string& out_dir,
                            const std::string& prefix, std::string* filename);

}  // namespace tensorrt
}  // namespace tensorflow

#endif  // GOOGLE_CUDA && GOOGLE_TENSORRT

#endif  // TENSORFLOW_COMPILER_TF2TENSORRT_EXPERIMENTAL_UTILS_TRT_OP_UTILS_H_