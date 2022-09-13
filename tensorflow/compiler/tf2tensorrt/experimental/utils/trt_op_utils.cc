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

#include "tensorflow/compiler/tf2tensorrt/experimental/utils/trt_op_utils.h"

#include <vector>

#include "absl/strings/string_view.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/compiler/tf2tensorrt/experimental/utils/session_utils.h"
#include "tensorflow/compiler/tf2tensorrt/ops/trt_ops.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/path.h"

#if GOOGLE_CUDA && GOOGLE_TENSORRT

namespace tensorflow {
namespace tensorrt {

std::string GetCanonicalEngineName(const std::string& node_name) {
  absl::string_view name(node_name);
  size_t last_slash = name.find_last_of('/');
  if (last_slash != absl::string_view::npos) {
    name.remove_prefix(last_slash + 1);
  }
  return std::string(name);
}

Status SetupGetCalibrationDataOp(Session* session, const std::string& prefix) {
  auto root = Scope::NewRootScope();
  auto input = ops::Placeholder(root.WithOpName("input"), DT_STRING);
  auto output = ops::GetCalibrationDataOp(root.WithOpName("output"), input);

  GraphDef graph;
  TF_RETURN_IF_ERROR(root.ToGraphDef(&graph));
  TF_RETURN_IF_ERROR(ImportGraphDefToSession(session, graph, prefix));
  return OkStatus();
}

Status GetCalibrationData(Session* session, const NodeDef& node,
                          const std::string& prefix, std::string* calib_data) {
  std::vector<Tensor> out_tensors;
  Tensor in_tensor(GetCanonicalEngineName(node.name()));
  TF_RETURN_IF_ERROR(session->Run({{prefix + "/input", in_tensor}},
                                  {prefix + "/output"}, {}, &out_tensors));
  Tensor out_tensor = out_tensors.at(0);
  *calib_data = out_tensor.scalar<tstring>()();
  return OkStatus();
}

Status SetupSerializeTRTResourceOp(Session* session,
                                   bool save_gpu_specific_engines,
                                   const std::string& prefix) {
  auto attrs = ops::SerializeTRTResource::Attrs()
                   .DeleteResource(true)
                   .SaveGpuSpecificEngines(save_gpu_specific_engines);
  auto root = Scope::NewRootScope();
  auto resource_name =
      ops::Placeholder(root.WithOpName("resource_name"), DT_STRING);
  auto filename = ops::Placeholder(root.WithOpName("filename"), DT_STRING);
  auto output = ops::SerializeTRTResource(root.WithOpName("output"),
                                          resource_name, filename, attrs);

  GraphDef graph;
  TF_RETURN_IF_ERROR(root.ToGraphDef(&graph));
  TF_RETURN_IF_ERROR(ImportGraphDefToSession(session, graph, prefix, true));
  return OkStatus();
}

Status SerializeTRTResource(Session* session, const NodeDef& node,
                            const std::string& out_dir,
                            const std::string& prefix, std::string* filename) {
  std::string engine_name = GetCanonicalEngineName(node.name());
  *filename =
      tensorflow::io::JoinPath(out_dir, "trt-serialized-engine." + engine_name);
  Tensor engine_name_tensor(engine_name);
  Tensor filepath_tensor(*filename);
  TF_RETURN_IF_ERROR(
      session->Run({{prefix + "/resource_name", engine_name_tensor},
                    {prefix + "/filename", filepath_tensor}},
                   {}, {prefix + "/output"}, nullptr));
  return OkStatus();
}

}  // namespace tensorrt
}  // namespace tensorflow

#endif  // GOOGLE_CUDA && GOOGLE_TENSORRT