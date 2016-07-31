/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_DEBUG_IO_UTILS_H_
#define TENSORFLOW_DEBUG_IO_UTILS_H_

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/env.h"

namespace tensorflow {

class DebugIO {
 public:
  // Publish a tensor to a debug target URL.
  //
  // Args:
  //   tensor_name: Name of the tensor being published: node_name followed by
  //     a colon, followed by the output slot index. E.g., "node_a:0".
  //   debug_op: Name of the debug op, e.g., "DebugIdentity".
  //   tensor: The Tensor object being published.
  //   wall_time_us: Time stamp for the Tensor. Unit: microseconds (us).
  //   debug_urls: An array of debug target URLs, e.g.,
  //     "file:///foo/tfdbg_dump", "grpc://localhot:11011"
  static Status PublishDebugTensor(const string& tensor_name,
                                   const string& debug_op, const Tensor& tensor,
                                   const uint64 wall_time_us,
                                   const gtl::ArraySlice<string>& debug_urls);

 private:
  static const char* const kFileURLScheme;
  static const char* const kGrpcURLScheme;
};

// Helper class for debug ops.
class DebugFileIO {
 public:
  // Encapsulate the Tensor in an Event protobuf and write it to a directory.
  // The actual path of the dump file will be a contactenation of
  // dump_root_dir, tensor_name, along with the wall_time.
  //
  // For example:
  //   let dump_root_dir = "/tmp/tfdbg_dump",
  //       node_name = "foo/bar",
  //       output_slot = 0,
  //       debug_op = DebugIdentity,
  //       and wall_time_us = 1467891234512345,
  // the dump file will be generated at path:
  //   /tmp/tfdbg_dump/foo/bar_0_DebugIdentity_1467891234512345.
  //
  // Args:
  //   node_name: Name of the node from which the tensor is output.
  //   output_slot: Output slot index.
  //   debug_op: Name of the debug op, e.g., "DebugIdentity".
  //   tensor: The Tensor object to be dumped to file.
  //   wall_time_us: Wall time at which the Tensor is generated during graph
  //     execution. Unit: microseconds (us).
  //   dump_root_dir: Root diretory for dumping the tensor.
  //   dump_file_path: The actual dump file path (passed as reference).
  static Status DumpTensorToDir(const string& node_name,
                                const int32 output_slot, const string& debug_op,
                                const Tensor& tensor, const uint64 wall_time_us,
                                const string& dump_root_dir,
                                string* dump_file_path);

  // Get the full path to the dump file.
  //
  // Args:
  //   dump_root_dir: The dump root directory, e.g., /tmp/tfdbg_dump
  //   node_name: Name of the node from which the dumped tensor is generated,
  //     e.g., foo/bar/node_a
  //   output_slot: Output slot index of the said node, e.g., 0.
  //   debug_op: Name of the debug op, e.g., DebugIdentity.
  //   wall_time_us: Time stamp of the dumped tensor, in microseconds (us).
  static string GetDumpFilePath(const string& dump_root_dir,
                                const string& node_name,
                                const int32 output_slot, const string& debug_op,
                                const uint64 wall_time_us);

 private:
  // Encapsulate the Tensor in an Event protobuf and write it to file.
  static Status DumpTensorToEventFile(
      const string& node_name, const int32 output_slot, const string& debug_op,
      const Tensor& tensor, const uint64 wall_time_us, const string& file_path);

  // Implemented ad hoc here for now.
  // TODO(cais): Replace with shared implementation once http://b/30497715 is
  // fixed.
  static Status RecursiveCreateDir(Env* env, const string& dir);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_DEBUG_IO_UTILS_H_
