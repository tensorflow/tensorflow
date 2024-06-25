/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_NODE_FILE_WRITER_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_NODE_FILE_WRITER_H_

#include <string>
#include <unordered_map>

#include "absl/container/flat_hash_set.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/util/env_var.h"

namespace tensorflow {

// Writes out the NodeDef and the input shapes/dtypes for an executed node to a
// file. This allows the set of executed nodes for a model or test to be
// examined and processed. Currently this is used by an internal tool which
// checks that ops executed by tests are deterministic.
class NodeFileWriter {
 public:
  // Creates or reuses a NodeFileWriter if environmental variable
  // TF_NODE_FILE_WRITER_DIRECTORY is set, which specifies the directory where
  // the node file will be created in. Otherwise, returns nullptr. When called
  // with the same device_name, the same NodeFileWriter will be returned.
  static absl::StatusOr<NodeFileWriter*> GetNodeFileWriterIfEnabled(
      const std::string& device_name, Env* env);

  // Records the execution of a node, if eligible, by writing the node to the
  // file. Only writes the node if the exact node with the given input
  // shapes/dtypes hasn't already been written. Should be called once every time
  // a node is run.
  Status RecordNodeExecution(OpKernel* op_kernel, OpKernelContext* context);

  const std::string& filename() { return filename_; }

 private:
  explicit NodeFileWriter(std::string filename)
      : filename_{std::move(filename)} {}

  Status Init(Env* env) {
    return env->NewWritableFile(filename_, &node_def_file_);
  }

  // Writes the NodeDef to a file, if it hasn't already been written yet.
  Status MaybeWriteNodeDefToFile(const NodeDef& def);

  const std::string filename_;
  mutex mu_;
  // Hashes of the NodeDefs already written to the file
  absl::flat_hash_set<uint64> written_hashes_ TF_GUARDED_BY(mu_);

  std::unique_ptr<WritableFile> node_def_file_ TF_PT_GUARDED_BY(mu_);
};

}  // namespace tensorflow
#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_NODE_FILE_WRITER_H_
