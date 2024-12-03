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

#include "tensorflow/core/common_runtime/node_file_writer.h"

#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_replace.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/platform/random.h"
#include "tensorflow/core/util/equal_graph_def.h"

namespace {

// Avoiding writing to disk very commonly executed ops that are known to be
// deterministic. This reduces the filesize.
const absl::flat_hash_set<std::string>* const kOpsToSkipWriting =
    new absl::flat_hash_set<std::string>{"Add",
                                         "AddV2",
                                         "BroadcastTo",
                                         "Cast",
                                         "ConcatV2",
                                         "Const",
                                         "_EagerConst",
                                         "Enter",
                                         "Exit",
                                         "Fill",
                                         "_HostSend",
                                         "Identity",
                                         "Less",
                                         "MatrixDiagV3",
                                         "Merge",
                                         "Mul",
                                         "NextIteration",
                                         "Pack",
                                         "RandomStandardNormal",
                                         "RandomUniform",
                                         "Range",
                                         "RealDiv",
                                         "Reshape",
                                         "_Send",
                                         "Shape",
                                         "StridedSlice",
                                         "Sub",
                                         "Switch",
                                         "Transpose",
                                         "_XlaCompile"};

// If a host int32 input has at most this many elements, the tensor value will
// be written to the file.
const int kMaxInt32Elems = 10;

}  // namespace

namespace tensorflow {

/*static*/ absl::StatusOr<NodeFileWriter*>
tensorflow::NodeFileWriter::GetNodeFileWriterIfEnabled(
    const std::string& device_name, Env* env) {
  // First get the directory from TF_NODE_FILE_WRITER_DIRECTORY.
  static const std::string* const node_dir = [] {
    std::string node_dir;
    TF_CHECK_OK(
        ReadStringFromEnvVar("TF_NODE_FILE_WRITER_DIRECTORY", "", &node_dir));
    if (node_dir == "test_undeclared_outputs_dir") {
      bool env_set = io::GetTestUndeclaredOutputsDir(&node_dir);
      if (!env_set || node_dir.empty()) {
        LOG(WARNING)
            << "TF_NODE_FILE_WRITER_DIRECTORY was set to "
               "'test_undeclared_outputs_dir', but the environmental "
               "variable TEST_UNDECLARED_OUTPUTS_DIR does not exist or "
               "is empty. NodeDef collection will be skipped.";
      } else {
        node_dir = io::JoinPath(node_dir, "node_defs");
      }
    }
    return new std::string{node_dir};
  }();
  if (node_dir->empty()) {
    return nullptr;
  }

  static mutex mu(LINKER_INITIALIZED);
  // Cache a NodeFileWriter* for each device name, so that different Sessions
  // each share the same NodeFileWriters. Sharing NodeFileWriters reduces the
  // total size of the outputted files, since it means if multiple Sessions run
  // the same op, the op is only written recorded to disk once.
  static auto* device_name_to_writer =
      new absl::flat_hash_map<std::string, NodeFileWriter*>{};
  mutex_lock l(mu);
  auto it = device_name_to_writer->find(device_name);
  if (it == device_name_to_writer->end()) {
    absl::Status s = env->CreateDir(*node_dir);
    if (!s.ok() && s.code() != error::ALREADY_EXISTS) {
      return s;
    }

    // Put the device name in the filename for debugging purposes. Also append
    // random number in case multiple processes write out nodes concurrently.
    std::string filename = strings::StrCat(
        "node_defs", absl::StrReplaceAll(device_name, {{"/", "_"}, {":", "_"}}),
        "_", random::New64());

    auto* writer = new NodeFileWriter{io::JoinPath(*node_dir, filename)};
    s = writer->Init(env);
    if (!s.ok()) {
      delete writer;
      return s;
    }
    it = device_name_to_writer->insert({device_name, writer}).first;
  }
  return it->second;
}

absl::Status NodeFileWriter::RecordNodeExecution(OpKernel* op_kernel,
                                                 OpKernelContext* context) {
  if (kOpsToSkipWriting->count(op_kernel->type_string())) {
    return absl::OkStatus();
  }
  NodeDef def;
  def.set_name("NodeFileWriter");
  def.set_op(op_kernel->def().op());
  *def.mutable_attr() = op_kernel->def().attr();
  // The input shapes/dtypes are stored in the 'attr' section of the NodeDef
  AttrValue& input_shapes = (*def.mutable_attr())["_input_shapes"];
  AttrValue& input_dtypes = (*def.mutable_attr())["_input_dtypes"];
  for (int i = 0; i < context->num_inputs(); i++) {
    if (!context->has_input(i) || context->input_is_ref(i)) {
      // Calling context->input(i) requires the input to exist and not be a ref,
      // so return immediately if that is not the case.
      return absl::OkStatus();
    }
    TensorShapeProto* shape_proto = input_shapes.mutable_list()->add_shape();
    const Tensor& input = context->input(i);
    input.shape().AsProto(shape_proto);
    input_dtypes.mutable_list()->add_type(context->input_dtype(i));
    // Store small int32 host inputs, as they often represent shapes.
    if (input.NumElements() <= kMaxInt32Elems && input.dtype() == DT_INT32 &&
        context->input_memory_type(i) == HOST_MEMORY) {
      AttrValue& input_tensor =
          (*def.mutable_attr())[strings::StrCat("_input_tensor_", i)];
      input.AsProtoField(input_tensor.mutable_tensor());
    } else if (!DataTypeIsFloating(input.dtype())) {
      // Skip ops with non-floating-point inputs, since these are not useful
      // when testing determinism.
      return absl::OkStatus();
    }
  }
  return MaybeWriteNodeDefToFile(def);
}

absl::Status NodeFileWriter::MaybeWriteNodeDefToFile(const NodeDef& def) {
  std::string def_str = def.SerializeAsString();
  uint64 size = def_str.size();
  std::string size_as_str;
  // The file consists of a series of records, each consisting of a 64-bit
  // little endian integer representing the size of the serialized NodeDef,
  // followed by the serialized NodeDef.
  for (unsigned int i = 0; i < 8; i++) {
    size_as_str.push_back((size >> (i * 8)) & 0xff);
  }

  EqualGraphDefOptions options;
  options.ignore_internal_attrs = false;
  uint64 hash = NodeDefHash(def, options);

  mutex_lock l{mu_};
  if (written_hashes_.count(hash) == 0) {
    TF_RETURN_IF_ERROR(node_def_file_->Append(size_as_str));
    TF_RETURN_IF_ERROR(node_def_file_->Append(def_str));
    written_hashes_.insert(hash);
    // Flush after each write, since NodeFileWriters are never destructed so the
    // file is never closed.
    TF_RETURN_IF_ERROR(node_def_file_->Flush());
  }
  return absl::OkStatus();
}

}  // namespace tensorflow
