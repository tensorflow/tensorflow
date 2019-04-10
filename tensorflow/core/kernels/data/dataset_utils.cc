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

#include "tensorflow/core/kernels/data/dataset_utils.h"

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow {
namespace data {

Status AsGraphDef(OpKernelContext* ctx, DatasetBase* dataset,
                  GraphDef* graph_def) {
  GraphDefBuilder b;
  DatasetBase::DatasetGraphDefBuilder db(&b);
  Node* input_node = nullptr;
  SerializationContext serialization_ctx({});
  TF_RETURN_IF_ERROR(
      db.AddInputDataset(&serialization_ctx, dataset, &input_node));
  TF_RETURN_IF_ERROR(b.ToGraphDef(graph_def));
  return Status::OK();
}

Status ComputeShortCircuitIndices(OpKernelConstruction* ctx,
                                  const NameAttrList& func,
                                  std::vector<int>* indices) {
  FunctionLibraryRuntime::Handle fn_handle;
  TF_RETURN_IF_ERROR(ctx->function_library()->Instantiate(
      func.name(), AttrSlice(&func.attr()), &fn_handle));
  auto cleanup = gtl::MakeCleanup([ctx, fn_handle]() {
    Status s = ctx->function_library()->ReleaseHandle(fn_handle);
    if (!s.ok()) {
      LOG(WARNING) << "Failed to release handle: " << s.error_message();
    }
  });

  // If the function contains any stateful operations, we conservatively execute
  // the entire function.
  if (ctx->function_library()->IsStateful(func.name())) {
    indices->clear();
    return Status::OK();
  }

  const FunctionBody* fn_body =
      ctx->function_library()->GetFunctionBody(fn_handle);
  indices->resize(fn_body->ret_nodes.size());

  for (size_t i = 0; i < fn_body->ret_nodes.size(); ++i) {
    Node* ret_node = fn_body->ret_nodes[i];
    Node* ret_input_node;
    TF_RETURN_IF_ERROR(ret_node->input_node(0, &ret_input_node));

    while (ret_input_node->def().op() == "Identity") {
      TF_RETURN_IF_ERROR(ret_input_node->input_node(0, &ret_input_node));
    }

    if (ret_input_node->def().op() == FunctionLibraryDefinition::kArgOp) {
      TF_RETURN_IF_ERROR(
          GetNodeAttr(ret_input_node->def(), "index", &((*indices)[i])));
    } else {
      indices->clear();
      break;
    }
  }
  return Status::OK();
}

std::vector<bool> ComputeMoveVector(const std::vector<int>& indices) {
  std::map<int, int> last_use;
  for (size_t i = 0; i < indices.size(); ++i) {
    last_use[indices[i]] = i;
  }
  std::vector<bool> can_move;
  can_move.resize(indices.size());
  for (size_t i = 0; i < indices.size(); ++i) {
    can_move[i] = last_use[indices[i]] == i;
  }
  return can_move;
}

Status VerifyTypesMatch(const DataTypeVector& expected,
                        const DataTypeVector& received) {
  if (expected.size() != received.size()) {
    return errors::InvalidArgument(
        "Number of components does not match: expected ", expected.size(),
        " types but got ", received.size(), ".");
  }
  for (size_t i = 0; i < expected.size(); ++i) {
    if (expected[i] != received[i]) {
      return errors::InvalidArgument("Data type mismatch at component ", i,
                                     ": expected ", DataTypeString(expected[i]),
                                     " but got ", DataTypeString(received[i]),
                                     ".");
    }
  }
  return Status::OK();
}

Status VerifyShapesCompatible(const std::vector<PartialTensorShape>& expected,
                              const std::vector<PartialTensorShape>& received) {
  if (expected.size() != received.size()) {
    return errors::InvalidArgument(
        "Number of components does not match: expected ", expected.size(),
        " shapes but got ", received.size(), ".");
  }
  for (size_t i = 0; i < expected.size(); ++i) {
    if (!expected[i].IsCompatibleWith(received[i])) {
      return errors::InvalidArgument("Incompatible shapes at component ", i,
                                     ": expected ", expected[i].DebugString(),
                                     " but got ", received[i].DebugString(),
                                     ".");
    }
  }

  return Status::OK();
}

namespace {

constexpr char kDelimiter[] = "@@";

}  // namespace

VariantTensorDataReader::VariantTensorDataReader(
    const tensorflow::VariantTensorData* data)
    : data_(data) {
  string metadata;
  data_->get_metadata(&metadata);
  auto keys = str_util::Split(metadata, kDelimiter, str_util::SkipEmpty());
  for (size_t i = 0; i < keys.size(); ++i) {
    map_[keys[i]] = i;
  }
}

Status VariantTensorDataReader::ReadScalar(StringPiece key, int64* val) {
  return ReadScalarInternal(key, val);
}

Status VariantTensorDataReader::ReadScalar(StringPiece key, string* val) {
  return ReadScalarInternal(key, val);
}

Status VariantTensorDataReader::ReadTensor(StringPiece key, Tensor* val) {
  return ReadTensorInternal(key, val);
}

bool VariantTensorDataReader::Contains(StringPiece key) {
  return map_.find(string(key)) != map_.end();
}

template <typename T>
Status VariantTensorDataReader::ReadScalarInternal(StringPiece key, T* val) {
  if (map_.find(string(key)) == map_.end()) {
    return errors::NotFound(key);
  }
  *val = data_->tensors(map_[string(key)]).scalar<T>()();
  return Status::OK();
}

Status VariantTensorDataReader::ReadTensorInternal(StringPiece key,
                                                   Tensor* val) {
  if (map_.find(string(key)) == map_.end()) {
    return errors::NotFound(key);
  }
  *val = data_->tensors(map_[string(key)]);
  return Status::OK();
}

Status VariantTensorDataWriter::WriteScalar(StringPiece key, const int64 val) {
  return WriteScalarInternal(key, val);
}

Status VariantTensorDataWriter::WriteScalar(StringPiece key,
                                            const string& val) {
  return WriteScalarInternal(key, val);
}

Status VariantTensorDataWriter::WriteTensor(StringPiece key,
                                            const Tensor& val) {
  return WriteTensorInternal(key, val);
}

Status VariantTensorDataWriter::Flush() {
  string metadata;
  for (size_t i = 0; i < keys_.size(); ++i) {
    strings::StrAppend(&metadata, kDelimiter, keys_[i]);
  }
  data_->set_metadata(metadata);
  return Status::OK();
}

template <typename T>
Status VariantTensorDataWriter::WriteScalarInternal(StringPiece key,
                                                    const T& val) {
  Tensor val_t = Tensor(DataTypeToEnum<T>::v(), TensorShape({}));
  val_t.scalar<T>()() = val;
  return WriteTensorInternal(key, val_t);
}

Status VariantTensorDataWriter::WriteTensorInternal(StringPiece key,
                                                    const Tensor& val) {
  DCHECK_EQ(key.find(kDelimiter), string::npos);
  keys_.push_back(string(key));
  *(data_->add_tensors()) = val;
  return Status::OK();
}

Status AddToFunctionLibrary(FunctionLibraryDefinition* base,
                            const FunctionLibraryDefinition& to_add) {
  for (const auto& fn : to_add.ListFunctionNames()) {
    if (auto found = base->Find(fn)) {
      if (!OpDefEqual(found->signature(), to_add.Find(fn)->signature())) {
        return errors::InvalidArgument("Cannot add function '", fn,
                                       "' because a different function with "
                                       "the same signature already exists.");
      }
      TF_RETURN_IF_ERROR(base->RemoveFunction(fn));
    }
  }
  return base->AddLibrary(to_add);
}

Status AddToFunctionLibrary(FunctionLibraryDefinition* base,
                            const FunctionDefLibrary& to_add) {
  for (const auto& fd : to_add.function()) {
    if (auto found = base->Find(fd.signature().name())) {
      if (!OpDefEqual(found->signature(), fd.signature())) {
        return errors::InvalidArgument("Cannot add function '",
                                       fd.signature().name(),
                                       "' because a different function with "
                                       "the same signature already exists.");
      }
      TF_RETURN_IF_ERROR(base->RemoveFunction(fd.signature().name()));
    }
  }
  return base->AddLibrary(to_add);
}

std::function<void(std::function<void()>)> RunnerWithMaxParallelism(
    std::function<void(std::function<void()>)> runner, int max_parallelism) {
  return std::bind(
      [max_parallelism](
          // Note: `runner` is a const reference to avoid copying it.
          const std::function<void(std::function<void()>)>& runner,
          std::function<void()> fn) {
        std::function<void()> scoped_fn = std::bind(
            [max_parallelism](const std::function<void()>& fn) {
              ScopedPerThreadMaxParallelism scope(max_parallelism);
              fn();
            },
            std::move(fn));
        runner(std::move(scoped_fn));
      },
      std::move(runner), std::placeholders::_1);
}

Status CreateFunctionLibraryDefinition(
    const FunctionLibraryDefinition* lib_def, const string& func_name,
    std::shared_ptr<FunctionLibraryDefinition>* result) {
  DCHECK(lib_def != nullptr);
  const FunctionDef* fdef = lib_def->Find(func_name);
  DCHECK(fdef != nullptr);
  *result = std::make_shared<FunctionLibraryDefinition>(
      lib_def->ReachableDefinitions(*fdef));
  TF_RETURN_IF_ERROR((*result)->AddFunctionDef(*fdef));
  return Status::OK();
}

}  // namespace data
}  // namespace tensorflow
