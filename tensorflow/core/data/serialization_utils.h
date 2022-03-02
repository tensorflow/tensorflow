/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_DATA_SERIALIZATION_UTILS_H_
#define TENSORFLOW_CORE_DATA_SERIALIZATION_UTILS_H_

#include <string>

#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace data {

// Reads dataset elements from the checkpoint reader using the given key prefix.
Status ReadElementsFromCheckpoint(IteratorContext* ctx,
                                  IteratorStateReader* reader,
                                  StringPiece key_prefix,
                                  std::vector<std::vector<Tensor>>* elements);

// Writes dataset elements to the checkpoint writer using the given key prefix.
// The elements can be read back by passing the same key prefix to
// ReadElementsFromCheckpoint. Only one list of elements can be written under
// the same key_prefix.
Status WriteElementsToCheckpoint(
    IteratorStateWriter* writer, StringPiece key_prefix,
    const std::vector<std::vector<Tensor>>& elements);

// Helper class for reading data from a vector of VariantTensorData objects.
class VariantTensorDataReader : public IteratorStateReader {
 public:
  explicit VariantTensorDataReader(
      const std::vector<const VariantTensorData*>& data);

  bool Contains(StringPiece key) const override;
  bool Contains(StringPiece name, StringPiece key) const override;

  Status ReadScalar(StringPiece key, int64_t* val) const override;
  Status ReadScalar(StringPiece name, StringPiece key,
                    int64_t* val) const override;
  Status ReadScalar(StringPiece key, tstring* val) const override;
  Status ReadScalar(StringPiece name, StringPiece key,
                    tstring* val) const override;
  Status ReadTensor(StringPiece key, Tensor* val) const override;
  Status ReadTensor(FunctionLibraryRuntime* flr, StringPiece key,
                    Tensor* val) const override;
  Status ReadTensor(StringPiece name, StringPiece key,
                    Tensor* val) const override;
  Status ReadTensor(FunctionLibraryRuntime* flr, StringPiece name,
                    StringPiece key, Tensor* val) const override;

 private:
  template <typename T>
  Status ReadScalarInternal(StringPiece name, StringPiece key, T* val) const;
  Status ReadTensorInternal(FunctionLibraryRuntime* flr, StringPiece name,
                            StringPiece key, Tensor* val) const;
  Status ReadDatasetInternal(FunctionLibraryRuntime* flr, StringPiece name,
                             StringPiece key, Tensor* val) const;

  std::map<string, std::map<string, size_t>> map_;
  std::map<string, const VariantTensorData*> data_;  // Not owned.
};

// Helper class used to build a list of VariantTensorData objects, one for each
// iterator which is determined from the key supplied from the Write* calls.
// Sample usage:
// VariantTensorDataWriter writer;
// writer.WriteScalar(full_name("buffer_size"), buffer_.size());
// writer.WriteScalar(full_name("num_threads"), threadpool_.size());
// ....
// std::vector<std::unique_ptr<VariantTensorData>> variants;
// writer.ReleaseData(&variants);
// Now the VariantTensorData objects can be used to serialize.
class VariantTensorDataWriter : public IteratorStateWriter {
 public:
  Status WriteScalar(StringPiece key, const int64_t val) override;
  Status WriteScalar(StringPiece name, StringPiece key,
                     const int64_t val) override;

  Status WriteScalar(StringPiece key, const tstring& val) override;
  Status WriteScalar(StringPiece name, StringPiece key,
                     const tstring& val) override;

  Status WriteTensor(StringPiece key, const Tensor& val) override;
  Status WriteTensor(StringPiece name, StringPiece key,
                     const Tensor& val) override;

  // Releases the built VariantTensorData's to `variants`. Clears out all
  // class state.
  void ReleaseData(std::vector<std::unique_ptr<VariantTensorData>>* variants);

  // Obtains a read-only version of the VariantTensorData's built.
  void GetData(std::vector<const VariantTensorData*>* variants);

 private:
  void MaybeFlush();
  void Reset();

  template <typename T>
  Status WriteScalarInternal(StringPiece name, StringPiece key, const T& val);
  Status WriteTensorInternal(StringPiece name, StringPiece key,
                             const Tensor& val);
  Status WriteDatasetInternal(StringPiece name, StringPiece key,
                              const DatasetBase* dataset);

  bool is_flushed_ = false;
  std::map<string, std::unique_ptr<VariantTensorData>> data_;
  std::map<string, std::vector<string>> keys_;
};

// Returns a GraphDef representation of the given dataset.
Status AsGraphDef(OpKernelContext* ctx, const DatasetBase* dataset,
                  SerializationContext&& serialization_ctx,
                  GraphDef* graph_def);

// Returns a GraphDef representation of the given dataset suitable for
// optimization rewrites. It sets serialization parameters to export a minimum
// graph with additional information for optimization (i.e. ignoring external
// state, not serializing data tensors, not failing if there are datasets which
// do not have AsGraphDef implemented). Sets the `dataset_node` parameter to the
// dataset's node name in the resulting GraphDef.
Status AsGraphDefForRewrite(OpKernelContext* ctx, const DatasetBase* input,
                            std::vector<std::pair<string, Tensor>>* input_list,
                            GraphDef* result, string* dataset_node);

}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_SERIALIZATION_UTILS_H_
