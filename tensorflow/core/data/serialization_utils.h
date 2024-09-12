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

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/dataset.pb.h"
#include "tensorflow/core/framework/variant_tensor_data.h"
#include "tensorflow/core/lib/core/status.h"
#include "tsl/platform/statusor.h"

namespace tensorflow {
namespace data {

inline constexpr absl::string_view kRetvalOp = "_Retval";

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

// Updates the dataset elements in the checkpoint for given `checkpoint_indices`
// using the given key prefix, assuming that vector of elements have
// checkpointed these before. The elements can be read back by passing the same
// key prefix to ReadElementsFromCheckpoint.
Status UpdateCheckpointElements(
    IteratorStateWriter* writer, StringPiece key_prefix,
    const std::vector<std::vector<Tensor>>& elements,
    const absl::flat_hash_set<int64_t>& checkpoint_indices);

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
  // Produces all key/value pairs stored in this reader. Useful for debugging.
  std::map<string, Tensor> ReadAllTensors();

  // For access to ReadAllTensors()
  friend absl::StatusOr<absl::flat_hash_map<std::string, int64_t>>
  CheckpointStats(const std::string& checkpoint_bytes);

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
  Status WriteScalar(StringPiece key, int64_t val) override;
  Status WriteScalar(StringPiece name, StringPiece key, int64_t val) override;

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

// Wrapper for encoding/decoding the iterator state stored in a Variant tensor.
// The `GetData()` method returns an VariantTensorData object which contains all
// the state needed to restore a single iterator.
//
// Usage example:
//
// Encoding:
//
//   Tensor t(DT_VARIANT, TensorShape({}));
//   t->scalar<Variant>()() = IteratorStateVariant();
//
// Encode() sets the type_name of the VariantTensorData object to
// IteratorStateVariant::TypeName().
//
// Decoding:
//
//   Variant v = <VariantTensorDataProto object>;
//   DecodeUnaryVariant(&v);
//   IteratorStateVariant* wrapper = v.get<IteratorStateVariant>();
//   IteratorStateReader reader({wrapper->GetData()});
//   iterator_resource->Restore(ctx, &reader);
//
// The type_name of the VariantTensorData object to be decoded must match
// IteratorStateVariant::TypeName().
class IteratorStateVariant {
 public:
  IteratorStateVariant() = default;
  IteratorStateVariant(const IteratorStateVariant& other);
  IteratorStateVariant& operator=(IteratorStateVariant&& other) = default;
  IteratorStateVariant& operator=(const IteratorStateVariant& other) = delete;

  static std::string TypeName();

  // Initializes `this` from a VariantTensorData object.
  Status InitializeFromVariantData(std::unique_ptr<VariantTensorData> data);

  // Returns a borrowed pointer to the underlying VariantTensorData.
  const VariantTensorData* GetData() const { return data_.get(); }

  // Encodes this `IteratorStateVariant` into `*data`. Data will be compressed
  // and stored as a scalar `CompressedElement` tensor, or left uncompressed if
  // compression fails.
  void Encode(VariantTensorData* data) const;

  // Decodes from `data`. If `data` contains a single scalar `CompressedElement`
  // tensor, it is assumed to be compressed by `Encode`, and will be
  // uncompressed as part of `Decode`.
  bool Decode(VariantTensorData data);

  std::string DebugString() const;

 private:
  // Returns the compressed element in `data`. If `data` does not contain a
  // compressed element, returns nullptr.
  static const CompressedElement* GetCompressedElement(
      const VariantTensorData& data);

  std::unique_ptr<VariantTensorData> data_;
};

// Returns a GraphDef representation of the given dataset.
Status AsGraphDef(const DatasetBase* dataset,
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

// Analyzes the bytes of a tf.data iterator checkpoint to identify all of the
// keys in the checkpoint along with their sizes in bytes.
absl::StatusOr<absl::flat_hash_map<std::string, int64_t>> CheckpointStats(
    const std::string& checkpoint_bytes);

}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_SERIALIZATION_UTILS_H_
