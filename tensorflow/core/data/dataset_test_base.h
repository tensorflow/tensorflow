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

#ifndef TENSORFLOW_CORE_DATA_DATASET_TEST_BASE_H_
#define TENSORFLOW_CORE_DATA_DATASET_TEST_BASE_H_

#include <stddef.h>

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/common_runtime/process_function_library_runtime.h"
#include "tensorflow/core/data/name_utils.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/cancellation.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/dataset_options.pb.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/function_handle_cache.h"
#include "tensorflow/core/framework/function_testlib.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/framework/variant_op_registry.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/lib/io/zlib_compression_options.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/refcount.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/threadpool.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/tensor_slice_reader_cache.h"

namespace tensorflow {
namespace data {

typedef std::vector<
    std::pair<string, tensorflow::FunctionDefHelper::AttrValueWrapper>>
    AttributeVector;

constexpr int kDefaultCPUNum = 2;
constexpr int kDefaultThreadNum = 2;

// Creates a tensor with the specified dtype, shape, and value.
template <typename T>
static Tensor CreateTensor(const TensorShape& input_shape,
                           gtl::ArraySlice<T> input_data) {
  Tensor tensor(DataTypeToEnum<T>::value, input_shape);
  test::FillValues<T>(&tensor, input_data);
  return tensor;
}

// Creates a tensor with the specified dtype and shape, with values 0, 1, 2, ...
template <typename T>
static Tensor CreateTensor(const TensorShape& input_shape) {
  Tensor tensor(DataTypeToEnum<T>::value, input_shape);
  test::FillIota<T>(&tensor, 0);
  return tensor;
}

// Creates a vector of tensors with the specified dtype, shape, and values.
template <typename T>
std::vector<Tensor> CreateTensors(
    const TensorShape& shape, const std::vector<gtl::ArraySlice<T>>& values) {
  std::vector<Tensor> result;
  result.reserve(values.size());
  for (auto& value : values) {
    result.emplace_back(CreateTensor<T>(shape, value));
  }
  return result;
}

enum class CompressionType { ZLIB = 0, GZIP = 1, RAW = 2, UNCOMPRESSED = 3 };

// Returns a string representation for the given compression type.
string ToString(CompressionType compression_type);

// Gets the specified zlib compression options according to the compression
// type. Note that `CompressionType::UNCOMPRESSED` is not supported because
// `ZlibCompressionOptions` does not have an option.
io::ZlibCompressionOptions GetZlibCompressionOptions(
    CompressionType compression_type);

// Used to specify parameters when writing data into files with compression.
// `input_buffer_size` and `output_buffer_size` specify the input and output
// buffer size when ZLIB and GZIP compression is used.
struct CompressionParams {
  CompressionType compression_type = CompressionType::UNCOMPRESSED;
  int32 input_buffer_size = 0;
  int32 output_buffer_size = 0;
};

// Writes the input data into the file without compression.
absl::Status WriteDataToFile(const string& filename, const char* data);

// Writes the input data into the file with the specified compression.
absl::Status WriteDataToFile(const string& filename, const char* data,
                             const CompressionParams& params);

// Writes the input data into the TFRecord file with the specified compression.
absl::Status WriteDataToTFRecordFile(
    const string& filename, const std::vector<absl::string_view>& records,
    const CompressionParams& params);

// Provides the parameters for running the dataset op.
class DatasetParams {
 public:
  DatasetParams(DataTypeVector output_dtypes,
                std::vector<PartialTensorShape> output_shapes,
                string node_name);

  virtual ~DatasetParams() = default;

  // Returns the inputs (except the input datasets) as a tensor vector.
  virtual std::vector<Tensor> GetInputTensors() const = 0;

  // Returns the dataset input names as a string vector.
  virtual absl::Status GetInputNames(
      std::vector<string>* input_names) const = 0;

  // Returns the dataset attributes as a vector.
  virtual absl::Status GetAttributes(AttributeVector* attributes) const = 0;

  // Checks if the tensor is a dataset variant tensor.
  static bool IsDatasetTensor(const Tensor& tensor);

  string node_name() const { return node_name_; }

  DataTypeVector output_dtypes() const { return output_dtypes_; }

  std::vector<PartialTensorShape> output_shapes() const {
    return output_shapes_;
  }

  string iterator_prefix() const { return iterator_prefix_; }

  const std::vector<std::shared_ptr<DatasetParams>>& input_dataset_params()
      const {
    return input_dataset_params_;
  }

  // Returns the functions that will be used when running the dataset op.
  virtual std::vector<FunctionDef> func_lib() const { return {}; }

  // Returns the dataset type for the op represented by these parameters. This
  // type usually needs to match the constant called `kDatasetType` defined in
  // the dataset kernel.
  virtual string dataset_type() const = 0;

  // Returns the dataset op name. By default, it returns the Op::kDatasetType
  // concatenated with "Dataset". For ops that do not have "Dataset" suffix,
  // this method can be overriden to return a different name.
  virtual string op_name() const {
    name_utils::OpNameParams params;
    params.op_version = op_version();
    return name_utils::OpName(dataset_type(), params);
  }

  virtual int op_version() const { return op_version_; }

 protected:
  std::vector<std::shared_ptr<DatasetParams>> input_dataset_params_;
  DataTypeVector output_dtypes_;
  std::vector<PartialTensorShape> output_shapes_;
  string node_name_;
  string iterator_prefix_ = "Iterator";
  int op_version_ = 1;
};

// `RangeDatasetParams` is a common dataset parameter type that are used in
// testing.
class RangeDatasetParams : public DatasetParams {
 public:
  RangeDatasetParams(int64_t start, int64_t stop, int64_t step,
                     DataTypeVector output_dtypes,
                     std::vector<PartialTensorShape> output_shapes,
                     string node_name);

  RangeDatasetParams(int64_t start, int64_t stop, int64_t step);

  RangeDatasetParams(int64_t start, int64_t stop, int64_t step,
                     DataTypeVector output_dtypes);

  std::vector<Tensor> GetInputTensors() const override;

  absl::Status GetInputNames(std::vector<string>* input_names) const override;

  absl::Status GetAttributes(AttributeVector* attr_vector) const override;

  string dataset_type() const override;

 private:
  int64_t start_;
  int64_t stop_;
  int64_t step_;
};

// `BatchDatasetParams` is a common dataset parameter type that are used in
// testing.
class BatchDatasetParams : public DatasetParams {
 public:
  template <typename T>
  BatchDatasetParams(T input_dataset_params, int64_t batch_size,
                     bool drop_remainder, bool parallel_copy,
                     DataTypeVector output_dtypes,
                     std::vector<PartialTensorShape> output_shapes,
                     string node_name)
      : DatasetParams(std::move(output_dtypes), std::move(output_shapes),
                      std::move(node_name)),
        batch_size_(batch_size),
        drop_remainder_(drop_remainder),
        parallel_copy_(parallel_copy) {
    input_dataset_params_.push_back(std::make_unique<T>(input_dataset_params));
    op_version_ = 2;
    iterator_prefix_ =
        name_utils::IteratorPrefix(input_dataset_params.dataset_type(),
                                   input_dataset_params.iterator_prefix());
  }

  std::vector<Tensor> GetInputTensors() const override;

  absl::Status GetInputNames(std::vector<string>* input_names) const override;

  absl::Status GetAttributes(AttributeVector* attr_vector) const override;

  string dataset_type() const override;

 private:
  int64_t batch_size_;
  bool drop_remainder_;
  bool parallel_copy_;
};

// `MapDatasetParams` is a common dataset parameter type that are used in
// testing.
class MapDatasetParams : public DatasetParams {
 public:
  template <typename T>
  MapDatasetParams(T input_dataset_params, std::vector<Tensor> other_arguments,
                   FunctionDefHelper::AttrValueWrapper func,
                   std::vector<FunctionDef> func_lib,
                   DataTypeVector type_arguments, DataTypeVector output_dtypes,
                   std::vector<PartialTensorShape> output_shapes,
                   bool use_inter_op_parallelism, bool preserve_cardinality,
                   string node_name)
      : DatasetParams(std::move(output_dtypes), std::move(output_shapes),
                      std::move(node_name)),
        other_arguments_(std::move(other_arguments)),
        func_(std::move(func)),
        func_lib_(std::move(func_lib)),
        type_arguments_(std::move(type_arguments)),
        use_inter_op_parallelism_(use_inter_op_parallelism),
        preserve_cardinality_(preserve_cardinality) {
    input_dataset_params_.push_back(std::make_unique<T>(input_dataset_params));
    iterator_prefix_ =
        name_utils::IteratorPrefix(input_dataset_params.dataset_type(),
                                   input_dataset_params.iterator_prefix());
  }

  std::vector<Tensor> GetInputTensors() const override;

  absl::Status GetInputNames(std::vector<string>* input_names) const override;

  absl::Status GetAttributes(AttributeVector* attr_vector) const override;

  string dataset_type() const override;

  std::vector<FunctionDef> func_lib() const override;

 private:
  std::vector<Tensor> other_arguments_;
  FunctionDefHelper::AttrValueWrapper func_;
  std::vector<FunctionDef> func_lib_;
  DataTypeVector type_arguments_;
  bool use_inter_op_parallelism_;
  bool preserve_cardinality_;
};

// `TensorSliceDatasetParams` is a common dataset parameter type that are used
// in testing.
class TensorSliceDatasetParams : public DatasetParams {
 public:
  TensorSliceDatasetParams(std::vector<Tensor> components, string node_name,
                           bool is_files = false);

  std::vector<Tensor> GetInputTensors() const override;

  absl::Status GetInputNames(std::vector<string>* input_names) const override;

  absl::Status GetAttributes(AttributeVector* attr_vector) const override;

  string dataset_type() const override;

  int64_t num_slices() const { return components_[0].dim_size(0); }

  size_t num_tensors_per_slice() const { return components_.size(); }

 private:
  DataTypeVector TensorSliceDtypes(const std::vector<Tensor>& input_components);

  std::vector<PartialTensorShape> TensorSliceShapes(
      const std::vector<Tensor>& input_components);

 public:
  std::vector<Tensor> components_;
  bool is_files_;
};

// `TakeDatasetParams` is a common dataset parameter type that are used in
// testing.
class TakeDatasetParams : public DatasetParams {
 public:
  template <typename T>
  TakeDatasetParams(T input_dataset_params, int count,
                    DataTypeVector output_dtypes,
                    std::vector<PartialTensorShape> output_shapes,
                    string node_name)
      : DatasetParams(std::move(output_dtypes), std::move(output_shapes),
                      std::move(node_name)),
        count_(count) {
    input_dataset_params_.push_back(std::make_unique<T>(input_dataset_params));
    iterator_prefix_ =
        name_utils::IteratorPrefix(input_dataset_params.dataset_type(),
                                   input_dataset_params.iterator_prefix());
  }

  std::vector<Tensor> GetInputTensors() const override;

  absl::Status GetInputNames(std::vector<string>* input_names) const override;

  absl::Status GetAttributes(AttributeVector* attr_vector) const override;

  string dataset_type() const override;

 private:
  int64_t count_;
};

// `ConcatenateDatasetParams` is a common dataset parameter type that are used
// in testing.
class ConcatenateDatasetParams : public DatasetParams {
 public:
  template <typename T, typename P>
  ConcatenateDatasetParams(T input_dataset_params_0, P input_dataset_params_1,
                           DataTypeVector output_dtypes,
                           std::vector<PartialTensorShape> output_shapes,
                           string node_name)
      : DatasetParams(std::move(output_dtypes), std::move(output_shapes),
                      std::move(node_name)) {
    input_dataset_params_.push_back(
        std::make_unique<T>(input_dataset_params_0));
    input_dataset_params_.push_back(
        std::make_unique<T>(input_dataset_params_1));
    iterator_prefix_ =
        name_utils::IteratorPrefix(input_dataset_params_0.dataset_type(),
                                   input_dataset_params_0.iterator_prefix());
  }

  std::vector<Tensor> GetInputTensors() const override;

  absl::Status GetInputNames(std::vector<string>* input_names) const override;

  absl::Status GetAttributes(AttributeVector* attr_vector) const override;

  string dataset_type() const override;
};

// `OptionsDatasetParams` is a common dataset parameter type that is used in
// testing.
class OptionsDatasetParams : public DatasetParams {
 public:
  template <typename T>
  OptionsDatasetParams(T input_dataset_params, const string& serialized_options,
                       DataTypeVector output_dtypes,
                       std::vector<PartialTensorShape> output_shapes,
                       string node_name)
      : DatasetParams(std::move(output_dtypes), std::move(output_shapes),
                      std::move(node_name)),
        serialized_options_(serialized_options) {
    input_dataset_params_.push_back(std::make_unique<T>(input_dataset_params));
  }

  std::vector<Tensor> GetInputTensors() const override;

  absl::Status GetInputNames(std::vector<string>* input_names) const override;

  absl::Status GetAttributes(AttributeVector* attr_vector) const override;

  string dataset_type() const override;

 private:
  string serialized_options_;
};

template <typename T>
struct GetNextTestCase {
  GetNextTestCase(T dataset_params, std::vector<Tensor> expected_outputs,
                  bool compare_order = true)
      : dataset_params(std::move(dataset_params)),
        expected_outputs(std::move(expected_outputs)),
        compare_order(compare_order) {}

  T dataset_params;
  std::vector<Tensor> expected_outputs;
  bool compare_order;
};

template <typename T>
struct SkipTestCase {
  SkipTestCase(T dataset_params, int num_to_skip, int expected_num_skipped,
               bool get_next = false, std::vector<Tensor> expected_outputs = {},
               bool compare_order = true)
      : dataset_params(std::move(dataset_params)),
        num_to_skip(num_to_skip),
        expected_num_skipped(expected_num_skipped),
        get_next(get_next),
        expected_outputs(std::move(expected_outputs)),
        compare_order(compare_order) {}

  T dataset_params;
  int num_to_skip;
  int expected_num_skipped;
  bool get_next;
  std::vector<Tensor> expected_outputs;
  bool compare_order;
};

template <typename T>
struct DatasetNodeNameTestCase {
  T dataset_params;
  string expected_node_name;
};

template <typename T>
struct DatasetTypeStringTestCase {
  T dataset_params;
  string expected_dataset_type_string;
};

template <typename T>
struct DatasetOutputDtypesTestCase {
  T dataset_params;
  DataTypeVector expected_output_dtypes;
};

template <typename T>
struct DatasetOutputShapesTestCase {
  T dataset_params;
  std::vector<PartialTensorShape> expected_output_shapes;
};

template <typename T>
struct CardinalityTestCase {
  T dataset_params;
  int64_t expected_cardinality;
};

template <typename T>
struct DatasetSaveTestCase {
  T dataset_params;
};

template <typename T>
struct IteratorOutputDtypesTestCase {
  T dataset_params;
  DataTypeVector expected_output_dtypes;
};

template <typename T>
struct IteratorOutputShapesTestCase {
  T dataset_params;
  std::vector<PartialTensorShape> expected_output_shapes;
};

template <typename T>
struct IteratorPrefixTestCase {
  T dataset_params;
  string expected_iterator_prefix;
};

template <typename T>
struct IteratorSaveAndRestoreTestCase {
  IteratorSaveAndRestoreTestCase(T dataset_params, std::vector<int> breakpoints,
                                 std::vector<Tensor> expected_outputs,
                                 bool compare_order = true)
      : dataset_params(std::move(dataset_params)),
        breakpoints(std::move(breakpoints)),
        expected_outputs(std::move(expected_outputs)),
        compare_order(compare_order) {}

  T dataset_params;
  std::vector<int> breakpoints;
  std::vector<Tensor> expected_outputs;
  bool compare_order;
};

// Class composing a dataset with its dependencies.
class TestDataset {
 public:
  // TestDataset expects that the caller has Ref'd the wrapped dataset. When
  // TestDataset is destroyed, it will Unref the dataset.
  TestDataset(std::unique_ptr<OpKernel> kernel_,
              std::unique_ptr<OpKernelContext::Params> ctx_params,
              std::unique_ptr<OpKernelContext> ctx,
              std::vector<std::unique_ptr<Tensor>> input_tensors,
              DatasetBase* dataset)
      : kernel_(std::move(kernel_)),
        ctx_params_(std::move(ctx_params)),
        ctx_(std::move(ctx)),
        input_tensors_(std::move(input_tensors)),
        dataset_(dataset),
        scoped_unref_(dataset) {}

  DatasetBase* dataset() const { return dataset_; }

  OpKernelContext* op_kernel_context() const { return ctx_.get(); }

 protected:
  std::unique_ptr<OpKernel> kernel_;
  std::unique_ptr<OpKernelContext::Params> ctx_params_;
  std::unique_ptr<OpKernelContext> ctx_;
  // The input tensors that this dataset depends on. They must outlive the
  // dataset.
  std::vector<std::unique_ptr<Tensor>> input_tensors_;
  DatasetBase* dataset_;
  core::ScopedUnref scoped_unref_;
};

// Class composing a dataset iterator with its dependencies.
class TestIterator {
 public:
  TestIterator(std::unique_ptr<IteratorContext> ctx,
               std::unique_ptr<IteratorBase> iterator)
      : iterator_(std::move(iterator)), ctx_(std::move(ctx)) {}

  IteratorBase* iterator() const { return iterator_.get(); }

  IteratorContext* ctx() const { return ctx_.get(); }

  absl::Status GetNext(std::vector<Tensor>* out_tensors,
                       bool* end_of_sequence) {
    return iterator_->GetNext(ctx(), out_tensors, end_of_sequence);
  }

 protected:
  std::unique_ptr<IteratorBase> iterator_;
  std::unique_ptr<IteratorContext> ctx_;
};

// Helpful functions to test Dataset op kernels.
class DatasetOpsTestBase : public ::testing::Test {
 public:
  DatasetOpsTestBase();

  // Initializes the runtime and creates a dataset and iterator.
  absl::Status Initialize(const DatasetParams& dataset_params);

  // Initializes the parts of the runtime needed to run dataset ops.
  absl::Status InitializeRuntime(const DatasetParams& dataset_params);

  // Creates a dataset.
  absl::Status MakeDataset(const DatasetParams& dataset_params,
                           std::unique_ptr<TestDataset>* dataset);

  // Creates an iterator for the given dataset, using the specified split
  // providers.
  absl::Status MakeIterator(
      const DatasetParams& dataset_params, const TestDataset& dataset,
      std::vector<std::unique_ptr<SplitProvider>> split_providers,
      std::unique_ptr<TestIterator>* iterator);
  // Creates an iterator for the given dataset.
  absl::Status MakeIterator(const DatasetParams& dataset_params,
                            const TestDataset& dataset,
                            std::unique_ptr<TestIterator>* iterator);

  // Runs the dataset operation according to the predefined dataset params and
  // produces outputs. Different from `MakeDataset()` which returns a Dataset
  // object, `RunDatasetOp()` executes the dataset kernel based on the input
  // DatasetParams and returns the produced outputs as a tensor vector. It can
  // be used to run some dataset operations that do not have an internal
  // customized `Dataset` class (e.g. `ReduceDatasetOp`).
  absl::Status RunDatasetOp(const DatasetParams& dataset_params,
                            std::vector<Tensor>* outputs);

  // The method validates whether the two tensors have the same shape, dtype,
  // and value.
  static absl::Status ExpectEqual(const Tensor& a, const Tensor& b);

  // The method validates whether the two tensor vectors have the same tensors.
  // If `compare_order` is false, the method will only evaluate whether the two
  // vectors have the same elements regardless of order.
  static absl::Status ExpectEqual(std::vector<Tensor> produced_tensors,
                                  std::vector<Tensor> expected_tensors,
                                  bool compare_order);

  // Checks `IteratorBase::GetNext()`.
  absl::Status CheckIteratorGetNext(const std::vector<Tensor>& expected_outputs,
                                    bool compare_order);

  // Checks `IteratorBase::GetNext()`.
  absl::Status CheckIteratorGetNext(TestIterator* iterator,
                                    const std::vector<Tensor>& expected_outputs,
                                    bool compare_order);

  // Checks `IteratorBase::GetNext()`.
  absl::Status CheckIteratorGetNext(IteratorBase* iterator,
                                    IteratorContext* ctx,
                                    const std::vector<Tensor>& expected_outputs,
                                    bool compare_order);

  // Checks `IteratorBase::Skip()`
  absl::Status CheckIteratorSkip(int num_to_skip, int expected_num_skipped,
                                 bool get_next,
                                 const std::vector<Tensor>& expected_outputs,
                                 bool compare_order);

  // Checks that iterating through the dataset using a split provider produces
  // the expected outputs.
  absl::Status CheckSplitProviderFullIteration(
      const DatasetParams& params, const std::vector<Tensor>& expected_outputs);

  // Checks that iterating through the dataset using a sharded split provider
  // with the given `num_shards` and `shard_index` produces the expected
  // outputs.
  absl::Status CheckSplitProviderShardedIteration(
      const DatasetParams& params, int64_t num_shards, int64_t shard_index,
      const std::vector<Tensor>& expected_outputs);

  // Checks `DatasetBase::node_name()`.
  absl::Status CheckDatasetNodeName(const string& expected_dataset_node_name);

  // Checks `DatasetBase::type_string()`.
  absl::Status CheckDatasetTypeString(const string& expected_type_str);

  // Checks `DatasetBase::output_dtypes()`.
  absl::Status CheckDatasetOutputDtypes(
      const DataTypeVector& expected_output_dtypes);

  // Checks `DatasetBase::output_shapes()`.
  absl::Status CheckDatasetOutputShapes(
      const std::vector<PartialTensorShape>& expected_output_shapes);

  // Checks `DatasetBase::Cardinality()`.
  absl::Status CheckDatasetCardinality(int expected_cardinality);

  // Checks `DatasetBase::options()`.
  absl::Status CheckDatasetOptions(const Options& expected_options);

  // Checks `IteratorBase::output_dtypes()`.
  absl::Status CheckIteratorOutputDtypes(
      const DataTypeVector& expected_output_dtypes);

  // Checks `IteratorBase::output_shapes()`.
  absl::Status CheckIteratorOutputShapes(
      const std::vector<PartialTensorShape>& expected_output_shapes);

  // Checks `IteratorBase::prefix()`.
  absl::Status CheckIteratorPrefix(const string& expected_iterator_prefix);

  absl::Status CheckIteratorSaveAndRestore(
      DatasetBase* dataset, IteratorContext* iterator_ctx,
      const std::string& iterator_prefix,
      const std::vector<Tensor>& expected_outputs,
      const std::vector<int>& breakpoints, bool compare_order);

  absl::Status CheckIteratorSaveAndRestore(
      const std::string& iterator_prefix,
      const std::vector<Tensor>& expected_outputs,
      const std::vector<int>& breakpoints, bool compare_order);

  // A class for testing variant tensors.
  class TestVariant {
   public:
    TestVariant() = default;
    explicit TestVariant(const std::vector<Tensor>& tensors)
        : tensors_(tensors) {}

    bool operator!=(const TestVariant& rhs) const {
      return !ExpectEqual(tensors_, rhs.tensors_, /*compare_order=*/true).ok();
    }

    constexpr static const char kTypeName[] = "tensorflow::data::TestVariant";

    string TypeName() const { return kTypeName; }

    // Encodes the contents of this object into `data`.  This function signature
    // is required for objects to be stored in `tensorflow::Variant`s.  See the
    // docs for `tensorflow::Variant` for more information and see
    // `tensorflow::Variant::Encode` for how this is used.
    void Encode(VariantTensorData* data) const {
      data->set_type_name(TypeName());
      for (const auto& tensor : tensors_) {
        data->add_tensor(tensor);
      }
    }

    // Decodes `data` and updates the contents of this object.  This function
    // signature is required for objects to be stored in `tensorflow::Variant`s.
    // See the docs for `tensorflow::Variant` for more information and see
    // `tensorflow::Variant::Decode` for how this is used.
    bool Decode(VariantTensorData data) {
      tensors_ = data.tensors();
      return true;
    }

    string DebugString() const {
      string result = "TestVariant([";
      for (const auto& tensor : tensors_) {
        if (&tensor != &tensors_[0]) result += ", ";
        result += tensor.DebugString();
      }
      result += "])";
      return result;
    }

   private:
    std::vector<Tensor> tensors_;
  };

  // Returns a scalar variant tensor containing a `TestVariant` object
  // containing `tensors`.
  static Tensor CreateTestVariantTensor(const std::vector<Tensor>& tensors) {
    Tensor tensor{DT_VARIANT, TensorShape({})};
    TestVariant test_variant{tensors};
    tensor.scalar<Variant>()() = test_variant;
    return tensor;
  }

 protected:
  // Make destructor protected so that DatasetOpsTestBase objects cannot
  // be instantiated directly. Only subclasses can be instantiated.
  ~DatasetOpsTestBase() override;

  // Creates a thread pool for parallel tasks.
  absl::Status InitThreadPool(int thread_num);

  // Initializes the runtime for computing the dataset operation and registers
  // the input function definitions. `InitThreadPool()' needs to be called
  // before this method if we want to run the tasks in parallel.
  absl::Status InitFunctionLibraryRuntime(const std::vector<FunctionDef>& flib,
                                          int cpu_num);

  // Creates a new op kernel based on the node definition.
  absl::Status CreateOpKernel(const NodeDef& node_def,
                              std::unique_ptr<OpKernel>* op_kernel);

  // Creates a new op kernel context.
  absl::Status CreateDatasetContext(
      OpKernel* dateset_kernel, absl::InlinedVector<TensorValue, 4>* inputs,
      std::unique_ptr<OpKernelContext::Params>* dataset_context_params,
      std::unique_ptr<OpKernelContext>* dataset_context);

  // Creates a new dataset.
  absl::Status CreateDataset(OpKernel* kernel, OpKernelContext* context,
                             DatasetBase** dataset);

  // Restores the state of the input iterator. It resets the iterator before
  // restoring it to make sure the input iterator does not hold any
  // resources or tasks. Otherwise, restoring an existing iterator may cause
  // the timeout issue or duplicated elements.
  absl::Status RestoreIterator(IteratorContext* ctx,
                               IteratorStateReader* reader,
                               const string& output_prefix,
                               const DatasetBase& dataset,
                               std::unique_ptr<IteratorBase>* iterator);

  // Fetches the dataset from the operation context.
  absl::Status GetDatasetFromContext(OpKernelContext* context, int output_index,
                                     DatasetBase** dataset);

  // Runs an operation producing outputs.
  absl::Status RunOpKernel(OpKernel* op_kernel, OpKernelContext* context);

  // Executes a function producing outputs.
  absl::Status RunFunction(const FunctionDef& fdef, test::function::Attrs attrs,
                           const std::vector<Tensor>& args,
                           const GraphConstructorOptions& graph_options,
                           std::vector<Tensor*> rets);

  // Checks that the size of `inputs` matches the requirement of the op kernel.
  absl::Status CheckOpKernelInput(
      const OpKernel& kernel,
      const absl::InlinedVector<TensorValue, 4>& inputs);

  // Creates a new context for running the dataset operation.
  absl::Status CreateOpKernelContext(
      OpKernel* kernel, absl::InlinedVector<TensorValue, 4>* inputs,
      std::unique_ptr<OpKernelContext>* context);

  // Creates a new context for running the dataset operation.
  absl::Status CreateOpKernelContext(
      OpKernel* kernel, absl::InlinedVector<TensorValue, 4>* inputs,
      std::unique_ptr<OpKernelContext::Params>* params,
      std::unique_ptr<OpKernelContext>* context);

  // Creates a new iterator context for iterating the dataset.
  absl::Status CreateIteratorContext(
      OpKernelContext* op_context,
      std::unique_ptr<IteratorContext>* iterator_context);

  // Creates a new iterator context for iterating the dataset.
  // Creates a new serialization context for serializing the dataset and
  // iterator.
  absl::Status CreateSerializationContext(
      std::unique_ptr<SerializationContext>* context);

  // Creates the dataset op kernel.
  absl::Status MakeGetOptionsOpKernel(const DatasetParams& dataset_params,
                                      std::unique_ptr<OpKernel>* op_kernel);

 private:
  // Runs the dataset operation according to the predefined dataset params and
  // the produced outputs will be stored in `dataset_ctx`.
  absl::Status RunDatasetOp(
      const DatasetParams& dataset_params,
      std::unique_ptr<OpKernel>* dataset_kernel,
      std::unique_ptr<OpKernelContext::Params>* dataset_ctx_params,
      std::vector<std::unique_ptr<Tensor>>* created_tensors,
      std::unique_ptr<OpKernelContext>* dataset_ctx);

  absl::Status MakeDataset(
      const DatasetParams& dataset_params,
      std::unique_ptr<OpKernel>* dataset_kernel,
      std::unique_ptr<OpKernelContext::Params>* dataset_ctx_params,
      std::unique_ptr<OpKernelContext>* dataset_ctx,
      std::vector<std::unique_ptr<Tensor>>* created_tensors,
      DatasetBase** dataset);

  // Creates the dataset op kernel.
  absl::Status MakeDatasetOpKernel(const DatasetParams& dataset_params,
                                   std::unique_ptr<OpKernel>* dataset_kernel);

  // Creates a dataset tensor according to the input dataset params.
  absl::Status MakeDatasetTensor(
      const DatasetParams& dataset_params,
      std::vector<std::unique_ptr<Tensor>>* created_tensors,
      std::unique_ptr<Tensor>* dataset);

  // Adds an empty tensor with the specified dtype and shape to the input
  // vector.
  absl::Status AddDatasetInput(absl::InlinedVector<TensorValue, 4>* inputs,
                               DataTypeVector input_types, DataType dtype,
                               const TensorShape& shape);

 protected:
  std::unique_ptr<Device> device_;
  DeviceType device_type_;
  int cpu_num_;
  int thread_num_;
  Allocator* allocator_;  // Owned by `AllocatorFactoryRegistry`.
  std::vector<AllocatorAttributes> allocator_attrs_;
  std::unique_ptr<ScopedStepContainer> step_container_;

  // Device manager is used by function handle cache and needs to outlive it.
  std::unique_ptr<DeviceMgr> device_mgr_;
  std::unique_ptr<ProcessFunctionLibraryRuntime> pflr_;
  FunctionLibraryRuntime* flr_;  // Owned by `pflr_`.
  std::unique_ptr<FunctionHandleCache> function_handle_cache_;
  std::function<void(std::function<void()>)> runner_;
  std::unique_ptr<FunctionLibraryDefinition> lib_def_;
  std::unique_ptr<ResourceMgr> resource_mgr_;
  std::unique_ptr<checkpoint::TensorSliceReaderCacheWrapper>
      slice_reader_cache_;
  std::unique_ptr<thread::ThreadPool> thread_pool_;
  std::vector<std::unique_ptr<Tensor>> tensors_;  // Owns tensors.
  mutex lock_for_refs_;  // Used as the Mutex for inputs added as refs.
  std::unique_ptr<CancellationManager> cancellation_manager_;

  // Indicates if the below fields have been initialized.
  bool initialized_ = false;
  std::unique_ptr<OpKernel> dataset_kernel_;
  std::unique_ptr<OpKernelContext::Params> params_;
  std::unique_ptr<OpKernelContext> dataset_ctx_;
  DatasetBase* dataset_ = nullptr;
  std::unique_ptr<IteratorContext> iterator_ctx_;
  std::unique_ptr<IteratorBase> iterator_;
};

#define ITERATOR_GET_NEXT_TEST_P(dataset_op_test_class, dataset_params_class, \
                                 test_cases)                                  \
  class ParameterizedGetNextTest                                              \
      : public dataset_op_test_class,                                         \
        public ::testing::WithParamInterface<                                 \
            GetNextTestCase<dataset_params_class>> {};                        \
                                                                              \
  TEST_P(ParameterizedGetNextTest, GetNext) {                                 \
    auto test_case = GetParam();                                              \
    TF_ASSERT_OK(Initialize(test_case.dataset_params));                       \
    TF_ASSERT_OK(                                                             \
        CheckIteratorGetNext(test_case.expected_outputs,                      \
                             /*compare_order=*/test_case.compare_order));     \
  }                                                                           \
                                                                              \
  INSTANTIATE_TEST_SUITE_P(                                                   \
      dataset_op_test_class, ParameterizedGetNextTest,                        \
      ::testing::ValuesIn(                                                    \
          std::vector<GetNextTestCase<dataset_params_class>>(test_cases)));

#define ITERATOR_SKIP_TEST_P(dataset_op_test_class, dataset_params_class,   \
                             test_cases)                                    \
  class ParameterizedSkipTest : public dataset_op_test_class,               \
                                public ::testing::WithParamInterface<       \
                                    SkipTestCase<dataset_params_class>> {}; \
                                                                            \
  TEST_P(ParameterizedSkipTest, Skip) {                                     \
    auto test_case = GetParam();                                            \
    TF_ASSERT_OK(Initialize(test_case.dataset_params));                     \
    TF_ASSERT_OK(CheckIteratorSkip(                                         \
        test_case.num_to_skip, test_case.expected_num_skipped,              \
        test_case.get_next, test_case.expected_outputs,                     \
        /*compare_order=*/test_case.compare_order));                        \
  }                                                                         \
                                                                            \
  INSTANTIATE_TEST_SUITE_P(                                                 \
      dataset_op_test_class, ParameterizedSkipTest,                         \
      ::testing::ValuesIn(                                                  \
          std::vector<SkipTestCase<dataset_params_class>>(test_cases)));

#define DATASET_NODE_NAME_TEST_P(dataset_op_test_class, dataset_params_class, \
                                 test_cases)                                  \
  class ParameterizedDatasetNodeNameTest                                      \
      : public dataset_op_test_class,                                         \
        public ::testing::WithParamInterface<                                 \
            DatasetNodeNameTestCase<dataset_params_class>> {};                \
                                                                              \
  TEST_P(ParameterizedDatasetNodeNameTest, DatasetNodeName) {                 \
    auto test_case = GetParam();                                              \
    TF_ASSERT_OK(Initialize(test_case.dataset_params));                       \
    TF_ASSERT_OK(CheckDatasetNodeName(test_case.expected_node_name));         \
  }                                                                           \
                                                                              \
  INSTANTIATE_TEST_SUITE_P(                                                   \
      dataset_op_test_class, ParameterizedDatasetNodeNameTest,                \
      ::testing::ValuesIn(                                                    \
          std::vector<DatasetNodeNameTestCase<dataset_params_class>>(         \
              test_cases)));

#define DATASET_TYPE_STRING_TEST_P(dataset_op_test_class,                \
                                   dataset_params_class, test_cases)     \
  class ParameterizedDatasetTypeStringTest                               \
      : public dataset_op_test_class,                                    \
        public ::testing::WithParamInterface<                            \
            DatasetTypeStringTestCase<dataset_params_class>> {};         \
                                                                         \
  TEST_P(ParameterizedDatasetTypeStringTest, DatasetTypeString) {        \
    auto test_case = GetParam();                                         \
    TF_ASSERT_OK(Initialize(test_case.dataset_params));                  \
    TF_ASSERT_OK(                                                        \
        CheckDatasetTypeString(test_case.expected_dataset_type_string)); \
  }                                                                      \
                                                                         \
  INSTANTIATE_TEST_SUITE_P(                                              \
      dataset_op_test_class, ParameterizedDatasetTypeStringTest,         \
      ::testing::ValuesIn(                                               \
          std::vector<DatasetTypeStringTestCase<dataset_params_class>>(  \
              test_cases)));

#define DATASET_OUTPUT_DTYPES_TEST_P(dataset_op_test_class,                   \
                                     dataset_params_class, test_cases)        \
                                                                              \
  class ParameterizedDatasetOutputDtypesTest                                  \
      : public dataset_op_test_class,                                         \
        public ::testing::WithParamInterface<                                 \
            DatasetOutputDtypesTestCase<dataset_params_class>> {};            \
                                                                              \
  TEST_P(ParameterizedDatasetOutputDtypesTest, DatasetOutputDtypes) {         \
    auto test_case = GetParam();                                              \
    TF_ASSERT_OK(Initialize(test_case.dataset_params));                       \
    TF_ASSERT_OK(CheckDatasetOutputDtypes(test_case.expected_output_dtypes)); \
  }                                                                           \
                                                                              \
  INSTANTIATE_TEST_SUITE_P(                                                   \
      dataset_op_test_class, ParameterizedDatasetOutputDtypesTest,            \
      ::testing::ValuesIn(                                                    \
          std::vector<DatasetOutputDtypesTestCase<dataset_params_class>>(     \
              test_cases)));

#define DATASET_OUTPUT_SHAPES_TEST_P(dataset_op_test_class,                   \
                                     dataset_params_class, test_cases)        \
                                                                              \
  class ParameterizedDatasetOutputShapesTest                                  \
      : public dataset_op_test_class,                                         \
        public ::testing::WithParamInterface<                                 \
            DatasetOutputShapesTestCase<dataset_params_class>> {};            \
                                                                              \
  TEST_P(ParameterizedDatasetOutputShapesTest, DatasetOutputShapes) {         \
    auto test_case = GetParam();                                              \
    TF_ASSERT_OK(Initialize(test_case.dataset_params));                       \
    TF_ASSERT_OK(CheckDatasetOutputShapes(test_case.expected_output_shapes)); \
  }                                                                           \
                                                                              \
  INSTANTIATE_TEST_SUITE_P(                                                   \
      dataset_op_test_class, ParameterizedDatasetOutputShapesTest,            \
      ::testing::ValuesIn(                                                    \
          std::vector<DatasetOutputShapesTestCase<dataset_params_class>>(     \
              test_cases)));

#define DATASET_CARDINALITY_TEST_P(dataset_op_test_class,                  \
                                   dataset_params_class, test_cases)       \
                                                                           \
  class ParameterizedCardinalityTest                                       \
      : public dataset_op_test_class,                                      \
        public ::testing::WithParamInterface<                              \
            CardinalityTestCase<dataset_params_class>> {};                 \
                                                                           \
  TEST_P(ParameterizedCardinalityTest, Cardinality) {                      \
    auto test_case = GetParam();                                           \
    TF_ASSERT_OK(Initialize(test_case.dataset_params));                    \
    TF_ASSERT_OK(CheckDatasetCardinality(test_case.expected_cardinality)); \
  }                                                                        \
                                                                           \
  INSTANTIATE_TEST_SUITE_P(                                                \
      dataset_op_test_class, ParameterizedCardinalityTest,                 \
      ::testing::ValuesIn(                                                 \
          std::vector<CardinalityTestCase<dataset_params_class>>(          \
              test_cases)));

#define ITERATOR_OUTPUT_DTYPES_TEST_P(dataset_op_test_class,                  \
                                      dataset_params_class, test_cases)       \
  class ParameterizedIteratorOutputDtypesTest                                 \
      : public dataset_op_test_class,                                         \
        public ::testing::WithParamInterface<                                 \
            IteratorOutputDtypesTestCase<dataset_params_class>> {};           \
                                                                              \
  TEST_P(ParameterizedIteratorOutputDtypesTest, IteratorOutputDtypes) {       \
    auto test_case = GetParam();                                              \
    TF_ASSERT_OK(Initialize(test_case.dataset_params));                       \
    TF_ASSERT_OK(CheckDatasetOutputDtypes(test_case.expected_output_dtypes)); \
  }                                                                           \
                                                                              \
  INSTANTIATE_TEST_SUITE_P(                                                   \
      dataset_op_test_class, ParameterizedIteratorOutputDtypesTest,           \
      ::testing::ValuesIn(                                                    \
          std::vector<IteratorOutputDtypesTestCase<dataset_params_class>>(    \
              test_cases)));

#define ITERATOR_OUTPUT_SHAPES_TEST_P(dataset_op_test_class,                   \
                                      dataset_params_class, test_cases)        \
  class ParameterizedIteratorOutputShapesTest                                  \
      : public dataset_op_test_class,                                          \
        public ::testing::WithParamInterface<                                  \
            IteratorOutputShapesTestCase<dataset_params_class>> {};            \
                                                                               \
  TEST_P(ParameterizedIteratorOutputShapesTest, IteratorOutputShapes) {        \
    auto test_case = GetParam();                                               \
    TF_ASSERT_OK(Initialize(test_case.dataset_params));                        \
    TF_ASSERT_OK(CheckIteratorOutputShapes(test_case.expected_output_shapes)); \
  }                                                                            \
                                                                               \
  INSTANTIATE_TEST_SUITE_P(                                                    \
      dataset_op_test_class, ParameterizedIteratorOutputShapesTest,            \
      ::testing::ValuesIn(                                                     \
          std::vector<IteratorOutputShapesTestCase<dataset_params_class>>(     \
              test_cases)));

#define ITERATOR_PREFIX_TEST_P(dataset_op_test_class, dataset_params_class, \
                               test_cases)                                  \
  class ParameterizedIteratorPrefixTest                                     \
      : public dataset_op_test_class,                                       \
        public ::testing::WithParamInterface<                               \
            IteratorPrefixTestCase<dataset_params_class>> {};               \
                                                                            \
  TEST_P(ParameterizedIteratorPrefixTest, IteratorPrefix) {                 \
    auto test_case = GetParam();                                            \
    TF_ASSERT_OK(Initialize(test_case.dataset_params));                     \
    TF_ASSERT_OK(CheckIteratorPrefix(test_case.expected_iterator_prefix));  \
  }                                                                         \
                                                                            \
  INSTANTIATE_TEST_SUITE_P(                                                 \
      dataset_op_test_class, ParameterizedIteratorPrefixTest,               \
      ::testing::ValuesIn(                                                  \
          std::vector<IteratorPrefixTestCase<dataset_params_class>>(        \
              test_cases)));

#define ITERATOR_SAVE_AND_RESTORE_TEST_P(dataset_op_test_class,              \
                                         dataset_params_class, test_cases)   \
  class ParameterizedIteratorSaveAndRestoreTest                              \
      : public dataset_op_test_class,                                        \
        public ::testing::WithParamInterface<                                \
            IteratorSaveAndRestoreTestCase<dataset_params_class>> {};        \
  TEST_P(ParameterizedIteratorSaveAndRestoreTest, IteratorSaveAndRestore) {  \
    auto test_case = GetParam();                                             \
    TF_ASSERT_OK(Initialize(test_case.dataset_params));                      \
    TF_ASSERT_OK(CheckIteratorSaveAndRestore(                                \
        test_case.dataset_params.iterator_prefix(),                          \
        test_case.expected_outputs, test_case.breakpoints,                   \
        test_case.compare_order));                                           \
  }                                                                          \
  INSTANTIATE_TEST_SUITE_P(                                                  \
      dataset_op_test_class, ParameterizedIteratorSaveAndRestoreTest,        \
      ::testing::ValuesIn(                                                   \
          std::vector<IteratorSaveAndRestoreTestCase<dataset_params_class>>( \
              test_cases)));

}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DATA_DATASET_TEST_BASE_H_
