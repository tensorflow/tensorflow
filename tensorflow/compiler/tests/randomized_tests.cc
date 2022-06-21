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

// Randomized tests for XLA implementations of Tensorflow operations.
//
// For each operator, the tests in this file choose a set of random inputs and
// attributes. The test then compares the outputs of the operator when executed
// via Tensorflow using the CPU device and when executed via XLA.
//
// By default, each test chooses a random seed nondeterministically (using
// std::random_device). However, a particular choice of random seed can be
// forced using the flag --tf_xla_random_seed; each test logs the
// flag value necessary to reproduce its outputs.
//
// Example usage:
// Run tests, comparing the Tensorflow CPU operators with their XLA-compiled
// counterparts:
// randomized_tests \
//   --tf_xla_test_use_jit=true --tf_xla_test_device=CPU:0 \
//   --tf_xla_test_repetitions=20

// TODO(phawkins): add tests for:
// * DepthwiseConv2DNative
// * Gather
// * InvertPermutation
// * MaxPoolGrad (requires implementation of forward operator)
// * Select
// * Unpack
//
// TODO(phawkins): improve tests for:
// * StridedSliceGrad (need to use shape function to compute sensible inputs)

#include <algorithm>
#include <random>
#include <unordered_map>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "tensorflow/compiler/jit/defs.h"
#include "tensorflow/compiler/jit/flags.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/framework/kernel_shape_util.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/bfloat16.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/util/command_line_flags.h"
#include "tensorflow/core/util/device_name_utils.h"
#include "tensorflow/core/util/tensor_format.h"

namespace tensorflow {
namespace {

// Command line flags: see main() below.
int64_t tf_xla_random_seed = 0;
int32_t tf_xla_test_repetitions = 20;
int64_t tf_xla_max_tensor_size = 10000LL;
string* tf_xla_test_device_ptr;       // initial value set in main()
string* tf_xla_reference_device_ptr;  // initial value set in main()
bool tf_xla_test_use_jit = true;
bool tf_xla_test_use_mlir = false;

string LocalDeviceToFullDeviceName(const string& device) {
  return absl::StrCat("/job:localhost/replica:0/task:0/device:", device);
}

constexpr std::array<DataType, 5> kAllXlaTypes = {
    {DT_INT32, DT_INT64, DT_FLOAT, DT_BOOL, DT_COMPLEX64}};
constexpr std::array<DataType, 4> kAllNumberTypes = {
    {DT_INT32, DT_INT64, DT_FLOAT, DT_COMPLEX64}};

// An OpTestBuilder is a graph builder class that takes as input an operator to
// test, its inputs and attributes, and builds a graph that executes the
// operator.
class OpTestBuilder {
 public:
  explicit OpTestBuilder(const string& op_name);

  // Adds an input 'tensor' as a Placeholder node.
  OpTestBuilder& Input(const Tensor& tensor);

  // Adds a random input tensor with 'type' as a Placeholder node.
  // If 'dims' is not provided, RandomDims() is used.
  OpTestBuilder& RandomInput(DataType type);
  OpTestBuilder& RandomInput(DataType type, std::vector<int64_t> dims);

  // As RandomInput but the values are unique.
  OpTestBuilder& RandomUniqueInput(DataType type, std::vector<int64_t> dims);

  // Add variadic input tensors as Placehodler nodes.
  OpTestBuilder& VariadicInput(const std::vector<Tensor>& tensor);

  // Sets an attribute.
  template <class T>
  OpTestBuilder& Attr(absl::string_view attr_name, T&& value);

  // Overload needed to allow {...} expressions for value.
  template <class T>
  OpTestBuilder& Attr(absl::string_view attr_name,
                      std::initializer_list<T> value);

  // Adds nodes that executes the operator under test on 'device' to 'graphdef'.
  // If 'use_jit' is true, marks the operator under test to be compiled by XLA.
  // The graph will consist of one Placeholder node per input, the operator
  // itself, and one Identity node per output. If 'test_node_def' is not null,
  // sets it to the NodeDef of the operator under test. Fills 'inputs' and
  // 'outputs' with the names of the input placeholder nodes and the output
  // identity nodes, respectively.
  Status BuildGraph(const string& name_prefix, const string& device,
                    bool use_jit, GraphDef* graphdef, NodeDef** test_node_def,
                    std::vector<string>* inputs,
                    std::vector<string>* outputs) const;

  struct InputDescription {
    Tensor tensor;

    DataType type = DT_INVALID;
    bool has_dims = false;
    bool needs_unique_values = false;
    std::vector<int64_t> dims;
  };

  const std::vector<InputDescription>& inputs() const { return inputs_; }

 private:
  NodeDef node_def_;
  std::vector<InputDescription> inputs_;
};

OpTestBuilder::OpTestBuilder(const string& op_name) {
  node_def_.set_op(op_name);
}

OpTestBuilder& OpTestBuilder::Input(const Tensor& tensor) {
  VLOG(1) << "Adding input: " << tensor.DebugString();
  InputDescription input;
  input.tensor = tensor;
  inputs_.push_back(input);
  return *this;
}

OpTestBuilder& OpTestBuilder::RandomInput(DataType type) {
  VLOG(1) << "Adding random input: " << type;
  InputDescription input;
  input.type = type;
  inputs_.push_back(input);
  return *this;
}

OpTestBuilder& OpTestBuilder::RandomInput(DataType type,
                                          std::vector<int64_t> dims) {
  VLOG(1) << "Adding input: " << type << " " << TensorShape(dims).DebugString();
  InputDescription input;
  input.type = type;
  input.has_dims = true;
  input.dims = std::move(dims);
  inputs_.push_back(input);
  return *this;
}

OpTestBuilder& OpTestBuilder::RandomUniqueInput(DataType type,
                                                std::vector<int64_t> dims) {
  VLOG(1) << "Adding input: " << type << " " << TensorShape(dims).DebugString();
  InputDescription input;
  input.type = type;
  input.has_dims = true;
  input.needs_unique_values = true;
  input.dims = std::move(dims);
  inputs_.push_back(input);
  return *this;
}

OpTestBuilder& OpTestBuilder::VariadicInput(
    const std::vector<Tensor>& tensors) {
  VLOG(1) << "Adding variadic input of length " << tensors.size() << ":";
  for (auto& t : tensors) {
    Input(t);
  }
  return *this;
}

template <class T>
OpTestBuilder& OpTestBuilder::Attr(absl::string_view attr_name, T&& value) {
  AddNodeAttr(attr_name, std::forward<T>(value), &node_def_);
  return *this;
}

template <class T>
OpTestBuilder& OpTestBuilder::Attr(absl::string_view attr_name,
                                   std::initializer_list<T> value) {
  Attr<std::initializer_list<T>>(attr_name, std::move(value));
  return *this;
}

Status OpTestBuilder::BuildGraph(const string& name_prefix,
                                 const string& device, bool use_jit,
                                 GraphDef* graphdef, NodeDef** test_node_def,
                                 std::vector<string>* inputs,
                                 std::vector<string>* outputs) const {
  OpRegistryInterface* op_registry = OpRegistry::Global();

  const OpDef* op_def;
  TF_RETURN_IF_ERROR(op_registry->LookUpOpDef(node_def_.op(), &op_def));

  NodeDef* test_def = graphdef->add_node();
  *test_def = node_def_;
  test_def->set_name(absl::StrCat(name_prefix, "_op_under_test"));
  test_def->set_device(device);
  AddDefaultsToNodeDef(*op_def, test_def);
  if (use_jit) {
    AddNodeAttr(kXlaCompileAttr, true, test_def);
  }
  VLOG(1) << "Op under test: " << test_def->DebugString();

  DataTypeVector input_types, output_types;
  TF_RETURN_IF_ERROR(
      InOutTypesForNode(*test_def, *op_def, &input_types, &output_types));

  // Build feed and fetch nodes.
  for (int i = 0; i < input_types.size(); ++i) {
    NodeDef* def = graphdef->add_node();
    string name = absl::StrCat(name_prefix, "_input_", i);
    TF_RETURN_IF_ERROR(NodeDefBuilder(name, "Placeholder")
                           .Device(device)
                           .Attr("dtype", input_types[i])
                           .Finalize(def));
    inputs->push_back(name);
    test_def->add_input(name);
  }

  for (int i = 0; i < output_types.size(); ++i) {
    NodeDef* def = graphdef->add_node();
    string name = absl::StrCat(name_prefix, "_output_", i);
    TF_RETURN_IF_ERROR(NodeDefBuilder(name, "Identity")
                           .Device(device)
                           .Attr("T", output_types[i])
                           .Input(test_def->name(), i, output_types[i])
                           .Finalize(def));
    outputs->push_back(name);
  }

  if (test_node_def) {
    *test_node_def = test_def;
  }

  return OkStatus();
}

// Test fixture. The fixture manages the random number generator and its seed,
// and has a number of convenience methods for building random Tensors, shapes,
// etc.
class OpTest : public ::testing::Test {
 public:
  OpTest();

  enum TestResult {
    // The test saw an unrecoverable error. Don't try any more runs.
    kFatalError,
    // The parameters of the test were invalid (e.g., the "golden"
    // implementation failed, or the parameters are oversize). Reruns are ok.
    kInvalid,
    // The test ran successfully, and we have a verdict. Does *not* mean the
    // test passed.
    kOk,
  };

  // Runs 'fn' up to --tf_xla_test_repetitions times, or until a test failure
  // occurs; whichever happens first. Reruns if the TestResult is kInvalid.
  void Repeatedly(const std::function<TestResult(void)>& fn);

  // Select a random element from 'candidates'.
  template <typename T>
  T Choose(absl::Span<const T> candidates);

  static constexpr int kDefaultMaxRank = 5;
  static constexpr int64_t kDefaultMaxDimensionSize = 256LL;

  // Returns true if 'dims' have a size less than tf_xla_max_tensor_size.
  bool TensorSizeIsOk(absl::Span<const int64_t> dims);

  // Returns a random dimension size, in the range [min, max).
  int64_t RandomDim(int64_t min = 0, int64_t max = kDefaultMaxDimensionSize);

  // Returns a random shape. The tensor has rank in the range [min_rank,
  // max_rank). Each dimension has size [min_size, max_size).
  std::vector<int64_t> RandomDims(int min_rank = 0,
                                  int max_rank = kDefaultMaxRank,
                                  int64_t min_size = 0,
                                  int64_t max_size = kDefaultMaxDimensionSize);

  // Given a shape 'dims', build dimensions that are broadcastable to 'dims'.
  std::vector<int64_t> BroadcastableToDims(std::vector<int64_t> dims);

  // Given a shape 'dims', build a pair of dimensions such that one broadcasts
  // to the other.
  std::pair<std::vector<int64_t>, std::vector<int64_t>> BroadcastableDims(
      std::vector<int64_t> dims);

  // Builds a random pair of broadcastable dims.
  // TODO(phawkins): currently the maximum rank is 3, because broadcasting > 3
  // dimensions is unimplemented by the Tensorflow Eigen code (b/29268487)
  std::pair<std::vector<int64_t>, std::vector<int64_t>> BroadcastableDims();

  // Returns a tensor filled with random but "reasonable" values from the middle
  // of the type's range. If the shape is omitted, a random shape is used.
  // TODO(phawkins): generalize this code to a caller-supplied distribution.
  Tensor RandomTensor(DataType dtype, bool needs_unique_values,
                      absl::Span<const int64_t> shape);
  Tensor RandomTensor(DataType dtype);

  // Like RandomTensor, but uses values >= 0.
  Tensor RandomNonNegativeTensor(DataType dtype,
                                 absl::Span<const int64_t> shape);
  Tensor RandomNonNegativeTensor(DataType dtype);

  // Like RandomTensor, but all values are in the range [lo, hi].
  template <typename T>
  Tensor RandomBoundedTensor(DataType dtype, T lo, T hi,
                             bool needs_unique_values,
                             absl::Span<const int64_t> shape);
  template <typename T>
  Tensor RandomBoundedTensor(DataType dtype, T lo, T hi,
                             bool needs_unique_values);

  // Like RandomTensor, but the value at index i is in the range [lo[i], hi[i]].
  Tensor RandomBoundedTensor(DataType dtype, Tensor lo, Tensor hi);

  // Like RandomTensor, but return a pair {left, right} with
  // left[i] <= right[i].
  std::pair<Tensor, Tensor> RandomLteTensors(DataType dtype,
                                             absl::Span<const int64_t> shape);
  std::pair<Tensor, Tensor> RandomLteTensors(DataType dtype);

  // Returns a random subset of the integers in the range [0, rank), suitable
  // for use as reduction indices.
  Tensor RandomReductionIndices(int rank);

  // Returns a random bit.
  bool RandomBool();

  // Randomly choose a seed for a random number generator.
  int64_t RandomSeed();

  struct WindowedSpatialDims {
    Padding padding;
    std::vector<int64_t> kernel_dims;
    std::vector<int64_t> stride_dims;
    std::vector<int64_t> input_dims;
    std::vector<int64_t> output_dims;
  };
  // Choose spatial dimensions for a windowed op such as pooling or convolution.
  WindowedSpatialDims ChooseWindowedSpatialDims(int num_spatial_dims);

  struct BatchMatMulArguments {
    std::vector<int64_t> lhs_dims;
    std::vector<int64_t> rhs_dims;
    DataType dtype;
    bool adj_lhs;
    bool adj_rhs;
  };
  // Choose arguments for the tf.BatchMatMul{V2} ops.
  BatchMatMulArguments ChooseBatchMatMulArguments(bool broadcastable_batch);

  struct ConcatArguments {
    std::vector<Tensor> values;
    Tensor axis;
    int n;
    DataType type;
    DataType type_idx;
  };
  // Choose arguments for the tf.Concat{V2} ops.
  ConcatArguments ChooseConcatArguments(bool int64_idx_allowed);

  struct EinsumArguments {
    std::vector<int64_t> lhs_dims;
    std::vector<int64_t> rhs_dims;
    DataType type;
    std::string equation;
  };
  // Choose arguments for the tf.{Xla}Einsum ops.
  EinsumArguments ChooseEinsumArguments();

  struct GatherArguments {
    int64_t batch_dims;
    DataType axis_type;
    DataType indices_type;
    DataType params_type;
    std::vector<int64_t> params_shape;
    Tensor indices;
    Tensor axis;
  };
  // Choose arguments for the tf.Gather{V2} ops.
  GatherArguments ChooseGatherArguments(bool axis_0);

  struct PadArguments {
    DataType input_type;
    DataType paddings_type;
    std::vector<int64_t> input_shape;
    Tensor paddings;
    Tensor constant_values;
  };
  // Choose arguments for the tf.Pad{V2} ops.
  PadArguments ChoosePadArguments();

  struct ScatterArguments {
    DataType type;
    DataType indices_type;
    Tensor indices;
    Tensor updates;
    std::vector<int64_t> shape;
  };
  // Choose arguments for ScatterNd and TensorScatterUpdate.
  ScatterArguments ChooseScatterArguments();

  struct SliceArguments {
    DataType type;
    DataType indices_type;
    std::vector<int64_t> shape;
    Tensor indices;
    std::vector<int64_t> size;
  };
  // Choose arguments for the tf.{XlaDynamicUpdate}Slice ops.
  SliceArguments ChooseSliceArguments(bool neg_one_size);

  struct XlaDotArguments {
    std::vector<int64_t> lhs_dims;
    std::vector<int64_t> rhs_dims;
    std::string dnums_encoded;
    std::string precision_config_encoded;
    DataType dtype;
  };
  // Choose arguments for tf.XlaDot operation.
  XlaDotArguments ChooseXlaDotArguments();

  // Builds dimensions for a windowed op such as pooling or convolution,
  // including a batch and feature dimension.
  std::vector<int64_t> ImageDims(TensorFormat format, int batch, int feature,
                                 const std::vector<int64_t>& spatial_dims);

  // Converts an int64 vector to an int32 vector.
  std::vector<int32> AsInt32s(const std::vector<int64_t>& int64s);

  std::mt19937& generator() { return *generator_; }

  // Run the test case described by 'builder' with and without XLA and check
  // that the outputs are close. Tensors x and y are close if they have the same
  // type, same shape, and have close values. For floating-point tensors, the
  // element-wise difference between x and y must no more than
  // atol + rtol * abs(x); or both elements may be NaN or infinity. For
  // non-floating-point tensors the element values must match exactly.
  TestResult ExpectTfAndXlaOutputsAreClose(const OpTestBuilder& builder,
                                           double atol = 1e-2,
                                           double rtol = 1e-2);

 protected:
  // Per-test state:
  std::unique_ptr<std::mt19937> generator_;

  std::unique_ptr<Session> session_;

  // Number of test cases built in 'session_'. Used to uniquify node names.
  int num_tests_ = 0;
};

OpTest::OpTest() {
  // Creates a random-number generator for the test case. Use the value of
  // --tf_xla_random_seed as the seed, if provided.
  int64_t s = tf_xla_random_seed;
  unsigned int seed;
  if (s <= 0) {
    std::random_device random_device;
    seed = random_device();
  } else {
    seed = static_cast<unsigned int>(s);
  }
  LOG(ERROR) << "Random seed for test case: " << seed
             << ". To reproduce the "
                "results of this test, pass flag --tf_xla_random_seed="
             << seed;
  generator_.reset(new std::mt19937(seed));

  // Create a session with an empty graph.
  SessionOptions session_options;
  session_.reset(NewSession(session_options));
  GraphDef def;
  TF_CHECK_OK(session_->Create(def));
}

namespace {
template <typename T>
Tensor TensorFromValues(DataType dtype, absl::Span<const int64_t> shape,
                        absl::Span<T> vals) {
  Tensor tensor(dtype, TensorShape(shape));
  test::FillValues<T>(&tensor, vals);
  return tensor;
}

int64_t ShapeNumVals(absl::Span<const int64_t> shape) {
  int64_t num_vals = 1;
  for (int i = 0; i < shape.size(); ++i) {
    num_vals *= shape[i];
  }
  return num_vals;
}
}  // namespace

// TensorGenerator is an abstact class that has one implementing class for each
// (DataType,T) pair. The implementing class implements RandomVals, which is
// the only Tensor generation code that is specific to the DataType.
template <typename T>
class TensorGenerator {
 public:
  explicit TensorGenerator(OpTest& test) : test_(test) {}
  virtual ~TensorGenerator() {}
  virtual DataType dtype() = 0;
  virtual void RandomVals(absl::optional<T> lo, absl::optional<T> hi,
                          bool needs_unique_values,
                          absl::FixedArray<T>& vals) = 0;

  Tensor RandomTensor(absl::optional<T> lo, absl::optional<T> hi,
                      bool needs_unique_values,
                      absl::Span<const int64_t> shape) {
    absl::FixedArray<T> vals(ShapeNumVals(shape));
    RandomVals(lo, hi, needs_unique_values, vals);
    return TensorFromValues<T>(dtype(), shape, absl::Span<T>(vals));
  }

  std::pair<Tensor, Tensor> RandomLteTensors(absl::Span<const int64_t> shape) {
    int64_t num_vals = ShapeNumVals(shape);
    absl::FixedArray<T> less(num_vals);
    RandomVals({}, {}, false, less);
    absl::FixedArray<T> greater(num_vals);
    RandomVals({}, {}, false, greater);
    for (int i = 0; i < num_vals; ++i) {
      if (less[i] > greater[i]) {
        std::swap(less[i], greater[i]);
      }
    }
    std::pair<Tensor, Tensor> pair(
        TensorFromValues<T>(dtype(), shape, absl::Span<T>(less)),
        TensorFromValues<T>(dtype(), shape, absl::Span<T>(greater)));
    return pair;
  }

 protected:
  OpTest& test_;
};

class TensorGeneratorFloat : public TensorGenerator<float> {
 public:
  explicit TensorGeneratorFloat(OpTest& test) : TensorGenerator(test) {}
  DataType dtype() override { return DT_FLOAT; }
  void RandomVals(absl::optional<float> lo, absl::optional<float> hi,
                  bool needs_unique_values,
                  absl::FixedArray<float>& vals) override {
    absl::flat_hash_set<float> already_generated;
    std::uniform_real_distribution<float> distribution(lo.value_or(-1.0f),
                                                       hi.value_or(1.0f));
    for (int64_t i = 0; i < vals.size(); ++i) {
      float generated;
      do {
        generated = distribution(test_.generator());
      } while (needs_unique_values &&
               !already_generated.insert(generated).second);
      vals[i] = (generated);
    }
  }
};

class TensorGeneratorDouble : public TensorGenerator<double> {
 public:
  explicit TensorGeneratorDouble(OpTest& test) : TensorGenerator(test) {}
  DataType dtype() override { return DT_DOUBLE; }
  void RandomVals(absl::optional<double> lo, absl::optional<double> hi,
                  bool needs_unique_values,
                  absl::FixedArray<double>& vals) override {
    absl::flat_hash_set<double> already_generated;
    std::uniform_real_distribution<double> distribution(lo.value_or(-1.0),
                                                        hi.value_or(1.0));
    for (int64_t i = 0; i < vals.size(); ++i) {
      double generated;
      do {
        generated = distribution(test_.generator());
      } while (needs_unique_values &&
               !already_generated.insert(generated).second);
      vals[i] = generated;
    }
  }
};

class TensorGeneratorComplex64 : public TensorGenerator<complex64> {
 public:
  explicit TensorGeneratorComplex64(OpTest& test) : TensorGenerator(test) {}
  DataType dtype() override { return DT_COMPLEX64; }
  void RandomVals(absl::optional<complex64> lo, absl::optional<complex64> hi,
                  bool needs_unique_values,
                  absl::FixedArray<complex64>& vals) override {
    absl::flat_hash_set<std::pair<float, float>> already_generated;
    if (lo || hi) {
      LOG(FATAL) << "Lower or upper bounds are not supported for complex64.";
    }
    std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
    for (int64_t i = 0; i < vals.size(); ++i) {
      complex64 generated;
      do {
        generated = complex64(distribution(test_.generator()),
                              distribution(test_.generator()));
      } while (needs_unique_values &&
               !already_generated
                    .insert(std::make_pair(generated.real(), generated.imag()))
                    .second);
      vals[i] = generated;
    }
  }
};

class TensorGeneratorInt32 : public TensorGenerator<int32> {
 public:
  explicit TensorGeneratorInt32(OpTest& test) : TensorGenerator(test) {}
  DataType dtype() override { return DT_INT32; }
  void RandomVals(absl::optional<int32> lo, absl::optional<int32> hi,
                  bool needs_unique_values,
                  absl::FixedArray<int32>& vals) override {
    absl::flat_hash_set<int32> already_generated;
    std::uniform_int_distribution<int32> distribution(lo.value_or(-(1 << 20)),
                                                      hi.value_or(1 << 20));
    for (int64_t i = 0; i < vals.size(); ++i) {
      int32_t generated;
      do {
        generated = distribution(test_.generator());
      } while (needs_unique_values &&
               !already_generated.insert(generated).second);
      vals[i] = generated;
    }
  }
};

class TensorGeneratorInt64 : public TensorGenerator<int64> {
 public:
  explicit TensorGeneratorInt64(OpTest& test) : TensorGenerator(test) {}
  DataType dtype() override { return DT_INT64; }
  void RandomVals(absl::optional<int64> lo, absl::optional<int64> hi,
                  bool needs_unique_values,
                  absl::FixedArray<int64>& vals) override {
    absl::flat_hash_set<int64_t> already_generated;
    std::uniform_int_distribution<int64_t> distribution(
        lo.value_or(-(1LL << 40)), hi.value_or(1LL << 40));
    for (int64_t i = 0; i < vals.size(); ++i) {
      int64_t generated;
      do {
        generated = distribution(test_.generator());
      } while (needs_unique_values &&
               !already_generated.insert(generated).second);
      vals[i] = generated;
    }
  }
};

class TensorGeneratorBool : public TensorGenerator<bool> {
 public:
  explicit TensorGeneratorBool(OpTest& test) : TensorGenerator(test) {}
  DataType dtype() override { return DT_BOOL; }
  void RandomVals(absl::optional<bool> lo, absl::optional<bool> hi,
                  bool needs_unique_values,
                  absl::FixedArray<bool>& vals) override {
    absl::flat_hash_set<bool> already_generated;
    if (lo || hi) {
      LOG(FATAL) << "Lower or upper bounds are not supported for bool.";
    }
    std::bernoulli_distribution distribution;
    for (int64_t i = 0; i < vals.size(); ++i) {
      bool generated;
      do {
        generated = distribution(test_.generator());
      } while (needs_unique_values &&
               !already_generated.insert(generated).second);
      vals[i] = generated;
    }
  }
};

void OpTest::Repeatedly(const std::function<TestResult(void)>& fn) {
  int const max_repetitions = tf_xla_test_repetitions;
  int valid_test_runs = 0;
  // We run up to 100 * max_repetitions times; the idea is that if we roll the
  // dice enough times we will find some valid parameters. We want to put an
  // upper limit on the number iterations just in case the probability of
  // finding feasible parameters is very low.
  for (int i = 0; !HasFailure() && i < max_repetitions * 100 &&
                  valid_test_runs < max_repetitions;
       ++i) {
    TestResult result = fn();
    switch (result) {
      case kOk:
        ++valid_test_runs;
        break;

      case kFatalError:
        ASSERT_TRUE(false) << "Test had fatal failure";
        return;

      case kInvalid:
        break;
    }
  }
  if (!HasFailure()) {
    EXPECT_GE(valid_test_runs, max_repetitions)
        << "Not enough test instances passed; this means that either the "
           "golden implementation is buggy or the operator harness is not "
           "producing well-formed test cases with a high probability.";
  }
}

template <typename T>
T OpTest::Choose(absl::Span<const T> candidates) {
  std::uniform_int_distribution<size_t> d(0, candidates.size() - 1);
  return candidates[d(generator())];
}

int64_t OpTest::RandomDim(int64_t min, int64_t max) {
  std::uniform_int_distribution<int64_t> size_distribution(min, max - 1);
  return size_distribution(generator());
}

bool OpTest::TensorSizeIsOk(absl::Span<const int64_t> dims) {
  int64_t size = 1LL;
  for (int64_t dim : dims) {
    size *= dim;
  }
  return size < tf_xla_max_tensor_size;
}

std::vector<int64_t> OpTest::RandomDims(int min_rank, int max_rank,
                                        int64_t min_size, int64_t max_size) {
  CHECK_LE(0, min_rank);
  CHECK_LE(min_rank, max_rank);
  std::uniform_int_distribution<int> rank_distribution(min_rank, max_rank);
  int rank = rank_distribution(generator());
  std::vector<int64_t> dims(rank);
  if (rank == 0) {
    return dims;
  }
  int64_t per_dim_limit = std::pow(tf_xla_max_tensor_size, 1.0 / rank);
  int64_t per_dim_max = std::min(max_size, per_dim_limit);
  std::generate(dims.begin(), dims.end(), [this, min_size, per_dim_max]() {
    return RandomDim(min_size, per_dim_max);
  });
  CHECK(TensorSizeIsOk(dims));  // Crash OK
  return dims;
}

bool OpTest::RandomBool() {
  std::bernoulli_distribution d(0.5);
  return d(generator());
}

int64_t OpTest::RandomSeed() {
  std::uniform_int_distribution<int64_t> seed_dist(
      std::numeric_limits<int64_t>::min(), std::numeric_limits<int64_t>::max());
  int64_t seed = seed_dist(generator());
  if (seed == 0) return 1;
  return seed;
}

Tensor OpTest::RandomTensor(DataType dtype, bool needs_unique_values,
                            absl::Span<const int64_t> shape) {
  switch (dtype) {
    case DT_FLOAT:
      return TensorGeneratorFloat(*this).RandomTensor(
          {}, {}, needs_unique_values, shape);
    case DT_DOUBLE:
      return TensorGeneratorDouble(*this).RandomTensor(
          {}, {}, needs_unique_values, shape);
    case DT_COMPLEX64:
      return TensorGeneratorComplex64(*this).RandomTensor(
          {}, {}, needs_unique_values, shape);
    case DT_INT32:
      return TensorGeneratorInt32(*this).RandomTensor(
          {}, {}, needs_unique_values, shape);
    case DT_INT64:
      return TensorGeneratorInt64(*this).RandomTensor(
          {}, {}, needs_unique_values, shape);
    case DT_BOOL:
      return TensorGeneratorBool(*this).RandomTensor(
          {}, {}, needs_unique_values, shape);
    default:
      LOG(FATAL) << "Unimplemented type " << dtype << " in RandomTensor";
  }
}

Tensor OpTest::RandomTensor(DataType dtype) {
  return RandomTensor(dtype, /*needs_unique_values=*/false, RandomDims());
}

Tensor OpTest::RandomNonNegativeTensor(DataType dtype,
                                       absl::Span<const int64_t> shape) {
  switch (dtype) {
    case DT_FLOAT:
      return TensorGeneratorFloat(*this).RandomTensor({0.0f}, {}, false, shape);
    case DT_DOUBLE:
      return TensorGeneratorDouble(*this).RandomTensor({0.0}, {}, false, shape);
    case DT_INT32:
      return TensorGeneratorInt32(*this).RandomTensor({0}, {}, false, shape);
    case DT_INT64:
      return TensorGeneratorInt64(*this).RandomTensor({0}, {}, false, shape);
    default:
      LOG(FATAL) << "Unimplemented type " << dtype
                 << " in RandomNonNegativeTensor";
  }
}

Tensor OpTest::RandomNonNegativeTensor(DataType dtype) {
  return RandomNonNegativeTensor(dtype, RandomDims());
}

template <typename T>
Tensor OpTest::RandomBoundedTensor(DataType dtype, T lo, T hi,
                                   bool needs_unique_values,
                                   absl::Span<const int64_t> shape) {
  switch (dtype) {
    case DT_FLOAT:
      return TensorGeneratorFloat(*this).RandomTensor(
          {lo}, {hi}, needs_unique_values, shape);
    case DT_DOUBLE:
      return TensorGeneratorDouble(*this).RandomTensor(
          {lo}, {hi}, needs_unique_values, shape);
    case DT_INT32:
      return TensorGeneratorInt32(*this).RandomTensor(
          {lo}, {hi}, needs_unique_values, shape);
    case DT_INT64:
      return TensorGeneratorInt64(*this).RandomTensor(
          {lo}, {hi}, needs_unique_values, shape);
    default:
      LOG(FATAL) << "RandomBoundedTensor does not support type " << dtype
                 << ".";
  }
}

template <typename T>
Tensor OpTest::RandomBoundedTensor(DataType dtype, T lo, T hi,
                                   bool needs_unique_values) {
  return RandomBoundedTensor<T>(dtype, lo, hi, needs_unique_values,
                                RandomDims());
}

Tensor OpTest::RandomBoundedTensor(DataType dtype, Tensor lo, Tensor hi) {
  TensorShape shape = lo.shape();
  if (hi.shape() != shape) {
    LOG(FATAL) << "hi and lo do not have the same shape in RandomBoundedTensor";
  }
  if (hi.dtype() != dtype) {
    LOG(FATAL) << "hi does not have the expected dtype in RandomBoundedTensor";
  }
  if (lo.dtype() != dtype) {
    LOG(FATAL) << "lo does not have the expected dtype in RandomBoundedTensor";
  }
  Tensor tensor(dtype, shape);
  switch (dtype) {
    case DT_FLOAT: {
      auto lo_flat = lo.flat<float>();
      auto hi_flat = hi.flat<float>();
      test::FillFn<float>(&tensor, [this, &lo_flat, &hi_flat](int i) -> float {
        std::uniform_real_distribution<float> distribution(lo_flat(i),
                                                           hi_flat(i));
        return distribution(generator());
      });
      break;
    }
    case DT_DOUBLE: {
      auto lo_flat = lo.flat<double>();
      auto hi_flat = hi.flat<double>();
      test::FillFn<double>(
          &tensor, [this, &lo_flat, &hi_flat](int i) -> double {
            std::uniform_real_distribution<double> distribution(lo_flat(i),
                                                                hi_flat(i));
            return distribution(generator());
          });
      break;
    }
    case DT_INT32: {
      auto lo_flat = lo.flat<int32>();
      auto hi_flat = hi.flat<int32>();
      test::FillFn<int32>(&tensor, [this, &lo_flat, &hi_flat](int i) -> int32 {
        std::uniform_int_distribution<int32> distribution(lo_flat(i),
                                                          hi_flat(i));
        return distribution(generator());
      });
      break;
    }
    case DT_INT64: {
      auto lo_flat = lo.flat<int64>();
      auto hi_flat = hi.flat<int64>();
      test::FillFn<int64_t>(
          &tensor, [this, &lo_flat, &hi_flat](int i) -> int64_t {
            std::uniform_int_distribution<int64_t> distribution(lo_flat(i),
                                                                hi_flat(i));
            return distribution(generator());
          });
      break;
    }
    default:
      LOG(FATAL) << "RandomBoundedTensor does not support type " << dtype
                 << ".";
  }
  return tensor;
}

std::pair<Tensor, Tensor> OpTest::RandomLteTensors(
    DataType dtype, absl::Span<const int64_t> shape) {
  switch (dtype) {
    case DT_FLOAT:
      return TensorGeneratorFloat(*this).RandomLteTensors(shape);
    case DT_DOUBLE:
      return TensorGeneratorDouble(*this).RandomLteTensors(shape);
    case DT_COMPLEX64:
      LOG(FATAL) << "RandomLteTensors unavailable for DT_COMPLEX64";
      break;
    case DT_INT32:
      return TensorGeneratorInt32(*this).RandomLteTensors(shape);
    case DT_INT64:
      return TensorGeneratorInt64(*this).RandomLteTensors(shape);
    case DT_BOOL:
      LOG(FATAL) << "RandomLteTensors unavailable for DT_BOOL";
      break;
    default:
      LOG(FATAL) << "Unimplemented type " << dtype << " in RandomLteTensors";
  }
  Tensor tensor(dtype, TensorShape(shape));
  return std::pair<Tensor, Tensor>(tensor, tensor);
}

std::pair<Tensor, Tensor> OpTest::RandomLteTensors(DataType dtype) {
  return RandomLteTensors(dtype, RandomDims());
}

std::vector<int64_t> OpTest::BroadcastableToDims(std::vector<int64_t> dims) {
  if (dims.empty()) return dims;

  // Remove some dimensions from the front of 'dims'.
  size_t skip =
      std::uniform_int_distribution<size_t>(0, dims.size() - 1)(generator());

  std::vector<int64_t> bdims(dims.begin() + skip, dims.end());

  // Randomly replace some of the remaining dimensions of 'dims' with 1.
  std::bernoulli_distribution random_bool;

  for (int64_t& dim : bdims) {
    if (random_bool(generator())) {
      dim = 1LL;
    }
  }
  return bdims;
}

std::pair<std::vector<int64_t>, std::vector<int64_t>> OpTest::BroadcastableDims(
    std::vector<int64_t> dims) {
  auto bdims = BroadcastableToDims(dims);
  // Possibly swap the roles of 'dims' and 'bdims'.
  std::bernoulli_distribution random_bool;
  if (random_bool(generator())) {
    dims.swap(bdims);
  }
  return {dims, bdims};
}

std::pair<std::vector<int64_t>, std::vector<int64_t>>
OpTest::BroadcastableDims() {
  return BroadcastableDims(RandomDims(0, 3));
}

Tensor OpTest::RandomReductionIndices(int rank) {
  std::bernoulli_distribution random_bool;
  std::vector<int32> indices;
  for (int i = 0; i < rank; ++i) {
    if (random_bool(generator())) {
      indices.push_back(i);
    }
  }
  return test::AsTensor<int32>(indices);
}

// Helper that converts 'values' to an int32 or int64 Tensor.
static Tensor AsIntTensor(DataType dtype, const std::vector<int64_t>& values) {
  switch (dtype) {
    case DT_INT32: {
      std::vector<int32> values32(values.begin(), values.end());
      return test::AsTensor<int32>(values32);
    }
    case DT_INT64:
      return test::AsTensor<int64_t>(values);
    default:
      LOG(FATAL);
  }
}

OpTest::BatchMatMulArguments OpTest::ChooseBatchMatMulArguments(
    bool broadcastable_batch) {
  BatchMatMulArguments a;
  a.dtype = Choose<DataType>({DT_FLOAT, DT_COMPLEX64});

  int64_t min_size = 0;
  int64_t max_size = 7;
  auto batch_dims_to = RandomDims(0, 3, min_size, max_size);
  int rank = batch_dims_to.size() + 2;
  std::pair<std::vector<int64_t>, std::vector<int64_t>> batch_dims_nobcast(
      batch_dims_to, batch_dims_to);
  auto batch_dims = broadcastable_batch ? BroadcastableDims(batch_dims_to)
                                        : batch_dims_nobcast;
  std::vector<int64_t> lhs_dims(batch_dims.first), rhs_dims(batch_dims.second);
  int64_t inner_dim = RandomDim();
  lhs_dims.push_back(RandomDim(min_size, max_size));
  lhs_dims.push_back(inner_dim);
  rhs_dims.push_back(inner_dim);
  rhs_dims.push_back(RandomDim(min_size, max_size));

  std::bernoulli_distribution random_bool;
  a.adj_lhs = random_bool(generator());
  a.adj_rhs = random_bool(generator());
  if (a.adj_lhs) {
    std::swap(lhs_dims[rank - 1], lhs_dims[rank - 2]);
  }
  if (a.adj_rhs) {
    std::swap(rhs_dims[rank - 1], rhs_dims[rank - 2]);
  }

  a.lhs_dims = lhs_dims;
  a.rhs_dims = rhs_dims;
  return a;
}

OpTest::ConcatArguments OpTest::ChooseConcatArguments(bool int64_idx_allowed) {
  ConcatArguments a;

  std::bernoulli_distribution random_bool;
  bool use_int64_idx = random_bool(generator());

  a.type = Choose<DataType>(kAllXlaTypes);
  a.type_idx = use_int64_idx ? DT_INT64 : DT_INT32;
  a.n = std::uniform_int_distribution<int>(2, 4)(generator());

  std::vector<int64_t> dims = RandomDims(1, 4, 0, 64);

  int axis =
      std::uniform_int_distribution<int32>(0, dims.size() - 1)(generator());
  a.axis =
      use_int64_idx ? test::AsScalar<int64>(axis) : test::AsScalar<int32>(axis);

  for (int i = 0; i < a.n; ++i) {
    std::vector<int64_t> shape = dims;
    shape[axis] = RandomDim(0, 64);
    a.values.push_back(RandomTensor(a.type, false, shape));
  }

  return a;
}

OpTest::EinsumArguments OpTest::ChooseEinsumArguments() {
  EinsumArguments a;

  enum EinsumType { matmul, batchmatmul, dot, outer };
  int op_kind = Choose<int>({matmul, batchmatmul, dot, outer});
  switch (op_kind) {
    case matmul:
    case batchmatmul: {
      std::vector<int64> dims;
      if (op_kind == matmul) {
        a.equation = "ij,jk->ik";
        dims = RandomDims(2, 2);
      } else {
        a.equation = "...ij,...jk->...ik";
        dims = RandomDims(2);
      }
      int64_t ndims = dims.size();
      int64_t inner_dim = RandomDim();
      a.lhs_dims = dims;
      a.rhs_dims = dims;
      a.lhs_dims[ndims - 1] = inner_dim;
      a.rhs_dims[ndims - 2] = inner_dim;
      break;
    }
    case dot: {
      a.equation = "i,i->";
      std::vector<int64> dims = RandomDims(1, 1);
      a.lhs_dims = dims;
      a.rhs_dims = dims;
      break;
    }
    case outer: {
      a.equation = "i,j->ij";
      a.lhs_dims = RandomDims(1, 1);
      a.rhs_dims = RandomDims(1, 1);
      break;
    }
  }

  a.type = Choose<DataType>(kAllXlaTypes);
  return a;
}

OpTest::GatherArguments OpTest::ChooseGatherArguments(bool axis_0) {
  GatherArguments a;

  a.axis_type = DT_INT32;
  a.indices_type = DT_INT32;
  a.params_type = Choose<DataType>(kAllXlaTypes);

  // Choose parameters such that
  // 0 <= batch_dims <= axis < params.rank <= kDefaultMaxRank
  a.batch_dims = 0;
  int64_t axis;
  if (axis_0) {
    axis = 0;
  } else {
    std::uniform_int_distribution<int64_t> axis_distribution(
        a.batch_dims, kDefaultMaxRank - 1);
    axis = axis_distribution(generator());
  }
  a.axis = test::AsScalar<int32>((int32)axis);
  a.params_shape = RandomDims(axis + 1, kDefaultMaxRank, 1, 16);
  std::vector<int64_t> indices_shape = RandomDims(0, 3, 0, 16);
  a.indices = RandomBoundedTensor<int32>(DT_INT32, 0, a.params_shape[axis] - 1,
                                         false, indices_shape);

  return a;
}

OpTest::PadArguments OpTest::ChoosePadArguments() {
  PadArguments a;

  a.input_type = Choose<DataType>(kAllXlaTypes);
  a.input_shape = RandomDims();
  int input_rank = a.input_shape.size();

  a.paddings_type = Choose<DataType>({DT_INT32, DT_INT64});
  std::vector<int64_t> paddings_vec;
  for (int i = 0; i < input_rank; ++i) {
    std::uniform_int_distribution<int> pad_distribution(0, a.input_shape[i]);
    int pad_size = pad_distribution(generator());
    std::uniform_int_distribution<int> lower_distribution(0, pad_size);
    int low_pad_size = lower_distribution(generator());
    paddings_vec.push_back(low_pad_size);
    paddings_vec.push_back(pad_size - low_pad_size);
    a.input_shape[i] -= pad_size;
  }
  CHECK(
      a.paddings.CopyFrom(AsIntTensor(a.paddings_type, paddings_vec),
                          TensorShape({static_cast<int64_t>(input_rank), 2})));

  a.constant_values = RandomTensor(a.input_type, false, {});

  return a;
}

OpTest::ScatterArguments OpTest::ChooseScatterArguments() {
  ScatterArguments a;

  a.type = Choose<DataType>(kAllXlaTypes);
  a.indices_type = DT_INT32;
  a.shape = RandomDims(1, kDefaultMaxRank, 1);
  int rank = a.shape.size();
  std::uniform_int_distribution<int32> index_len_dist(1, rank);
  int index_len = index_len_dist(generator());
  std::vector<int64_t> indices_first = RandomDims(1, kDefaultMaxRank - 1, 1);
  std::vector<int64_t> indices_shape(indices_first);
  indices_shape.push_back(index_len);
  std::vector<int64_t> updates_shape(indices_first);
  for (int i = 0; i < rank - index_len; ++i) {
    updates_shape.push_back(a.shape[index_len + i]);
  }
  Tensor indices_lo(a.indices_type, TensorShape(indices_shape));
  test::FillFn<int32>(&indices_lo, [](int i) -> int32 { return 0; });
  Tensor indices_hi(a.indices_type, TensorShape(indices_shape));
  test::FillFn<int32>(&indices_hi, [index_len, &a](int i) -> int32 {
    int idx_dim = i % index_len;
    return a.shape[idx_dim] - 1;
  });
  a.indices = RandomBoundedTensor(a.indices_type, indices_lo, indices_hi);
  a.updates = RandomTensor(a.type, false, updates_shape);

  return a;
}

OpTest::SliceArguments OpTest::ChooseSliceArguments(bool neg_one_size) {
  SliceArguments a;

  a.type = Choose<DataType>(kAllXlaTypes);
  a.indices_type = DT_INT32;
  a.shape = RandomDims();
  int rank = a.shape.size();

  std::vector<int32> indices(rank);
  a.size.resize(rank);
  for (int i = 0; i < rank; ++i) {
    indices[i] =
        std::uniform_int_distribution<int32>(0, a.shape[i])(generator());
    int64_t low = neg_one_size ? -1 : 0;
    a.size[i] = std::uniform_int_distribution<int64_t>(
        low, a.shape[i] - indices[i])(generator());
  }
  a.indices = test::AsTensor<int32>(indices);

  return a;
}

OpTest::WindowedSpatialDims OpTest::ChooseWindowedSpatialDims(
    int num_spatial_dims) {
  WindowedSpatialDims d;
  d.padding = Choose<Padding>({SAME, VALID});
  std::uniform_int_distribution<int> random_int(1, 5);
  d.kernel_dims.resize(num_spatial_dims);
  d.input_dims.resize(num_spatial_dims);
  d.output_dims.resize(num_spatial_dims);
  d.stride_dims.resize(num_spatial_dims);
  for (int i = 0; i < num_spatial_dims; ++i) {
    Status s;
    // Repeatedly try different filter/stride sizes until we find a valid
    // combination.
    do {
      // CPU implementations require stride <= kernel size.
      d.kernel_dims[i] = random_int(generator()),
      d.input_dims[i] = RandomDim(d.kernel_dims[i]);
      d.stride_dims[i] =
          std::uniform_int_distribution<int>(1, d.kernel_dims[i])(generator());
      int64_t pad_dummy;
      s = GetWindowedOutputSize(d.input_dims[i], d.kernel_dims[i],
                                d.stride_dims[i], d.padding, &d.output_dims[i],
                                &pad_dummy);
    } while (!s.ok());
  }
  return d;
}

OpTest::XlaDotArguments OpTest::ChooseXlaDotArguments() {
  std::vector<int64_t> batch_dims = RandomDims(0, 2);
  std::vector<int64_t> contracting_dims = RandomDims(0, 2);
  std::vector<int64_t> lhs_outer_dims = RandomDims(0, 2);
  std::vector<int64_t> rhs_outer_dims = RandomDims(0, 2);

  XlaDotArguments a;
  a.lhs_dims.insert(a.lhs_dims.end(), batch_dims.begin(), batch_dims.end());
  a.lhs_dims.insert(a.lhs_dims.end(), contracting_dims.begin(),
                    contracting_dims.end());
  a.lhs_dims.insert(a.lhs_dims.end(), lhs_outer_dims.begin(),
                    lhs_outer_dims.end());
  a.rhs_dims.insert(a.rhs_dims.end(), batch_dims.begin(), batch_dims.end());
  a.rhs_dims.insert(a.rhs_dims.end(), contracting_dims.begin(),
                    contracting_dims.end());
  a.rhs_dims.insert(a.rhs_dims.end(), rhs_outer_dims.begin(),
                    rhs_outer_dims.end());

  xla::DotDimensionNumbers dnums;
  for (auto i = 0; i < batch_dims.size(); ++i) {
    dnums.add_lhs_batch_dimensions(i);
    dnums.add_rhs_batch_dimensions(i);
  }
  for (auto i = 0; i < contracting_dims.size(); ++i) {
    dnums.add_lhs_contracting_dimensions(batch_dims.size() + i);
    dnums.add_rhs_contracting_dimensions(batch_dims.size() + i);
  }
  dnums.SerializeToString(&a.dnums_encoded);

  a.precision_config_encoded = "";

  a.dtype = Choose<DataType>(kAllXlaTypes);
  return a;
}

std::vector<int64_t> OpTest::ImageDims(
    TensorFormat format, int batch, int feature,
    const std::vector<int64_t>& spatial_dims) {
  std::vector<int64_t> dims;
  switch (format) {
    case FORMAT_NHWC:
      dims.push_back(batch);
      for (int dim : spatial_dims) {
        dims.push_back(dim);
      }
      dims.push_back(feature);
      break;
    case FORMAT_NCHW:
      dims.push_back(batch);
      dims.push_back(feature);
      for (int dim : spatial_dims) {
        dims.push_back(dim);
      }
      break;
    default:
      LOG(FATAL) << "Tensor format " << ToString(format) << " not supported.";
  }
  return dims;
}

std::vector<int32> OpTest::AsInt32s(const std::vector<int64_t>& int64s) {
  return std::vector<int32>(int64s.begin(), int64s.end());
}

// Functions for comparing tensors.

template <typename T>
double Abs(T x) {
  return std::fabs(x);
}

template <>
double Abs<complex64>(complex64 x) {
  return std::abs(x);
}

template <typename T>
bool IsClose(const T& x, const T& y, double atol, double rtol) {
  if (std::isnan(x) && std::isnan(y)) return true;
  if (x == y) return true;  // Allow inf == inf.
  return Abs(x - y) < atol + rtol * Abs(x);
}

template <>
bool IsClose<complex64>(const complex64& x, const complex64& y, double atol,
                        double rtol) {
  if (std::isnan(x.real()) && std::isnan(y.real())) {
    if (std::isnan(x.imag()) && std::isnan(y.imag())) {
      return true;
    }
    if (x.imag() == y.imag()) return true;  // Allow inf == inf.
    return Abs(x.imag() - y.imag()) < atol + rtol * Abs(x.imag());
  } else if (std::isnan(x.imag()) && std::isnan(y.imag())) {
    if (x.real() == y.real()) return true;  // Allow inf == inf.
    return Abs(x.real() - y.real()) < atol + rtol * Abs(x.real());
  }
  if (x == y) return true;  // Allow inf == inf.
  return Abs(x - y) < atol + rtol * Abs(x);
}

template <typename T>
string Str(T x) {
  return absl::StrCat(x);
}
template <>
string Str<complex64>(complex64 x) {
  return absl::StrCat("(", x.real(), ", ", x.imag(), ")");
}

template <typename T>
Status TensorsAreCloseImpl(const Tensor& x, const Tensor& y, double atol,
                           double rtol) {
  auto Tx = x.flat<T>();
  auto Ty = y.flat<T>();
  for (int i = 0; i < Tx.size(); ++i) {
    if (!IsClose(Tx(i), Ty(i), atol, rtol)) {
      return errors::InvalidArgument(
          absl::StrCat(i, "-th tensor element isn't close: ", Str(Tx(i)),
                       " vs. ", Str(Ty(i)), ". x = ", x.DebugString(),
                       "y = ", y.DebugString(), "atol = ", atol,
                       " rtol = ", rtol, " tol = ", atol + rtol * Abs(Tx(i))));
    }
  }
  return OkStatus();
}

template <typename T>
Status TensorsAreEqualImpl(const Tensor& x, const Tensor& y) {
  auto Tx = x.flat<T>();
  auto Ty = y.flat<T>();
  for (int i = 0; i < Tx.size(); ++i) {
    if (Tx(i) != Ty(i)) {
      return errors::InvalidArgument(absl::StrCat(
          i, "-th tensor element isn't equal: ", Str(Tx(i)), " vs. ",
          Str(Ty(i)), ". x = ", x.DebugString(), "y = ", y.DebugString()));
    }
  }
  return OkStatus();
}

Status TensorsAreEqualImplBfloat16(const Tensor& x, const Tensor& y) {
  auto Tx = x.flat<bfloat16>();
  auto Ty = y.flat<bfloat16>();
  for (int i = 0; i < Tx.size(); ++i) {
    if (Tx(i) != Ty(i)) {
      return errors::InvalidArgument(absl::StrCat(
          i, "-th tensor element isn't equal: ", static_cast<float>(Tx(i)),
          " vs. ", static_cast<float>(Ty(i)), ". x = ", x.DebugString(),
          "y = ", y.DebugString()));
    }
  }
  return OkStatus();
}

// Tests if "x" and "y" are tensors of the same type, same shape, and with
// close values. For floating-point tensors, the element-wise difference between
// x and y must no more than atol + rtol * abs(x). For non-floating-point
// tensors the values must match exactly.
Status TensorsAreClose(const Tensor& a, const Tensor& b, double atol,
                       double rtol) {
  if (a.dtype() != b.dtype()) {
    return errors::InvalidArgument(absl::StrCat(
        "Tensors have different types: ", DataTypeString(a.dtype()), " and ",
        DataTypeString(b.dtype())));
  }
  if (!a.IsSameSize(b)) {
    return errors::InvalidArgument(
        absl::StrCat("Tensors have different shapes: ", a.shape().DebugString(),
                     " and ", b.shape().DebugString()));
  }

  switch (a.dtype()) {
    case DT_FLOAT:
      return TensorsAreCloseImpl<float>(a, b, atol, rtol);
    case DT_DOUBLE:
      return TensorsAreCloseImpl<double>(a, b, atol, rtol);
    case DT_COMPLEX64:
      return TensorsAreCloseImpl<complex64>(a, b, atol, rtol);
    case DT_INT32:
      return TensorsAreEqualImpl<int32>(a, b);
    case DT_INT64:
      return TensorsAreEqualImpl<int64_t>(a, b);
    case DT_BOOL:
      return TensorsAreEqualImpl<bool>(a, b);
    case DT_BFLOAT16:
      return TensorsAreEqualImplBfloat16(a, b);
    default:
      LOG(FATAL) << "Unexpected type : " << DataTypeString(a.dtype());
  }
}

OpTest::TestResult OpTest::ExpectTfAndXlaOutputsAreClose(
    const OpTestBuilder& builder, double atol, double rtol) {
  const std::vector<OpTestBuilder::InputDescription>& inputs = builder.inputs();
  std::vector<Tensor> input_tensors;
  input_tensors.reserve(inputs.size());
  for (const OpTestBuilder::InputDescription& input : inputs) {
    if (input.type == DT_INVALID) {
      input_tensors.push_back(input.tensor);
    } else {
      std::vector<int64_t> dims;
      if (input.has_dims) {
        dims = input.dims;
      } else {
        dims = RandomDims();
      }
      if (!TensorSizeIsOk(dims)) {
        VLOG(1) << "Input: " << input.type << " "
                << TensorShape(input.dims).DebugString();
        VLOG(1) << "Ignoring oversize dims.";
        return kInvalid;
      }
      input_tensors.push_back(
          RandomTensor(input.type, input.needs_unique_values, dims));
    }
    VLOG(1) << "Input: " << input_tensors.back().DebugString();
  }

  string reference_device =
      LocalDeviceToFullDeviceName(*tf_xla_reference_device_ptr);
  string test_device = LocalDeviceToFullDeviceName(*tf_xla_test_device_ptr);

  DeviceNameUtils::ParsedName parsed_name;
  if (!DeviceNameUtils::ParseLocalName(*tf_xla_test_device_ptr, &parsed_name)) {
    LOG(ERROR) << "Could not parse device name: " << *tf_xla_test_device_ptr;
    return kFatalError;
  }
  DeviceType test_device_type(parsed_name.type);
  ++num_tests_;

  GraphDef graph;
  std::vector<string> expected_inputs, test_inputs;
  std::vector<string> expected_fetches, test_fetches;
  Status status = builder.BuildGraph(
      absl::StrCat("test", num_tests_, "_expected"), reference_device,
      /*use_jit=*/false, &graph, /*test_node_def=*/nullptr, &expected_inputs,
      &expected_fetches);
  if (!status.ok()) {
    LOG(ERROR) << "Expected graph construction failed: " << status;
    return kFatalError;
  }

  NodeDef* node_def;
  status = builder.BuildGraph(absl::StrCat("test", num_tests_, "_test"),
                              test_device, tf_xla_test_use_jit, &graph,
                              &node_def, &test_inputs, &test_fetches);
  if (!status.ok()) {
    LOG(ERROR) << "Test graph construction failed: " << status;
    return kFatalError;
  }

  // Check that there's a kernel corresponding to 'node_def' on the device under
  // test.
  status = FindKernelDef(test_device_type, *node_def, nullptr, nullptr);
  if (!status.ok()) {
    VLOG(1) << "Skipping test because there is no corresponding registered "
            << "kernel on the test device: " << status;
    return kInvalid;
  }

  status = session_->Extend(graph);
  if (!status.ok()) {
    LOG(ERROR) << "Session::Extend() failed: " << status;
    return kFatalError;
  }

  std::vector<std::pair<string, Tensor>> expected_feeds(expected_inputs.size());
  std::vector<std::pair<string, Tensor>> test_feeds(test_inputs.size());
  CHECK_EQ(input_tensors.size(), expected_inputs.size());
  CHECK_EQ(input_tensors.size(), test_inputs.size());

  for (int i = 0; i < input_tensors.size(); ++i) {
    expected_feeds[i] = {expected_inputs[i], input_tensors[i]};
    test_feeds[i] = {test_inputs[i], input_tensors[i]};
  }

  std::vector<Tensor> expected_outputs, test_outputs;
  VLOG(1) << "Running expected graph";
  Status s =
      session_->Run(expected_feeds, expected_fetches, {}, &expected_outputs);
  if (!s.ok()) {
    VLOG(1) << "Expected graph failed with status: " << s << ". Ignoring test";
    return kInvalid;
  }
  for (const Tensor& expected : expected_outputs) {
    VLOG(1) << "Expected: " << expected.DebugString();
  }

  VLOG(1) << "Running test graph";
  status = session_->Run(test_feeds, test_fetches, {}, &test_outputs);
  if (!status.ok()) {
    LOG(ERROR) << "Test graph failed: " << status;
    return kFatalError;
  }

  CHECK_EQ(expected_outputs.size(), test_outputs.size());
  for (int j = 0; s.ok() && j < test_outputs.size(); ++j) {
    s = TensorsAreClose(expected_outputs[j], test_outputs[j], atol, rtol);
  }
  TF_EXPECT_OK(s);

  return kOk;
}

TEST_F(OpTest, _EagerConst) {
  Repeatedly([this]() {
    auto type = Choose<DataType>(kAllXlaTypes);
    return ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("_EagerConst").RandomInput(type).Attr("T", type));
  });
}

TEST_F(OpTest, Abs) {
  Repeatedly([this]() {
    auto type = Choose<DataType>({DT_INT32, DT_FLOAT, DT_COMPLEX64});
    return ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("Abs").RandomInput(type).Attr("T", type));
  });
}

TEST_F(OpTest, Acos) {
  Repeatedly([this]() {
    return ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("Acos")
            .Input(RandomBoundedTensor<float>(DT_FLOAT, -1, 1, false))
            .Attr("T", DT_FLOAT));
  });
}

TEST_F(OpTest, Acosh) {
  Repeatedly([this]() {
    return ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("Acosh").RandomInput(DT_FLOAT).Attr("T", DT_FLOAT));
  });
}

TEST_F(OpTest, Add) {
  Repeatedly([this]() {
    auto type = Choose<DataType>({DT_INT32, DT_FLOAT, DT_COMPLEX64});
    auto dims = BroadcastableDims();
    return ExpectTfAndXlaOutputsAreClose(OpTestBuilder("Add")
                                             .RandomInput(type, dims.first)
                                             .RandomInput(type, dims.second)
                                             .Attr("T", type));
  });
}

TEST_F(OpTest, AddN) {
  Repeatedly([this]() {
    auto type = Choose<DataType>({DT_INT32, DT_FLOAT, DT_COMPLEX64});
    int n = std::uniform_int_distribution<int>(1, 5)(generator());

    auto shape = RandomDims();

    OpTestBuilder builder("AddN");
    builder.Attr("T", type);
    builder.Attr("N", n);
    for (int i = 0; i < n; ++i) {
      builder.RandomInput(type, shape);
    }
    return ExpectTfAndXlaOutputsAreClose(builder);
  });
}

TEST_F(OpTest, AddV2) {
  Repeatedly([this]() {
    auto type = Choose<DataType>(kAllXlaTypes);
    auto dims = BroadcastableDims();
    return ExpectTfAndXlaOutputsAreClose(OpTestBuilder("AddV2")
                                             .RandomInput(type, dims.first)
                                             .RandomInput(type, dims.second)
                                             .Attr("T", type));
  });
}

TEST_F(OpTest, All) {
  Repeatedly([this]() {
    std::vector<int64_t> data_dims = RandomDims();
    Tensor indices = RandomReductionIndices(data_dims.size());
    bool keep_dims = Choose<bool>({false, true});
    return ExpectTfAndXlaOutputsAreClose(OpTestBuilder("All")
                                             .RandomInput(DT_BOOL, data_dims)
                                             .Input(indices)
                                             .Attr("keep_dims", keep_dims));
  });
}

TEST_F(OpTest, Angle) {
  Repeatedly([this]() {
    return ExpectTfAndXlaOutputsAreClose(OpTestBuilder("Angle")
                                             .RandomInput(DT_COMPLEX64)
                                             .Attr("T", DT_COMPLEX64));
  });
}

TEST_F(OpTest, Any) {
  Repeatedly([this]() {
    std::vector<int64_t> data_dims = RandomDims();
    Tensor indices = RandomReductionIndices(data_dims.size());
    bool keep_dims = Choose<bool>({false, true});
    return ExpectTfAndXlaOutputsAreClose(OpTestBuilder("Any")
                                             .RandomInput(DT_BOOL, data_dims)
                                             .Input(indices)
                                             .Attr("keep_dims", keep_dims));
  });
}

TEST_F(OpTest, ApproximateEqual) {
  Repeatedly([this]() {
    auto dims = BroadcastableDims();
    auto type = Choose<DataType>({DT_FLOAT, DT_COMPLEX64});
    return ExpectTfAndXlaOutputsAreClose(OpTestBuilder("ApproximateEqual")
                                             .RandomInput(type, dims.first)
                                             .RandomInput(type, dims.second)
                                             .Attr("T", DT_FLOAT));
  });
}

TEST_F(OpTest, ArgMax) {
  Repeatedly([this]() {
    auto type = Choose<DataType>({DT_BOOL, DT_FLOAT});
    std::vector<int64_t> dims = RandomDims(1, 5, 1);
    int num_dims = dims.size();
    int reduce_dim =
        std::uniform_int_distribution<int32>(-num_dims, num_dims)(generator());
    return ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("ArgMax")
            .RandomInput(type, dims)
            .Input(test::AsScalar<int32>(reduce_dim))
            .Attr("T", type)
            .Attr("Tidx", DT_INT32)
            .Attr("output_type", DT_INT32));
  });
}

TEST_F(OpTest, ArgMin) {
  Repeatedly([this]() {
    auto type = Choose<DataType>({DT_BOOL, DT_FLOAT});
    std::vector<int64_t> dims = RandomDims(1, 5, 1);
    int num_dims = dims.size();
    int reduce_dim =
        std::uniform_int_distribution<int32>(-num_dims, num_dims)(generator());
    return ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("ArgMin")
            .RandomInput(type, dims)
            .Input(test::AsScalar<int32>(reduce_dim))
            .Attr("T", type)
            .Attr("Tidx", DT_INT32)
            .Attr("output_type", DT_INT32));
  });
}

TEST_F(OpTest, Asin) {
  Repeatedly([this]() {
    return ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("Asin")
            .Input(RandomBoundedTensor<float>(DT_FLOAT, -1, 1, false))
            .Attr("T", DT_FLOAT));
  });
}

TEST_F(OpTest, Asinh) {
  Repeatedly([this]() {
    return ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("Asinh").RandomInput(DT_FLOAT).Attr("T", DT_FLOAT));
  });
}

TEST_F(OpTest, Atanh) {
  Repeatedly([this]() {
    return ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("Atanh").RandomInput(DT_FLOAT).Attr("T", DT_FLOAT));
  });
}

TEST_F(OpTest, Atan) {
  Repeatedly([this]() {
    return ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("Atan").RandomInput(DT_FLOAT).Attr("T", DT_FLOAT));
  });
}

TEST_F(OpTest, Atan2) {
  Repeatedly([this]() {
    auto dims = BroadcastableDims();
    return ExpectTfAndXlaOutputsAreClose(OpTestBuilder("Atan2")
                                             .RandomInput(DT_FLOAT, dims.first)
                                             .RandomInput(DT_FLOAT, dims.second)
                                             .Attr("T", DT_FLOAT));
  });
}

TEST_F(OpTest, AvgPool) {
  Repeatedly([this]() {
    std::uniform_int_distribution<int> random_int(1, 5);
    std::vector<int64_t> dims = RandomDims(4, 4, 1);
    int kernel_rows =
        std::uniform_int_distribution<int>(1, dims[1])(generator());
    int kernel_cols =
        std::uniform_int_distribution<int>(1, dims[2])(generator());
    int stride_rows = random_int(generator()),
        stride_cols = random_int(generator());
    string padding = Choose<string>({"SAME", "VALID"});
    return ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("AvgPool")
            .RandomInput(DT_FLOAT, dims)
            .Attr("T", DT_FLOAT)
            .Attr("ksize", {1, kernel_rows, kernel_cols, 1})
            .Attr("strides", {1, stride_rows, stride_cols, 1})
            .Attr("padding", padding)
            .Attr("data_format", "NHWC"));
  });
  // TODO(phawkins): the CPU device only implements spatial pooling. Add tests
  // for batch pooling when supported.
}

TEST_F(OpTest, AvgPool3D) {
  if (tensorflow::tf_xla_test_use_mlir) GTEST_SKIP() << "b/201095155";
  if (!tensorflow::tf_xla_test_use_mlir) GTEST_SKIP() << "b/197140886";
  Repeatedly([this]() {
    std::uniform_int_distribution<int> random_int(1, 5);
    std::vector<int64_t> dims = RandomDims(5, 5, 1);

    std::vector<int64_t> input_dims, kernel_dims, stride_dims;
    for (int i = 0; i < 3; ++i) {
      kernel_dims.push_back(
          std::uniform_int_distribution<int>(1, dims[i])(generator()));
      input_dims.push_back(dims[i]);
      stride_dims.push_back(random_int(generator()));
    }
    int64_t batch = dims[3];
    int64_t feature = dims[4];

    string padding = Choose<string>({"SAME", "VALID"});
    return ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("AvgPool3D")
            .RandomInput(DT_FLOAT,
                         ImageDims(FORMAT_NHWC, batch, feature, input_dims))
            .Attr("T", DT_FLOAT)
            .Attr("ksize", ImageDims(FORMAT_NHWC, 1, 1, kernel_dims))
            .Attr("strides", ImageDims(FORMAT_NHWC, 1, 1, stride_dims))
            .Attr("padding", padding)
            .Attr("data_format", "NDHWC"));
  });
  // TODO(phawkins): test NCHW format (not supported by CPU)
}

TEST_F(OpTest, AvgPoolGrad) {
  if (tensorflow::tf_xla_test_use_mlir) GTEST_SKIP() << "b/201095155";
  if (!tensorflow::tf_xla_test_use_mlir) GTEST_SKIP() << "b/197140886";
  Repeatedly([this]() {
    int batch = RandomDim(1), features = RandomDim(1);
    WindowedSpatialDims d = ChooseWindowedSpatialDims(2);
    std::vector<int32> input_dims =
        AsInt32s(ImageDims(FORMAT_NHWC, batch, features, d.input_dims));
    std::vector<int64_t> output_dims =
        ImageDims(FORMAT_NHWC, batch, features, d.output_dims);
    return ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("AvgPoolGrad")
            .Input(test::AsTensor<int32>(input_dims))
            .RandomInput(DT_FLOAT, output_dims)
            .Attr("T", DT_FLOAT)
            .Attr("ksize", ImageDims(FORMAT_NHWC, 1, 1, d.kernel_dims))
            .Attr("strides", ImageDims(FORMAT_NHWC, 1, 1, d.stride_dims))
            .Attr("padding", d.padding == SAME ? "SAME" : "VALID")
            .Attr("data_format", "NHWC"));
  });
}

TEST_F(OpTest, AvgPool3DGrad) {
  if (tensorflow::tf_xla_test_use_mlir) GTEST_SKIP() << "b/201095155";
  if (!tensorflow::tf_xla_test_use_mlir) GTEST_SKIP() << "b/197140886";
  Repeatedly([this]() {
    int batch = RandomDim(1), features = RandomDim(1);
    WindowedSpatialDims d = ChooseWindowedSpatialDims(3);
    std::vector<int32> input_dims =
        AsInt32s(ImageDims(FORMAT_NHWC, batch, features, d.input_dims));
    std::vector<int64_t> output_dims =
        ImageDims(FORMAT_NHWC, batch, features, d.output_dims);
    return ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("AvgPool3DGrad")
            .Input(test::AsTensor<int32>(input_dims))
            .RandomInput(DT_FLOAT, output_dims)
            .Attr("T", DT_FLOAT)
            .Attr("ksize", ImageDims(FORMAT_NHWC, 1, 1, d.kernel_dims))
            .Attr("strides", ImageDims(FORMAT_NHWC, 1, 1, d.stride_dims))
            .Attr("padding", d.padding == SAME ? "SAME" : "VALID")
            .Attr("data_format", "NDHWC"));
  });
}

TEST_F(OpTest, BatchMatMul) {
  // See note about failing Kokoro tests: b/214080339#comment22
  if (tensorflow::tf_xla_test_use_mlir) GTEST_SKIP() << "b/201095155";
  if (!tensorflow::tf_xla_test_use_mlir) GTEST_SKIP() << "b/197140886";
  Repeatedly([this]() {
    const BatchMatMulArguments a = ChooseBatchMatMulArguments(false);
    return ExpectTfAndXlaOutputsAreClose(OpTestBuilder("BatchMatMul")
                                             .RandomInput(a.dtype, a.lhs_dims)
                                             .RandomInput(a.dtype, a.rhs_dims)
                                             .Attr("T", a.dtype)
                                             .Attr("adj_x", a.adj_lhs)
                                             .Attr("adj_y", a.adj_rhs));
  });
}

TEST_F(OpTest, BatchMatMulV2) {
  if (tensorflow::tf_xla_test_use_mlir) GTEST_SKIP() << "b/201095155";
  // :randomized_tests_seeded is flaky with --tf_xla_random_seed=200839030
  // See b/229622638.
  if (!tensorflow::tf_xla_test_use_mlir) GTEST_SKIP() << "b/197140886";
  Repeatedly([this]() {
    const BatchMatMulArguments a = ChooseBatchMatMulArguments(true);
    return ExpectTfAndXlaOutputsAreClose(OpTestBuilder("BatchMatMulV2")
                                             .RandomInput(a.dtype, a.lhs_dims)
                                             .RandomInput(a.dtype, a.rhs_dims)
                                             .Attr("T", a.dtype)
                                             .Attr("adj_x", a.adj_lhs)
                                             .Attr("adj_y", a.adj_rhs));
  });
}

TEST_F(OpTest, BatchToSpace) {
  if (tensorflow::tf_xla_test_use_mlir) GTEST_SKIP() << "b/201095155";
  if (!tensorflow::tf_xla_test_use_mlir) GTEST_SKIP() << "b/197140886";
  Repeatedly([this]() {
    const int num_block_dims = 2;
    std::vector<int64_t> block_dims =
        RandomDims(num_block_dims, num_block_dims, 0, 5);
    int64_t block_size = RandomDim(2, 5);

    std::vector<int64_t> input_dims(1 + num_block_dims + 1);
    input_dims[0] = RandomDim();
    for (int i = 0; i < num_block_dims; ++i) {
      input_dims[0] *= block_size;
      input_dims[1 + i] = block_dims[i];
    }
    input_dims[1 + num_block_dims] = RandomDim();

    std::vector<int64_t> crop_vals;
    std::uniform_int_distribution<int> distribution(0, 4);
    for (int i = 0; i < num_block_dims; ++i) {
      // Chooses crop values; does not always choose legal values.
      crop_vals.push_back(distribution(generator()));
      crop_vals.push_back(distribution(generator()));
    }
    Tensor crops;
    CHECK(crops.CopyFrom(AsIntTensor(DT_INT32, crop_vals),
                         TensorShape({num_block_dims, 2})));

    auto type = Choose<DataType>({DT_INT32, DT_FLOAT, DT_COMPLEX64});
    return ExpectTfAndXlaOutputsAreClose(OpTestBuilder("BatchToSpace")
                                             .RandomInput(type, input_dims)
                                             .Input(crops)
                                             .Attr("T", type)
                                             .Attr("block_size", block_size));
  });
}

TEST_F(OpTest, BatchToSpaceND) {
  Repeatedly([this]() {
    std::vector<int64_t> block_dims = RandomDims(1, 3, 0, 5);
    int num_block_dims = block_dims.size();
    std::vector<int64_t> remaining_dims = RandomDims(0, 3);
    std::vector<int64_t> block_multipliers =
        RandomDims(block_dims.size(), block_dims.size(), 0, 4);

    std::vector<int64_t> input_dims(1 + num_block_dims + remaining_dims.size());
    input_dims[0] = RandomDim();
    for (int i = 0; i < num_block_dims; ++i) {
      input_dims[0] *= block_dims[i];
    }
    std::copy(block_multipliers.begin(), block_multipliers.end(),
              input_dims.begin() + 1);
    std::copy(remaining_dims.begin(), remaining_dims.end(),
              input_dims.begin() + 1 + num_block_dims);

    std::vector<int64_t> crop_vals;
    std::uniform_int_distribution<int> distribution(0, 3);
    for (int i = 0; i < num_block_dims; ++i) {
      // Chooses crop values; does not always choose legal values.
      crop_vals.push_back(distribution(generator()));
      crop_vals.push_back(distribution(generator()));
    }
    Tensor crops;
    CHECK(crops.CopyFrom(AsIntTensor(DT_INT32, crop_vals),
                         TensorShape({num_block_dims, 2})));

    auto type = Choose<DataType>({DT_INT32, DT_FLOAT, DT_COMPLEX64});
    return ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("BatchToSpaceND")
            .RandomInput(type, input_dims)
            .Input(test::AsTensor<int32>(
                std::vector<int32>(block_dims.begin(), block_dims.end())))
            .Input(crops)
            .Attr("T", type));
  });
}

TEST_F(OpTest, BiasAdd) {
  Repeatedly([this]() {
    auto x_dims = RandomDims(2, kDefaultMaxRank);
    auto y_dims = {x_dims[x_dims.size() - 1]};
    // TODO(phawkins): test both data formats.
    auto type = Choose<DataType>({DT_INT32, DT_FLOAT, DT_COMPLEX64});
    return ExpectTfAndXlaOutputsAreClose(OpTestBuilder("BiasAdd")
                                             .RandomInput(type, x_dims)
                                             .RandomInput(type, y_dims)
                                             .Attr("T", type));
  });
}

TEST_F(OpTest, BiasAddGrad) {
  Repeatedly([this]() {
    // TODO(phawkins): test both data formats.
    auto type = Choose<DataType>({DT_INT32, DT_FLOAT, DT_COMPLEX64});
    return ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("BiasAddGrad").RandomInput(type).Attr("T", type));
  });
}

TEST_F(OpTest, BiasAddV1) {
  Repeatedly([this]() {
    auto x_dims = RandomDims(2, kDefaultMaxRank);
    auto y_dims = {x_dims[x_dims.size() - 1]};
    auto type = Choose<DataType>({DT_INT32, DT_FLOAT, DT_COMPLEX64});
    return ExpectTfAndXlaOutputsAreClose(OpTestBuilder("BiasAddV1")
                                             .RandomInput(type, x_dims)
                                             .RandomInput(type, y_dims)
                                             .Attr("T", type));
  });
}

TEST_F(OpTest, Bitcast) {
  if (tensorflow::tf_xla_test_use_mlir) GTEST_SKIP() << "b/201095155";
  Repeatedly([this]() {  // NOLINT: due to GTEST_SKIP
    auto src_type = Choose<DataType>(kAllNumberTypes);
    auto dst_type = Choose<DataType>(kAllNumberTypes);
    return ExpectTfAndXlaOutputsAreClose(OpTestBuilder("Bitcast")
                                             .RandomInput(src_type)
                                             .Attr("T", src_type)
                                             .Attr("type", dst_type));
  });
}

TEST_F(OpTest, BitwiseAnd) {
  Repeatedly([this]() {
    DataType type = DT_INT32;
    auto dims = BroadcastableDims();
    return ExpectTfAndXlaOutputsAreClose(OpTestBuilder("BitwiseAnd")
                                             .RandomInput(type, dims.first)
                                             .RandomInput(type, dims.second)
                                             .Attr("T", type));
  });
}

TEST_F(OpTest, BitwiseOr) {
  Repeatedly([this]() {
    DataType type = DT_INT32;
    auto dims = BroadcastableDims();
    return ExpectTfAndXlaOutputsAreClose(OpTestBuilder("BitwiseOr")
                                             .RandomInput(type, dims.first)
                                             .RandomInput(type, dims.second)
                                             .Attr("T", type));
  });
}

TEST_F(OpTest, BitwiseXor) {
  Repeatedly([this]() {
    auto dims = BroadcastableDims();
    return ExpectTfAndXlaOutputsAreClose(OpTestBuilder("BitwiseXor")
                                             .RandomInput(DT_INT32, dims.first)
                                             .RandomInput(DT_INT32, dims.second)
                                             .Attr("T", DT_INT32));
  });
}

TEST_F(OpTest, BroadcastArgs) {
  Repeatedly([this]() {
    // TODO(phawkins): only int32 seems to be implemented in Tensorflow.
    // auto type = Choose<DataType>({DT_INT32, DT_INT64});
    DataType type = DT_INT32;
    auto dims = BroadcastableDims();
    return ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("BroadcastArgs")
            .Input(AsIntTensor(type, dims.first))
            .Input(AsIntTensor(type, dims.second))
            .Attr("T", type));
  });
}

TEST_F(OpTest, BroadcastGradientArgs) {
  Repeatedly([this]() {
    // TODO(phawkins): only int32 seems to be implemented in Tensorflow.
    // auto type = Choose<DataType>({DT_INT32, DT_INT64});
    DataType type = DT_INT32;
    auto dims = BroadcastableDims();
    return ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("BroadcastGradientArgs")
            .Input(AsIntTensor(type, dims.first))
            .Input(AsIntTensor(type, dims.second))
            .Attr("T", type));
  });
}

TEST_F(OpTest, BroadcastTo) {
  Repeatedly([this]() {
    auto type = Choose<DataType>(kAllXlaTypes);
    auto type_idx = Choose<DataType>({DT_INT32, DT_INT64});
    auto dims_to = RandomDims();
    auto dims_from = BroadcastableToDims(dims_to);
    return ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("BroadcastTo")
            .RandomInput(type, dims_from)
            .Input(AsIntTensor(type_idx, dims_to))
            .Attr("T", type)
            .Attr("Tidx", type_idx));
  });
}

TEST_F(OpTest, Cast) {
  Repeatedly([this]() {
    DataType src_type, dst_type;
    src_type = Choose<DataType>({DT_INT32, DT_FLOAT, DT_BOOL, DT_COMPLEX64});
    dst_type = Choose<DataType>({DT_INT32, DT_FLOAT, DT_BOOL, DT_COMPLEX64});
    return ExpectTfAndXlaOutputsAreClose(OpTestBuilder("Cast")
                                             .RandomInput(src_type)
                                             .Attr("SrcT", src_type)
                                             .Attr("DstT", dst_type));
  });
}

TEST_F(OpTest, CastBF16) {
  if (tensorflow::tf_xla_test_use_mlir) GTEST_SKIP() << "b/201095155";
  Repeatedly([this]() {
    DataType src_type, dst_type;
    src_type = Choose<DataType>({DT_FLOAT});
    dst_type = Choose<DataType>({DT_BFLOAT16});
    return ExpectTfAndXlaOutputsAreClose(OpTestBuilder("Cast")
                                             .RandomInput(src_type)
                                             .Attr("SrcT", src_type)
                                             .Attr("DstT", dst_type)
                                             .Attr("Truncate", true));
  });
}

TEST_F(OpTest, Ceil) {
  Repeatedly([this]() {
    return ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("Ceil").RandomInput(DT_FLOAT).Attr("T", DT_FLOAT));
  });
}

TEST_F(OpTest, ClipByValue) {
  // TODO(b/211012085): Change input_dims to BroadcastableDimsN(3). The
  //                    compiled ClipByValue fails in this case.
  //                    --tf_xla_random_seed=200839030
  Repeatedly([this]() {
    auto type = Choose<DataType>({DT_INT32, DT_INT64, DT_FLOAT});
    // ClipByValue requires that broadcasting min and max tensors do not cause
    // the returned shape to be larger than the input shape.
    auto input_dims = RandomDims();
    // clip_value_min must be <= clip_value_max for correct results. Different
    // implementations handle the max < min case differently, so ensure that
    // min <= max.
    auto min_max_dims = BroadcastableToDims(input_dims);
    auto min_max = RandomLteTensors(type, min_max_dims);
    return ExpectTfAndXlaOutputsAreClose(OpTestBuilder("ClipByValue")
                                             .RandomInput(type, input_dims)
                                             .Input(min_max.first)
                                             .Input(min_max.second)
                                             .Attr("T", type));
  });
}

TEST_F(OpTest, Complex) {
  Repeatedly([this]() {
    auto dims = BroadcastableDims();
    return ExpectTfAndXlaOutputsAreClose(OpTestBuilder("Complex")
                                             .RandomInput(DT_FLOAT, dims.first)
                                             .RandomInput(DT_FLOAT, dims.second)
                                             .Attr("T", DT_FLOAT));
  });
}

TEST_F(OpTest, Concat) {
  Repeatedly([this]() {  // NOLINT: due to GTEST_SKIP
    ConcatArguments a = ChooseConcatArguments(false);
    return ExpectTfAndXlaOutputsAreClose(OpTestBuilder("Concat")
                                             .Input(a.axis)
                                             .VariadicInput(a.values)
                                             .Attr("N", a.n)
                                             .Attr("T", a.type));
  });
}

TEST_F(OpTest, ConcatV2) {
  Repeatedly([this]() {
    ConcatArguments a = ChooseConcatArguments(true);
    return ExpectTfAndXlaOutputsAreClose(OpTestBuilder("ConcatV2")
                                             .VariadicInput(a.values)
                                             .Input(a.axis)
                                             .Attr("N", a.n)
                                             .Attr("T", a.type)
                                             .Attr("Tidx", a.type_idx));
  });
}

TEST_F(OpTest, ConcatOffset) {
  Repeatedly([this]() {
    int n = std::uniform_int_distribution<int>(2, 5)(generator());

    std::vector<int64_t> dims = RandomDims(1);
    int concat_dim =
        std::uniform_int_distribution<int32>(0, dims.size() - 1)(generator());

    OpTestBuilder builder("ConcatOffset");
    builder.Input(test::AsScalar<int32>(concat_dim));
    builder.Attr("N", n);
    for (int i = 0; i < n; ++i) {
      std::vector<int32> shape(dims.begin(), dims.end());
      shape[concat_dim] = RandomDim();
      builder.Input(test::AsTensor<int32>(shape));
    }
    return ExpectTfAndXlaOutputsAreClose(builder);
  });
}

TEST_F(OpTest, Conj) {
  Repeatedly([this]() {
    return ExpectTfAndXlaOutputsAreClose(OpTestBuilder("Conj")
                                             .RandomInput(DT_COMPLEX64)
                                             .Attr("T", DT_COMPLEX64));
  });
}

TEST_F(OpTest, Const) {
  Repeatedly([this]() {
    auto type = Choose<DataType>({DT_FLOAT});
    return ExpectTfAndXlaOutputsAreClose(OpTestBuilder("Const")
                                             .Attr("value", RandomTensor(type))
                                             .Attr("dtype", type));
  });
}

TEST_F(OpTest, FFT) {
  Repeatedly([this]() {
    std::vector<int64_t> dims = RandomDims(1, kDefaultMaxRank);
    return ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("FFT").RandomInput(DT_COMPLEX64, dims));
  });
}

TEST_F(OpTest, FFT2D) {
  Repeatedly([this]() {
    std::vector<int64_t> dims = RandomDims(2, kDefaultMaxRank);
    return ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("FFT2D").RandomInput(DT_COMPLEX64, dims));
  });
}

TEST_F(OpTest, FFT3D) {
  Repeatedly([this]() {
    std::vector<int64_t> dims = RandomDims(3, kDefaultMaxRank);
    return ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("FFT3D").RandomInput(DT_COMPLEX64, dims));
  });
}

TEST_F(OpTest, IFFT) {
  Repeatedly([this]() {
    std::vector<int64_t> dims = RandomDims(1, kDefaultMaxRank);
    return ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("IFFT").RandomInput(DT_COMPLEX64, dims));
  });
}

TEST_F(OpTest, IFFT2D) {
  Repeatedly([this]() {
    std::vector<int64_t> dims = RandomDims(2, kDefaultMaxRank);
    return ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("IFFT2D").RandomInput(DT_COMPLEX64, dims));
  });
}

TEST_F(OpTest, IFFT3D) {
  Repeatedly([this]() {
    std::vector<int64_t> dims = RandomDims(3, kDefaultMaxRank);
    return ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("IFFT3D").RandomInput(DT_COMPLEX64, dims));
  });
}

TEST_F(OpTest, RFFT) {
  Repeatedly([this]() {
    std::vector<int64_t> dims = RandomDims(1, kDefaultMaxRank, 3);
    Tensor fft_shape = test::AsTensor<int32>(AsInt32s({dims[dims.size() - 1]}));
    return ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("RFFT").RandomInput(DT_FLOAT, dims).Input(fft_shape));
  });
}

TEST_F(OpTest, RFFT2D) {
  Repeatedly([this]() {
    std::vector<int64_t> dims = RandomDims(2, kDefaultMaxRank, 3);
    Tensor fft_shape = test::AsTensor<int32>(
        AsInt32s({dims[dims.size() - 2], dims[dims.size() - 1]}));
    return ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("RFFT2D").RandomInput(DT_FLOAT, dims).Input(fft_shape));
  });
}

TEST_F(OpTest, RFFT3D) {
  Repeatedly([this]() {
    std::vector<int64_t> dims = RandomDims(3, kDefaultMaxRank, 3);
    Tensor fft_shape = test::AsTensor<int32>(AsInt32s(
        {dims[dims.size() - 3], dims[dims.size() - 2], dims[dims.size() - 1]}));
    return ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("RFFT3D").RandomInput(DT_FLOAT, dims).Input(fft_shape));
  });
}

TEST_F(OpTest, IRFFT) {
  Repeatedly([this]() {
    std::vector<int64_t> dims = RandomDims(1, kDefaultMaxRank, 3);
    int64_t orig_size = dims[dims.size() - 1];
    dims[dims.size() - 1] = dims[dims.size() - 1] / 2 + 1;
    Tensor fft_shape = test::AsTensor<int32>(AsInt32s({orig_size}));
    return ExpectTfAndXlaOutputsAreClose(OpTestBuilder("IRFFT")
                                             .RandomInput(DT_COMPLEX64, dims)
                                             .Input(fft_shape));
  });
}

TEST_F(OpTest, IRFFT2D) {
  Repeatedly([this]() {
    std::vector<int64_t> dims = RandomDims(2, kDefaultMaxRank, 3);
    std::vector<int64_t> orig_size = {dims[dims.size() - 2],
                                      dims[dims.size() - 1]};
    dims[dims.size() - 1] = dims[dims.size() - 1] / 2 + 1;
    Tensor fft_shape = test::AsTensor<int32>(AsInt32s({orig_size}));
    return ExpectTfAndXlaOutputsAreClose(OpTestBuilder("IRFFT2D")
                                             .RandomInput(DT_COMPLEX64, dims)
                                             .Input(fft_shape));
  });
}

TEST_F(OpTest, IRFFT3D) {
  Repeatedly([this]() {
    std::vector<int64_t> dims = RandomDims(3, kDefaultMaxRank, 3);
    std::vector<int64_t> orig_size = {
        dims[dims.size() - 3], dims[dims.size() - 2], dims[dims.size() - 1]};
    dims[dims.size() - 1] = dims[dims.size() - 1] / 2 + 1;
    Tensor fft_shape = test::AsTensor<int32>(AsInt32s({orig_size}));
    return ExpectTfAndXlaOutputsAreClose(OpTestBuilder("IRFFT3D")
                                             .RandomInput(DT_COMPLEX64, dims)
                                             .Input(fft_shape));
  });
}

TEST_F(OpTest, Conv2D) {
  if (tensorflow::tf_xla_test_use_mlir) GTEST_SKIP() << "b/201095155";
  if (!tensorflow::tf_xla_test_use_mlir) GTEST_SKIP() << "b/197140886";
  Repeatedly([this]() {
    WindowedSpatialDims d = ChooseWindowedSpatialDims(2);
    std::uniform_int_distribution<int> random_int(1, 5);
    int features_in = random_int(generator());
    int features_out = random_int(generator());

    int64_t batch = RandomDim();

    std::vector<int64_t> data_dims =
        ImageDims(FORMAT_NHWC, batch, features_in, d.input_dims);

    std::vector<int64_t> kernel_dims = {d.kernel_dims[0], d.kernel_dims[1],
                                        features_in, features_out};
    DataType type = DT_FLOAT;
    return ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("Conv2D")
            .RandomInput(type, data_dims)
            .RandomInput(type, kernel_dims)
            .Attr("T", type)
            .Attr("strides", ImageDims(FORMAT_NHWC, 1, 1, d.stride_dims))
            .Attr("padding", d.padding == SAME ? "SAME" : "VALID")
            .Attr("data_format", "NHWC"));
  });
}

TEST_F(OpTest, Conv2DBackpropFilter) {
  Repeatedly([this]() {
    WindowedSpatialDims d = ChooseWindowedSpatialDims(2);
    std::uniform_int_distribution<int> random_int(1, 5);
    int features_in = random_int(generator());
    int features_out = random_int(generator());
    int32_t batch = RandomDim();
    std::vector<int64_t> activations =
        ImageDims(FORMAT_NHWC, batch, features_in, d.input_dims);
    std::vector<int64_t> backprop =
        ImageDims(FORMAT_NHWC, batch, features_out, d.output_dims);
    Tensor kernel_shape = test::AsTensor<int32>(AsInt32s(
        {d.kernel_dims[0], d.kernel_dims[1], features_in, features_out}));
    DataType type = DT_FLOAT;
    return ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("Conv2DBackpropFilter")
            .RandomInput(type, activations)
            .Input(kernel_shape)
            .RandomInput(type, backprop)
            .Attr("T", type)
            .Attr("strides", ImageDims(FORMAT_NHWC, 1, 1, d.stride_dims))
            .Attr("padding", d.padding == SAME ? "SAME" : "VALID")
            .Attr("data_format", "NHWC"));
  });
}

TEST_F(OpTest, Conv2DBackpropInput) {
  Repeatedly([this]() {
    WindowedSpatialDims d = ChooseWindowedSpatialDims(2);
    std::uniform_int_distribution<int> random_int(1, 5);
    int features_in = random_int(generator());
    int features_out = random_int(generator());
    int32_t batch = RandomDim();
    Tensor in_shape = test::AsTensor<int32>(
        AsInt32s(ImageDims(FORMAT_NHWC, batch, features_in, d.input_dims)));
    std::vector<int64_t> backprop =
        ImageDims(FORMAT_NHWC, batch, features_out, d.output_dims);
    std::vector<int64_t> kernel = {d.kernel_dims[0], d.kernel_dims[1],
                                   features_in, features_out};
    DataType type = DT_FLOAT;
    return ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("Conv2DBackpropInput")
            .Input(in_shape)
            .RandomInput(type, kernel)
            .RandomInput(type, backprop)
            .Attr("T", type)
            .Attr("strides", ImageDims(FORMAT_NHWC, 1, 1, d.stride_dims))
            .Attr("padding", d.padding == SAME ? "SAME" : "VALID")
            .Attr("data_format", "NHWC"));
  });
}

TEST_F(OpTest, Conv3D) {
  if (tensorflow::tf_xla_test_use_mlir) GTEST_SKIP() << "b/201095155";
  if (!tensorflow::tf_xla_test_use_mlir) GTEST_SKIP() << "b/197140886";
  Repeatedly([this]() {
    WindowedSpatialDims d = ChooseWindowedSpatialDims(3);
    std::uniform_int_distribution<int> random_int(1, 5);
    int features_in = random_int(generator());
    int features_out = random_int(generator());
    std::vector<int64_t> data = {RandomDim(), d.input_dims[0], d.input_dims[1],
                                 d.input_dims[2], features_in};

    std::vector<int64_t> kernel = {d.kernel_dims[0], d.kernel_dims[1],
                                   d.kernel_dims[2], features_in, features_out};
    DataType type = DT_FLOAT;
    return ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("Conv3D")
            .RandomInput(type, data)
            .RandomInput(type, kernel)
            .Attr("T", type)
            .Attr("strides", ImageDims(FORMAT_NHWC, 1, 1, d.stride_dims))
            .Attr("padding", d.padding == SAME ? "SAME" : "VALID"));
  });
}

TEST_F(OpTest, Conv3DBackpropFilter) {
  if (tensorflow::tf_xla_test_use_mlir) GTEST_SKIP() << "b/201095155";
  if (!tensorflow::tf_xla_test_use_mlir) GTEST_SKIP() << "b/197140886";
  Repeatedly([this]() {
    WindowedSpatialDims d = ChooseWindowedSpatialDims(3);
    std::uniform_int_distribution<int> random_int(1, 5);
    int features_in = random_int(generator());
    int features_out = random_int(generator());
    int32_t batch = RandomDim(1);
    std::vector<int64_t> activations =
        ImageDims(FORMAT_NHWC, batch, features_in, d.input_dims);
    std::vector<int64_t> backprop =
        ImageDims(FORMAT_NHWC, batch, features_out, d.output_dims);
    Tensor kernel_shape = test::AsTensor<int32>(
        AsInt32s({d.kernel_dims[0], d.kernel_dims[1], d.kernel_dims[2],
                  features_in, features_out}));
    DataType type = DT_FLOAT;
    return ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("Conv3DBackpropFilterV2")
            .RandomInput(type, activations)
            .Input(kernel_shape)
            .RandomInput(type, backprop)
            .Attr("T", type)
            .Attr("strides", ImageDims(FORMAT_NHWC, 1, 1, d.stride_dims))
            .Attr("padding", d.padding == SAME ? "SAME" : "VALID"));
  });
}

TEST_F(OpTest, Conv3DBackpropInput) {
  if (tensorflow::tf_xla_test_use_mlir) GTEST_SKIP() << "b/201095155";
  if (!tensorflow::tf_xla_test_use_mlir) GTEST_SKIP() << "b/197140886";
  Repeatedly([this]() {
    WindowedSpatialDims d = ChooseWindowedSpatialDims(3);
    std::uniform_int_distribution<int> random_int(1, 5);
    int features_in = random_int(generator());
    int features_out = random_int(generator());
    int32_t batch = RandomDim(1);
    Tensor in_shape = test::AsTensor<int32>(
        AsInt32s(ImageDims(FORMAT_NHWC, batch, features_in, d.input_dims)));
    std::vector<int64_t> backprop =
        ImageDims(FORMAT_NHWC, batch, features_out, d.output_dims);
    std::vector<int64_t> kernel = {d.kernel_dims[0], d.kernel_dims[1],
                                   d.kernel_dims[2], features_in, features_out};
    auto type = Choose<DataType>({DT_FLOAT, DT_COMPLEX64});
    return ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("Conv3DBackpropInputV2")
            .Input(in_shape)
            .RandomInput(type, kernel)
            .RandomInput(type, backprop)
            .Attr("T", type)
            .Attr("strides", ImageDims(FORMAT_NHWC, 1, 1, d.stride_dims))
            .Attr("padding", d.padding == SAME ? "SAME" : "VALID"));
  });
}

TEST_F(OpTest, ComplexAbs) {
  Repeatedly([this]() {
    auto type = DT_COMPLEX64;
    auto type_out = DT_FLOAT;
    return ExpectTfAndXlaOutputsAreClose(OpTestBuilder("ComplexAbs")
                                             .RandomInput(type)
                                             .Attr("T", type)
                                             .Attr("Tout", type_out));
  });
}

TEST_F(OpTest, Cos) {
  Repeatedly([this]() {
    auto type = Choose<DataType>({DT_FLOAT, DT_COMPLEX64});
    return ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("Cos").RandomInput(type).Attr("T", type));
  });
}

TEST_F(OpTest, Cosh) {
  Repeatedly([this]() {
    auto type = Choose<DataType>({DT_FLOAT, DT_COMPLEX64});
    return ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("Cosh").RandomInput(type).Attr("T", type));
  });
}

TEST_F(OpTest, DepthToSpace) {
  Repeatedly([this]() {
    int64_t block = RandomDim(2, 5);
    std::vector<int64_t> input_dims = RandomDims(4, 4);
    input_dims[1] = (input_dims[1] + (block - 1)) / block;
    input_dims[2] = (input_dims[2] + (block - 1)) / block;
    input_dims[3] *= block * block;
    auto type = Choose<DataType>(kAllXlaTypes);
    return ExpectTfAndXlaOutputsAreClose(OpTestBuilder("DepthToSpace")
                                             .RandomInput(type, input_dims)
                                             .Attr("T", type)
                                             .Attr("block_size", block));
  });
}

TEST_F(OpTest, DepthwiseConv2DNative) {
  if (tensorflow::tf_xla_test_use_mlir) GTEST_SKIP() << "b/201095155";
  if (!tensorflow::tf_xla_test_use_mlir) GTEST_SKIP() << "b/197140886";
  Repeatedly([this]() {
    WindowedSpatialDims d = ChooseWindowedSpatialDims(2);
    std::uniform_int_distribution<int> random_int(1, 5);
    int features_in = random_int(generator());
    int depth_multiplier = random_int(generator());
    std::vector<int64_t> input_dims = {RandomDim(), d.input_dims[0],
                                       d.input_dims[1], features_in};

    std::vector<int64_t> kernel_dims = {d.kernel_dims[0], d.kernel_dims[1],
                                        features_in, depth_multiplier};
    std::vector<int64_t> strides = ImageDims(FORMAT_NHWC, 1, 1, d.stride_dims);
    strides[2] = strides[1];  // Current impl only supports equal strides
    return ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("DepthwiseConv2dNative")
            .RandomInput(DT_FLOAT, input_dims)
            .RandomInput(DT_FLOAT, kernel_dims)
            .Attr("T", DT_FLOAT)
            .Attr("strides", strides)
            .Attr("padding", d.padding == SAME ? "SAME" : "VALID"));
  });
}

TEST_F(OpTest, DepthwiseConv2DNativeBackpropFilter) {
  if (tensorflow::tf_xla_test_use_mlir) GTEST_SKIP() << "b/201095155";
  if (!tensorflow::tf_xla_test_use_mlir) GTEST_SKIP() << "b/197140886";
  Repeatedly([this]() {
    WindowedSpatialDims d = ChooseWindowedSpatialDims(2);
    std::uniform_int_distribution<int> random_int(1, 5);
    int features_in = random_int(generator());
    int depth_multiplier = random_int(generator());
    int32_t batch = RandomDim();
    std::vector<int64_t> activations =
        ImageDims(FORMAT_NHWC, batch, features_in, d.input_dims);
    std::vector<int64_t> backprop = ImageDims(
        FORMAT_NHWC, batch, features_in * depth_multiplier, d.output_dims);
    Tensor kernel_shape = test::AsTensor<int32>(AsInt32s(
        {d.kernel_dims[0], d.kernel_dims[1], features_in, depth_multiplier}));
    std::vector<int64_t> strides = ImageDims(FORMAT_NHWC, 1, 1, d.stride_dims);
    strides[2] = strides[1];  // Current impl only supports equal strides
    return ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("DepthwiseConv2dNativeBackpropFilter")
            .RandomInput(DT_FLOAT, activations)
            .Input(kernel_shape)
            .RandomInput(DT_FLOAT, backprop)
            .Attr("T", DT_FLOAT)
            .Attr("strides", strides)
            .Attr("padding", d.padding == SAME ? "SAME" : "VALID")
            .Attr("data_format", "NHWC"));
  });
}

TEST_F(OpTest, DepthwiseConv2DBackpropInput) {
  if (tensorflow::tf_xla_test_use_mlir) GTEST_SKIP() << "b/201095155";
  if (!tensorflow::tf_xla_test_use_mlir) GTEST_SKIP() << "b/197140886";
  Repeatedly([this]() {
    WindowedSpatialDims d = ChooseWindowedSpatialDims(2);
    std::uniform_int_distribution<int> random_int(1, 5);
    int features_in = random_int(generator());
    int depth_multiplier = random_int(generator());
    int32_t batch = RandomDim();
    Tensor in_shape = test::AsTensor<int32>(
        AsInt32s(ImageDims(FORMAT_NHWC, batch, features_in, d.input_dims)));
    std::vector<int64_t> backprop = ImageDims(
        FORMAT_NHWC, batch, features_in * depth_multiplier, d.output_dims);
    std::vector<int64_t> kernel = {d.kernel_dims[0], d.kernel_dims[1],
                                   features_in, depth_multiplier};
    std::vector<int64_t> strides = ImageDims(FORMAT_NHWC, 1, 1, d.stride_dims);
    strides[2] = strides[1];  // Current impl only supports equal strides
    return ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("DepthwiseConv2dNativeBackpropInput")
            .Input(in_shape)
            .RandomInput(DT_FLOAT, kernel)
            .RandomInput(DT_FLOAT, backprop)
            .Attr("T", DT_FLOAT)
            .Attr("strides", strides)
            .Attr("padding", d.padding == SAME ? "SAME" : "VALID")
            .Attr("data_format", "NHWC"));
  });
}

TEST_F(OpTest, Diag) {
  Repeatedly([this]() {
    auto type = Choose<DataType>(kAllXlaTypes);
    std::vector<int64_t> dims;
    // Diag causes a quadratic blowup in output size.
    int64_t size;
    do {
      dims = RandomDims(1);
      size = TensorShape(dims).num_elements();
    } while (size * size > tf_xla_max_tensor_size);
    return ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("Diag").RandomInput(type, dims).Attr("T", type));
  });
}

TEST_F(OpTest, DiagPart) {
  Repeatedly([this]() {
    auto type = Choose<DataType>(kAllXlaTypes);
    auto dims = RandomDims(1, 3);
    // Duplicate the random dims.
    std::vector<int64_t> doubled_dims(dims.size() * 2);
    std::copy(dims.begin(), dims.end(), doubled_dims.begin());
    std::copy(dims.begin(), dims.end(), doubled_dims.begin() + dims.size());
    return ExpectTfAndXlaOutputsAreClose(OpTestBuilder("DiagPart")
                                             .RandomInput(type, doubled_dims)
                                             .Attr("T", type));
  });
}

TEST_F(OpTest, Digamma) {
  Repeatedly([this]() {
    return ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("Digamma").RandomInput(DT_FLOAT).Attr("T", DT_FLOAT));
  });
}

TEST_F(OpTest, Div) {
  Repeatedly([this]() {
    auto type = Choose<DataType>({DT_INT32, DT_FLOAT, DT_COMPLEX64});
    auto dims = BroadcastableDims();
    return ExpectTfAndXlaOutputsAreClose(OpTestBuilder("Div")
                                             .RandomInput(type, dims.first)
                                             .RandomInput(type, dims.second)
                                             .Attr("T", type));
  });
}

TEST_F(OpTest, DivNoNan) {
  Repeatedly([this]() {
    auto type = Choose<DataType>({DT_FLOAT, DT_COMPLEX64});
    auto dims = BroadcastableDims();
    return ExpectTfAndXlaOutputsAreClose(OpTestBuilder("DivNoNan")
                                             .RandomInput(type, dims.first)
                                             .RandomInput(type, dims.second)
                                             .Attr("T", type));
  });
}

TEST_F(OpTest, DynamicStitch) {
  Repeatedly([this]() {
    auto type = Choose<DataType>(kAllXlaTypes);
    int n = std::uniform_int_distribution<int>(2, 5)(generator());
    OpTestBuilder builder("DynamicStitch");
    builder.Attr("T", type);
    builder.Attr("N", n);
    std::vector<std::vector<int64_t>> index_dims;
    int size = 0;
    // TODO(phawkins): the XLA implementation of DynamicStitch does not
    // accept an empty set of indices.
    do {
      size = 0;
      index_dims.clear();
      for (int i = 0; i < n; ++i) {
        std::vector<int64_t> dims = RandomDims(0, 3, 0, 5);
        size += TensorShape(dims).num_elements();
        index_dims.push_back(dims);
      }
    } while (size == 0);

    // Shuffle the range of indices that cover the output.
    // TODO(phawkins): The documentation for DynamicStitch doesn't require
    // that the indices cover all positions of the output. The XLA
    // implementation does so require. However, the native TF implementation
    // leaves undefined values if we don't cover everything, so we can't
    // really test that case anyway.
    std::vector<int32> indices(size);
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), generator());

    int pos = 0;
    for (int i = 0; i < n; ++i) {
      TensorShape shape(index_dims[i]);
      Tensor t = test::AsTensor<int32>(
          absl::Span<const int32>(indices).subspan(pos, shape.num_elements()),
          shape);
      builder.Input(t);
      pos += t.NumElements();
    }

    std::vector<int64_t> constant_dims = RandomDims(0, 3, 0, 5);
    for (int i = 0; i < n; ++i) {
      std::vector<int64_t> dims(index_dims[i].begin(), index_dims[i].end());
      std::copy(constant_dims.begin(), constant_dims.end(),
                std::back_inserter(dims));
      builder.RandomInput(type, dims);
    }
    return ExpectTfAndXlaOutputsAreClose(builder);
  });
}

TEST_F(OpTest, Einsum) {
  Repeatedly([this]() {
    const EinsumArguments a = ChooseEinsumArguments();
    return ExpectTfAndXlaOutputsAreClose(OpTestBuilder("Einsum")
                                             .RandomInput(a.type, a.lhs_dims)
                                             .RandomInput(a.type, a.rhs_dims)
                                             .Attr("equation", a.equation)
                                             .Attr("T", a.type)
                                             .Attr("N", 2));
  });
}

TEST_F(OpTest, Empty) {
  Repeatedly([this]() {
    auto type = Choose<DataType>({kAllXlaTypes});
    return ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("Empty")
            .Input(AsIntTensor(DT_INT32, RandomDims()))
            .Attr("init", true)
            .Attr("dtype", type));
  });
}

TEST_F(OpTest, Elu) {
  Repeatedly([this]() {
    return ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("Elu").RandomInput(DT_FLOAT).Attr("T", DT_FLOAT));
  });
}

TEST_F(OpTest, EluGrad) {
  Repeatedly([this]() {
    auto dims = RandomDims();
    return ExpectTfAndXlaOutputsAreClose(OpTestBuilder("EluGrad")
                                             .RandomInput(DT_FLOAT, dims)
                                             .RandomInput(DT_FLOAT, dims)
                                             .Attr("T", DT_FLOAT));
  });
}

TEST_F(OpTest, ScatterNd) {
  Repeatedly([this]() {
    auto a = ChooseScatterArguments();
    auto shape = test::AsTensor<int32>(
        std::vector<int32>(a.shape.begin(), a.shape.end()));
    return ExpectTfAndXlaOutputsAreClose(OpTestBuilder("ScatterNd")
                                             .Input(a.indices)
                                             .Input(a.updates)
                                             .Input(shape)
                                             .Attr("T", a.type)
                                             .Attr("Tindices", a.indices_type));
  });
}

TEST_F(OpTest, Selu) {
  Repeatedly([this]() {
    return ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("Selu").RandomInput(DT_FLOAT).Attr("T", DT_FLOAT));
  });
}

TEST_F(OpTest, SeluGrad) {
  Repeatedly([this]() {
    auto dims = RandomDims();
    return ExpectTfAndXlaOutputsAreClose(OpTestBuilder("SeluGrad")
                                             .RandomInput(DT_FLOAT, dims)
                                             .RandomInput(DT_FLOAT, dims)
                                             .Attr("T", DT_FLOAT));
  });
}

TEST_F(OpTest, Equal) {
  Repeatedly([this]() {
    auto type = Choose<DataType>({DT_INT32, DT_FLOAT, DT_COMPLEX64});
    auto dims = BroadcastableDims();
    return ExpectTfAndXlaOutputsAreClose(OpTestBuilder("Equal")
                                             .RandomInput(type, dims.first)
                                             .RandomInput(type, dims.second)
                                             .Attr("T", type));
  });
}

TEST_F(OpTest, Erf) {
  Repeatedly([this]() {
    return ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("Erf").RandomInput(DT_FLOAT).Attr("T", DT_FLOAT));
  });
}

TEST_F(OpTest, Erfc) {
  Repeatedly([this]() {
    return ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("Erfc").RandomInput(DT_FLOAT).Attr("T", DT_FLOAT));
  });
}

TEST_F(OpTest, Exp) {
  Repeatedly([this]() {
    auto type = Choose<DataType>({DT_FLOAT, DT_COMPLEX64});
    return ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("Exp").RandomInput(type).Attr("T", type));
  });
}

TEST_F(OpTest, Expm1) {
  Repeatedly([this]() {
    auto type = Choose<DataType>({DT_FLOAT, DT_COMPLEX64});
    return ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("Expm1").RandomInput(type).Attr("T", type));
  });
}

TEST_F(OpTest, ExpandDims) {
  Repeatedly([this]() {
    auto type = Choose<DataType>(kAllXlaTypes);
    std::vector<int64_t> in_dims = RandomDims();
    Tensor dim(DT_INT32, TensorShape());
    std::uniform_int_distribution<int32> d(-1 - in_dims.size(), in_dims.size());
    dim.scalar<int32>()() = d(generator());
    return ExpectTfAndXlaOutputsAreClose(OpTestBuilder("ExpandDims")
                                             .RandomInput(type, in_dims)
                                             .Input(dim)
                                             .Attr("T", type));
  });
}

TEST_F(OpTest, Fill) {
  Repeatedly([this]() {
    auto type = Choose<DataType>(kAllXlaTypes);
    std::vector<int64_t> dims = RandomDims();
    std::vector<int32> shape(dims.begin(), dims.end());
    return ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("Fill")
            .Input(test::AsTensor<int32>(shape))
            .RandomInput(type, {})
            .Attr("T", type));
  });
}

TEST_F(OpTest, Floor) {
  Repeatedly([this]() {
    return ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("Floor").RandomInput(DT_FLOAT).Attr("T", DT_FLOAT));
  });
}

TEST_F(OpTest, FloorDiv) {
  Repeatedly([this]() {
    DataType type = DT_INT32;
    auto dims = BroadcastableDims();
    return ExpectTfAndXlaOutputsAreClose(OpTestBuilder("FloorDiv")
                                             .RandomInput(type, dims.first)
                                             .RandomInput(type, dims.second)
                                             .Attr("T", type));
  });
}

TEST_F(OpTest, FloorMod) {
  Repeatedly([this]() {
    auto type = Choose<DataType>({DT_INT32, DT_FLOAT});
    auto dims = BroadcastableDims();
    return ExpectTfAndXlaOutputsAreClose(OpTestBuilder("FloorMod")
                                             .RandomInput(type, dims.first)
                                             .RandomInput(type, dims.second)
                                             .Attr("T", type));
  });
}

TEST_F(OpTest, Gather) {
  Repeatedly([this]() {
    GatherArguments a = ChooseGatherArguments(true);
    return ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("Gather")
            .RandomInput(a.params_type, a.params_shape)
            .Input(a.indices)
            .Attr("Tparams", a.params_type)
            .Attr("Tindices", a.indices_type));
  });
}

TEST_F(OpTest, GatherV2) {
  Repeatedly([this]() {
    GatherArguments a = ChooseGatherArguments(false);
    return ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("GatherV2")
            .RandomInput(a.params_type, a.params_shape)
            .Input(a.indices)
            .Input(a.axis)
            .Attr("batch_dims", a.batch_dims)
            .Attr("Taxis", a.axis_type)
            .Attr("Tindices", a.indices_type)
            .Attr("Tparams", a.params_type));
  });
}

TEST_F(OpTest, GatherNd) {
  // :randomized_tests_mlir fails with --tf_xla_random_seed=459353625
  // --test_arg=--tf_xla_test_repetitions=100
  if (tensorflow::tf_xla_test_use_mlir) GTEST_SKIP() << "b/201095155";
  // See b/214080339#comment27 as this test causes Kokoro to crash.
  if (!tensorflow::tf_xla_test_use_mlir) GTEST_SKIP() << "b/197140886";
  Repeatedly([this]() {  // NOLINT: due to GTEST_SKIP
    auto params_type = Choose<DataType>(kAllXlaTypes);
    // GatherNd seems undefined on the case where params has rank 0.
    std::vector<int64_t> params_shape = RandomDims(1);
    auto indices_type = DT_INT32;
    std::vector<int64_t> output_outer_shape = RandomDims(0, 4, 0, 32);
    int64_t index_len = RandomDim(0, params_shape.size() + 1);
    std::vector<int64_t> output_shape(output_outer_shape);
    output_shape.push_back(index_len);
    Tensor lo(indices_type, TensorShape(output_shape));
    test::FillFn<int32>(&lo, [](int i) -> int32 { return 0; });
    Tensor hi(indices_type, TensorShape(output_shape));
    test::FillFn<int32>(&hi, [index_len, &params_shape](int i) -> int32 {
      int idx_dim = i % index_len;
      return params_shape[idx_dim] - 1;
    });
    Tensor indices = RandomBoundedTensor(indices_type, lo, hi);
    return ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("GatherNd")
            .RandomInput(params_type, params_shape)
            .Input(indices)
            .Attr("Tindices", indices_type)
            .Attr("Tparams", params_type));
  });
}

TEST_F(OpTest, Greater) {
  Repeatedly([this]() {
    auto type = Choose<DataType>({DT_INT32, DT_FLOAT});
    auto dims = BroadcastableDims();
    return ExpectTfAndXlaOutputsAreClose(OpTestBuilder("Greater")
                                             .RandomInput(type, dims.first)
                                             .RandomInput(type, dims.second)
                                             .Attr("T", type));
  });
}

TEST_F(OpTest, GreaterEqual) {
  Repeatedly([this]() {
    auto type = Choose<DataType>({DT_INT32, DT_FLOAT});
    auto dims = BroadcastableDims();
    return ExpectTfAndXlaOutputsAreClose(OpTestBuilder("GreaterEqual")
                                             .RandomInput(type, dims.first)
                                             .RandomInput(type, dims.second)
                                             .Attr("T", type));
  });
}

TEST_F(OpTest, Identity) {
  Repeatedly([this]() {
    auto type = Choose<DataType>(kAllXlaTypes);
    return ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("Identity").RandomInput(type).Attr("T", type));
  });
}

TEST_F(OpTest, Imag) {
  Repeatedly([this]() {
    return ExpectTfAndXlaOutputsAreClose(OpTestBuilder("Imag")
                                             .RandomInput(DT_COMPLEX64)
                                             .Attr("T", DT_COMPLEX64));
  });
}

TEST_F(OpTest, InplaceUpdate) {
  Repeatedly([this]() {
    auto type = Choose<DataType>(kAllXlaTypes);
    std::vector<int64_t> common_dims =
        RandomDims(0, kDefaultMaxRank - 1, 0, kDefaultMaxDimensionSize);
    // TODO(b/211012712): Once needs_unique_values case is linear instead of
    // quadratic time, use default Dim max instead of 8.
    std::vector<int64_t> v_dims{RandomDim(1, 8)};
    v_dims.insert(v_dims.end(), common_dims.begin(), common_dims.end());
    std::vector<int64_t> x_dims{RandomDim(v_dims[0])};
    x_dims.insert(x_dims.end(), common_dims.begin(), common_dims.end());
    std::vector<int64_t> i_shape{v_dims[0]};
    Tensor i =
        RandomBoundedTensor<int32>(DT_INT32, 0, x_dims[0] - 1, true, i_shape);
    return ExpectTfAndXlaOutputsAreClose(OpTestBuilder("InplaceUpdate")
                                             .RandomInput(type, x_dims)
                                             .Input(i)
                                             .RandomInput(type, v_dims)
                                             .Attr("T", type));
  });
}

TEST_F(OpTest, Inv) {
  Repeatedly([this]() {
    auto type = Choose<DataType>({DT_INT32, DT_FLOAT, DT_COMPLEX64});
    return ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("Inv").RandomInput(type).Attr("T", type));
  });
}

TEST_F(OpTest, Invert) {
  Repeatedly([this]() {
    DataType type = DT_INT32;
    return ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("Invert").RandomInput(type).Attr("T", type));
  });
}

TEST_F(OpTest, InvertPermutation) {
  Repeatedly([this]() {
    // TODO(b/211012712): Once needs_unique_values case is linear instead of
    // quadratic time, use default Dim max instead of 8.
    int64_t len = RandomDim(0, 8);
    Tensor x = RandomBoundedTensor<int32>(DT_INT32, 0, len - 1, true, {len});
    return ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("InvertPermutation").Input(x).Attr("T", DT_INT32));
  });
}

TEST_F(OpTest, IsFinite) {
  Repeatedly([this]() {
    return ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("IsFinite").RandomInput(DT_FLOAT).Attr("T", DT_FLOAT));
  });
}

TEST_F(OpTest, IsInf) {
  Repeatedly([this]() {
    return ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("IsInf").RandomInput(DT_FLOAT).Attr("T", DT_FLOAT));
  });
}

TEST_F(OpTest, IsNan) {
  Repeatedly([this]() {
    return ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("IsNan").RandomInput(DT_FLOAT).Attr("T", DT_FLOAT));
  });
}

TEST_F(OpTest, L2Loss) {
  Repeatedly([this]() {
    DataType type = DT_FLOAT;
    return ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("L2Loss").RandomInput(type).Attr("T", type));
  });
}

TEST_F(OpTest, LeakyRelu) {
  Repeatedly([this]() {
    std::uniform_real_distribution<float> alpha(-2.0f, 2.0f);
    return ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("LeakyRelu")
            .RandomInput(DT_FLOAT)
            .Attr("T", DT_FLOAT)
            .Attr("alpha", alpha(generator())));
  });
}

TEST_F(OpTest, LeakyReluGrad) {
  Repeatedly([this]() {
    auto dims = RandomDims(1);
    std::uniform_real_distribution<float> alpha(-2.0f, 2.0f);
    return ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("LeakyReluGrad")
            .RandomInput(DT_FLOAT, dims)
            .RandomInput(DT_FLOAT, dims)
            .Attr("T", DT_FLOAT)
            .Attr("alpha", alpha(generator())));
  });
}

TEST_F(OpTest, LeftShift) {
  Repeatedly([this]() {
    bool is64 = RandomBool();
    auto dims = RandomDims();
    auto type = is64 ? DT_INT64 : DT_INT32;
    int max_shift = is64 ? 63 : 31;
    auto y = RandomBoundedTensor(type, 0, max_shift, false, dims);
    return ExpectTfAndXlaOutputsAreClose(OpTestBuilder("LeftShift")
                                             .RandomInput(type, dims)
                                             .Input(y)
                                             .Attr("T", type));
  });
}

TEST_F(OpTest, Less) {
  Repeatedly([this]() {
    auto type = Choose<DataType>({DT_INT32, DT_FLOAT});
    auto dims = BroadcastableDims();
    return ExpectTfAndXlaOutputsAreClose(OpTestBuilder("Less")
                                             .RandomInput(type, dims.first)
                                             .RandomInput(type, dims.second)
                                             .Attr("T", type));
  });
}

TEST_F(OpTest, LessEqual) {
  Repeatedly([this]() {
    auto type = Choose<DataType>({DT_INT32, DT_FLOAT});
    auto dims = BroadcastableDims();
    return ExpectTfAndXlaOutputsAreClose(OpTestBuilder("LessEqual")
                                             .RandomInput(type, dims.first)
                                             .RandomInput(type, dims.second)
                                             .Attr("T", type));
  });
}

TEST_F(OpTest, Lgamma) {
  Repeatedly([this]() {
    return ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("Lgamma").RandomInput(DT_FLOAT).Attr("T", DT_FLOAT));
  });
}

TEST_F(OpTest, LinSpace) {
  Repeatedly([this]() {
    auto ToScalar = [](DataType type, int x) {
      if (type == DT_INT32) return test::AsScalar<int32>(x);
      return test::AsScalar<int64_t>(x);
    };
    std::uniform_int_distribution<int> distribution(-50, 50);
    auto type = Choose<DataType>({DT_INT32, DT_INT64});
    return ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("LinSpace")
            .RandomInput(DT_FLOAT, {})
            .RandomInput(DT_FLOAT, {})
            .Input(ToScalar(type, distribution(generator())))
            .Attr("T", DT_FLOAT)
            .Attr("Tidx", type));
  });
}

TEST_F(OpTest, Log) {
  Repeatedly([this]() {
    auto type = Choose<DataType>({DT_FLOAT, DT_COMPLEX64});
    return ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("Log").RandomInput(type).Attr("T", type));
  });
}

TEST_F(OpTest, Log1p) {
  Repeatedly([this]() {
    auto type = Choose<DataType>({DT_FLOAT, DT_COMPLEX64});
    return ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("Log1p").RandomInput(type).Attr("T", DT_FLOAT));
  });
}

TEST_F(OpTest, LogicalAnd) {
  Repeatedly([this]() {
    auto dims = BroadcastableDims();
    return ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("LogicalAnd")
            .RandomInput(DT_BOOL, dims.first)
            .RandomInput(DT_BOOL, dims.second));
  });
}

TEST_F(OpTest, LogicalNot) {
  Repeatedly([this]() {
    return ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("LogicalNot").RandomInput(DT_BOOL));
  });
}

TEST_F(OpTest, LogicalOr) {
  Repeatedly([this]() {
    auto dims = BroadcastableDims();
    return ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("LogicalOr")
            .RandomInput(DT_BOOL, dims.first)
            .RandomInput(DT_BOOL, dims.second));
  });
}

TEST_F(OpTest, LogSoftmax) {
  Repeatedly([this]() {
    return ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("LogSoftmax")
            .RandomInput(DT_FLOAT, RandomDims(2, 2))
            .Attr("T", DT_FLOAT));
  });
}

TEST_F(OpTest, LRN) {
  Repeatedly([this]() {
    // TODO(b/31362467): Crashes with 0 dims on GPU. Re-enable when fixed.
    std::vector<int64_t> data_dims = RandomDims(4, 4, 1, 8);
    // CuDNN requires depth_radius > 0.
    std::uniform_int_distribution<int> radius(1, data_dims[3]);
    std::uniform_real_distribution<float> coeff(0.01, 2.0);
    return ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("LRN")
            .RandomInput(DT_FLOAT, data_dims)
            .Attr("T", DT_FLOAT)
            .Attr("depth_radius", radius(generator()))
            .Attr("bias", coeff(generator()))
            .Attr("alpha", coeff(generator()))
            .Attr("beta", coeff(generator())));
  });
}

TEST_F(OpTest, LRNGrad) {
  Repeatedly([this]() {
    // TODO(b/31362467): Crashes with 0 dims on GPU. Re-enable when fixed.
    std::vector<int64_t> dims = RandomDims(4, 4, 1, 8);
    // CuDNN requires depth_radius > 0.
    std::uniform_int_distribution<int> radius(1, dims[3]);
    std::uniform_real_distribution<float> coeff(0.0, 2.0);
    return ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("LRNGrad")
            .RandomInput(DT_FLOAT, dims)
            .RandomInput(DT_FLOAT, dims)
            .RandomInput(DT_FLOAT, dims)
            .Attr("T", DT_FLOAT)
            .Attr("depth_radius", radius(generator()))
            .Attr("bias", coeff(generator()))
            .Attr("alpha", coeff(generator()))
            .Attr("beta", coeff(generator())));
  });
}

TEST_F(OpTest, MatMul) {
  Repeatedly([this]() {
    int64_t x = RandomDim();
    int64_t y = RandomDim();
    int64_t z = RandomDim();

    std::vector<int64_t> a_dims = {x, y};
    std::vector<int64_t> b_dims = {y, z};

    std::bernoulli_distribution random_bool;
    bool transpose_a = random_bool(generator());
    bool transpose_b = random_bool(generator());
    if (transpose_a) {
      std::swap(a_dims[0], a_dims[1]);
    }
    if (transpose_b) {
      std::swap(b_dims[0], b_dims[1]);
    }

    auto type = Choose<DataType>({DT_FLOAT, DT_COMPLEX64});
    return ExpectTfAndXlaOutputsAreClose(OpTestBuilder("MatMul")
                                             .RandomInput(type, a_dims)
                                             .RandomInput(type, b_dims)
                                             .Attr("T", type)
                                             .Attr("transpose_a", transpose_a)
                                             .Attr("transpose_b", transpose_b));
  });
}

TEST_F(OpTest, MatrixBandPart) {
  Repeatedly([this]() {
    auto type = Choose<DataType>(kAllXlaTypes);
    auto index_type = Choose<DataType>({DT_INT32, DT_INT64});
    auto num_lower =
        RandomBoundedTensor<int32>(index_type, -2 * kDefaultMaxDimensionSize,
                                   2 * kDefaultMaxDimensionSize, false, {});
    auto num_upper =
        RandomBoundedTensor<int32>(index_type, -2 * kDefaultMaxDimensionSize,
                                   2 * kDefaultMaxDimensionSize, false, {});
    return ExpectTfAndXlaOutputsAreClose(OpTestBuilder("MatrixBandPart")
                                             .RandomInput(type)
                                             .Input(num_lower)
                                             .Input(num_upper)
                                             .Attr("T", type)
                                             .Attr("Tindex", index_type));
  });
}

TEST_F(OpTest, MatrixDiag) {
  Repeatedly([this]() {
    auto type = Choose<DataType>({DT_INT32, DT_FLOAT, DT_COMPLEX64});
    return ExpectTfAndXlaOutputsAreClose(OpTestBuilder("MatrixDiag")
                                             .RandomInput(type, RandomDims(1))
                                             .Attr("T", type));
  });
}

TEST_F(OpTest, MatrixDiagPart) {
  if (tensorflow::tf_xla_test_use_mlir) GTEST_SKIP() << "b/201095155";
  Repeatedly([this]() {
    auto type = Choose<DataType>({DT_INT32, DT_FLOAT, DT_COMPLEX64});
    return ExpectTfAndXlaOutputsAreClose(OpTestBuilder("MatrixDiagPart")
                                             .RandomInput(type, RandomDims(2))
                                             .Attr("T", type));
  });
}

TEST_F(OpTest, MatrixDiagPartV3) {
  if (tensorflow::tf_xla_test_use_mlir) GTEST_SKIP() << "b/201095155";
  if (!tensorflow::tf_xla_test_use_mlir) GTEST_SKIP() << "b/197140886";
  Repeatedly([this]() {  // NOLINT: due to GTEST_SKIP
    auto type = Choose<DataType>(kAllXlaTypes);
    auto align = Choose<std::string>(
        {"LEFT_RIGHT", "RIGHT_LEFT", "LEFT_LEFT", "RIGHT_RIGHT"});
    auto k0 = std::uniform_int_distribution<int32>(
        -2 * kDefaultMaxDimensionSize,
        2 * kDefaultMaxDimensionSize)(generator());
    auto k1 = std::uniform_int_distribution<int32>(
        k0, 2 * kDefaultMaxDimensionSize)(generator());
    auto k = test::AsTensor<int32>({k0, k1});
    return ExpectTfAndXlaOutputsAreClose(OpTestBuilder("MatrixDiagPartV3")
                                             .RandomInput(type)
                                             .Input(k)
                                             .RandomInput(type, {})
                                             .Attr("align", align)
                                             .Attr("T", type));
  });
}

TEST_F(OpTest, MatrixSetDiag) {
  Repeatedly([this]() {
    auto type = Choose<DataType>(kAllXlaTypes);
    auto shape = RandomDims(2);
    int rank = shape.size();
    std::vector<int64_t> diagonal_shape(shape);
    diagonal_shape.pop_back();
    diagonal_shape.pop_back();
    diagonal_shape.push_back(std::min(shape[rank - 2], shape[rank - 1]));
    return ExpectTfAndXlaOutputsAreClose(OpTestBuilder("MatrixSetDiag")
                                             .RandomInput(type, shape)
                                             .RandomInput(type, diagonal_shape)
                                             .Attr("T", type));
  });
}

TEST_F(OpTest, MatrixSetDiagV2) {
  Repeatedly([this]() {
    auto type = Choose<DataType>(kAllXlaTypes);
    auto shape = RandomDims(2, kDefaultMaxRank, 1 /* non-zero dims */);
    int rank = shape.size();
    int64_t max_num_diags = shape[rank - 2] + shape[rank - 1] - 1;
    int64_t num_diags =
        std::uniform_int_distribution<int64_t>(2, max_num_diags)(generator());
    int32 k0 = std::uniform_int_distribution<int32>(
        -shape[rank - 2] + 1, shape[rank - 1] - num_diags)(generator());
    int32 k1 = k0 + num_diags - 1;
    Tensor k = test::AsTensor<int32>({k0, k1});
    int64_t max_diag_len = std::min(shape[rank - 2] + std::min(k1, 0),
                                    shape[rank - 1] + std::min(-k0, 0));
    std::vector<int64_t> diagonal_shape(shape);
    diagonal_shape.pop_back();
    diagonal_shape.pop_back();
    diagonal_shape.push_back(num_diags);
    diagonal_shape.push_back(max_diag_len);
    return ExpectTfAndXlaOutputsAreClose(OpTestBuilder("MatrixSetDiagV2")
                                             .RandomInput(type, shape)
                                             .RandomInput(type, diagonal_shape)
                                             .Input(k)
                                             .Attr("T", type));
  });
}

TEST_F(OpTest, Max) {
  Repeatedly([this]() {
    auto type = Choose<DataType>({DT_INT32, DT_FLOAT});
    std::vector<int64_t> data_dims = RandomDims();
    Tensor indices = RandomReductionIndices(data_dims.size());
    bool keep_dims = Choose<bool>({false, true});
    return ExpectTfAndXlaOutputsAreClose(OpTestBuilder("Max")
                                             .RandomInput(type, data_dims)
                                             .Input(indices)
                                             .Attr("T", type)
                                             .Attr("keep_dims", keep_dims));
  });
}

TEST_F(OpTest, Maximum) {
  Repeatedly([this]() {
    auto type = Choose<DataType>({DT_INT32, DT_FLOAT});
    auto dims = BroadcastableDims();
    return ExpectTfAndXlaOutputsAreClose(OpTestBuilder("Maximum")
                                             .RandomInput(type, dims.first)
                                             .RandomInput(type, dims.second)
                                             .Attr("T", type));
  });
}

TEST_F(OpTest, MaxPool) {
  Repeatedly([this]() {
    std::uniform_int_distribution<int> random_int(1, 5);
    std::vector<int64_t> dims = RandomDims(4, 4, 1);
    int kernel_rows =
        std::uniform_int_distribution<int>(1, dims[1])(generator());
    int kernel_cols =
        std::uniform_int_distribution<int>(1, dims[2])(generator());
    int stride_rows = random_int(generator()),
        stride_cols = random_int(generator());

    string padding = Choose<string>({"SAME", "VALID"});
    return ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("MaxPool")
            .RandomInput(DT_FLOAT, dims)
            .Attr("T", DT_FLOAT)
            .Attr("ksize", {1, kernel_rows, kernel_cols, 1})
            .Attr("strides", {1, stride_rows, stride_cols, 1})
            .Attr("padding", padding)
            .Attr("data_format", "NHWC"));
  });
  // TODO(phawkins): test NCHW format (not supported by CPU)
}

TEST_F(OpTest, MaxPool3D) {
  if (tensorflow::tf_xla_test_use_mlir) GTEST_SKIP() << "b/201095155";
  if (!tensorflow::tf_xla_test_use_mlir) GTEST_SKIP() << "b/197140886";
  Repeatedly([this]() {
    std::uniform_int_distribution<int> random_int(1, 5);
    std::vector<int64_t> dims = RandomDims(5, 5, 1);

    std::vector<int64_t> input_dims, kernel_dims, stride_dims;
    kernel_dims.push_back(1);
    stride_dims.push_back(1);
    for (int i = 0; i < 3; ++i) {
      kernel_dims.push_back(
          std::uniform_int_distribution<int>(1, dims[i])(generator()));
      input_dims.push_back(dims[i]);
      stride_dims.push_back(random_int(generator()));
    }
    kernel_dims.push_back(1);
    stride_dims.push_back(1);
    int64_t batch = dims[3];
    int64_t feature = dims[4];

    string padding = Choose<string>({"SAME", "VALID"});
    return ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("MaxPool3D")
            .RandomInput(DT_FLOAT,
                         ImageDims(FORMAT_NHWC, batch, feature, input_dims))
            .Attr("T", DT_FLOAT)
            .Attr("ksize", kernel_dims)
            .Attr("strides", stride_dims)
            .Attr("padding", padding)
            .Attr("data_format", "NDHWC"));
  });
  // TODO(phawkins): test NCHW format (not supported by CPU)
}

TEST_F(OpTest, Mean) {
  Repeatedly([this]() {
    auto type = Choose<DataType>({DT_INT32, DT_FLOAT, DT_COMPLEX64});
    // TODO(phawkins): CPU and XLA differ output for reducing across a
    // size-0 dimension (nan vs 0). For now, require size >= 1.
    std::vector<int64_t> data_dims = RandomDims(0, kDefaultMaxRank, 1);
    Tensor indices = RandomReductionIndices(data_dims.size());
    bool keep_dims = Choose<bool>({false, true});
    return ExpectTfAndXlaOutputsAreClose(OpTestBuilder("Mean")
                                             .RandomInput(type, data_dims)
                                             .Input(indices)
                                             .Attr("T", type)
                                             .Attr("keep_dims", keep_dims));
  });
}

TEST_F(OpTest, Min) {
  Repeatedly([this]() {
    auto type = Choose<DataType>({DT_INT32, DT_FLOAT});
    std::vector<int64_t> data_dims = RandomDims();
    Tensor indices = RandomReductionIndices(data_dims.size());
    bool keep_dims = Choose<bool>({false, true});
    return ExpectTfAndXlaOutputsAreClose(OpTestBuilder("Min")
                                             .RandomInput(type, data_dims)
                                             .Input(indices)
                                             .Attr("T", type)
                                             .Attr("keep_dims", keep_dims));
  });
}

TEST_F(OpTest, Minimum) {
  Repeatedly([this]() {
    auto type = Choose<DataType>({DT_INT32, DT_FLOAT});
    auto dims = BroadcastableDims();
    return ExpectTfAndXlaOutputsAreClose(OpTestBuilder("Minimum")
                                             .RandomInput(type, dims.first)
                                             .RandomInput(type, dims.second)
                                             .Attr("T", type));
  });
}

TEST_F(OpTest, Mod) {
  Repeatedly([this]() {
    auto dims = BroadcastableDims();
    return ExpectTfAndXlaOutputsAreClose(OpTestBuilder("Mod")
                                             .RandomInput(DT_INT32, dims.first)
                                             .RandomInput(DT_INT32, dims.second)
                                             .Attr("T", DT_INT32));
  });
}

TEST_F(OpTest, Mul) {
  Repeatedly([this]() {
    auto type = Choose<DataType>({DT_INT32, DT_FLOAT, DT_COMPLEX64});
    auto dims = BroadcastableDims();
    return ExpectTfAndXlaOutputsAreClose(OpTestBuilder("Mul")
                                             .RandomInput(type, dims.first)
                                             .RandomInput(type, dims.second)
                                             .Attr("T", type));
  });
}

TEST_F(OpTest, MulNoNan) {
  Repeatedly([this]() {
    auto type = Choose<DataType>({DT_INT32, DT_FLOAT, DT_COMPLEX64});
    auto dims = BroadcastableDims();
    return ExpectTfAndXlaOutputsAreClose(OpTestBuilder("Mul")
                                             .RandomInput(type, dims.first)
                                             .RandomInput(type, dims.second)
                                             .Attr("T", type));
  });
}

TEST_F(OpTest, Neg) {
  Repeatedly([this]() {
    auto type = Choose<DataType>({DT_INT32, DT_FLOAT, DT_COMPLEX64});
    return ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("Neg").RandomInput(type).Attr("T", type));
  });
}

TEST_F(OpTest, NextAfter) {
  Repeatedly([this]() {
    auto type = Choose<DataType>({DT_FLOAT});
    auto dims = RandomDims();
    return ExpectTfAndXlaOutputsAreClose(OpTestBuilder("NextAfter")
                                             .RandomInput(type, dims)
                                             .RandomInput(type, dims)
                                             .Attr("T", type));
  });
}

TEST_F(OpTest, NotEqual) {
  Repeatedly([this]() {
    auto type = Choose<DataType>({DT_INT32, DT_FLOAT, DT_COMPLEX64});
    auto dims = BroadcastableDims();
    return ExpectTfAndXlaOutputsAreClose(OpTestBuilder("NotEqual")
                                             .RandomInput(type, dims.first)
                                             .RandomInput(type, dims.second)
                                             .Attr("T", type));
  });
}

TEST_F(OpTest, OneHot) {
  Repeatedly([this]() {
    auto type = Choose<DataType>(kAllXlaTypes);

    std::vector<int64_t> dims = RandomDims();
    int num_dims = dims.size();

    int32_t depth = RandomDim();

    Tensor indices(DT_INT32, TensorShape(dims));
    std::uniform_int_distribution<int32> distribution(-depth * 2, depth * 2);
    test::FillFn<int32>(&indices, [this, &distribution](int i) -> int32 {
      return distribution(generator());
    });

    int axis = std::uniform_int_distribution<int32>(-num_dims - 5,
                                                    num_dims + 5)(generator());

    OpTestBuilder builder("OneHot");
    builder.Attr("T", type);
    builder.Attr("TI", DT_INT32);
    builder.Attr("axis", axis);
    builder.Input(indices);
    builder.Input(test::AsScalar<int32>(depth));
    builder.RandomInput(type, {});
    builder.RandomInput(type, {});
    return ExpectTfAndXlaOutputsAreClose(builder);
  });
}

TEST_F(OpTest, OnesLike) {
  if (tensorflow::tf_xla_test_use_mlir) GTEST_SKIP() << "b/201095155";
  Repeatedly([this]() {
    auto type = Choose<DataType>({DT_INT32, DT_FLOAT, DT_COMPLEX64});
    return ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("OnesLike").RandomInput(type).Attr("T", type));
  });
}

TEST_F(OpTest, Pack) {
  Repeatedly([this]() {
    auto type = Choose<DataType>(kAllXlaTypes);
    int n = std::uniform_int_distribution<int>(1, 5)(generator());

    std::vector<int64_t> dims = RandomDims();
    int num_dims = dims.size();
    int axis = std::uniform_int_distribution<int32>(-num_dims - 1,
                                                    num_dims)(generator());

    OpTestBuilder builder("Pack");
    builder.Attr("T", type);
    builder.Attr("N", n);
    builder.Attr("axis", axis);
    for (int i = 0; i < n; ++i) {
      builder.RandomInput(type, dims);
    }
    return ExpectTfAndXlaOutputsAreClose(builder);
  });
}

TEST_F(OpTest, Pad) {
  // See note about failing Kokoro tests: b/214080339#comment22
  if (tensorflow::tf_xla_test_use_mlir) GTEST_SKIP() << "b/201095155";
  Repeatedly([this]() {
    auto a = ChoosePadArguments();
    return ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("Pad")
            .RandomInput(a.input_type, a.input_shape)
            .Input(a.paddings)
            .Attr("T", a.input_type)
            .Attr("Tpaddings", a.paddings_type));
  });
}

TEST_F(OpTest, PadV2) {
  Repeatedly([this]() {
    auto a = ChoosePadArguments();
    return ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("PadV2")
            .RandomInput(a.input_type, a.input_shape)
            .Input(a.paddings)
            .Input(a.constant_values)
            .Attr("T", a.input_type)
            .Attr("Tpaddings", a.paddings_type));
  });
}

TEST_F(OpTest, Pow) {
  // TODO(phawkins): Feeding large DT_INT32 values to Pow() leads to
  // nontermination.
  Repeatedly([this]() {
    auto dims = BroadcastableDims();
    auto type = Choose<DataType>({DT_FLOAT, DT_COMPLEX64});
    return ExpectTfAndXlaOutputsAreClose(OpTestBuilder("Pow")
                                             .RandomInput(type, dims.first)
                                             .RandomInput(type, dims.second)
                                             .Attr("T", type));
  });
}

TEST_F(OpTest, Prod) {
  Repeatedly([this]() {
    auto type = Choose<DataType>({DT_INT32, DT_FLOAT, DT_COMPLEX64});
    std::vector<int64_t> data_dims = RandomDims();
    Tensor indices = RandomReductionIndices(data_dims.size());
    bool keep_dims = Choose<bool>({false, true});
    return ExpectTfAndXlaOutputsAreClose(OpTestBuilder("Prod")
                                             .RandomInput(type, data_dims)
                                             .Input(indices)
                                             .Attr("T", type)
                                             .Attr("keep_dims", keep_dims));
  });
}

TEST_F(OpTest, Qr) {
  Repeatedly([this]() {
    auto type = Choose<DataType>({DT_FLOAT, DT_COMPLEX64});
    return ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("Qr")
            .RandomInput(type, RandomDims(2, kDefaultMaxRank, 1))
            .Attr("T", type)
            .Attr("full_matrices", true));
  });
}

TEST_F(OpTest, QuantizeAndDequantizeV2) {
  Repeatedly([this]() {
    std::uniform_int_distribution<int64_t> num_bits_dist(1, 64);
    int64_t num_bits = num_bits_dist(generator());
    std::string round_mode = Choose<std::string>({"HALF_TO_EVEN", "HALF_UP"});
    auto dims = RandomDims(0, kDefaultMaxRank, 1);
    return ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("QuantizeAndDequantizeV2")
            .RandomInput(DT_FLOAT, dims)
            .RandomInput(DT_FLOAT, dims)  // unused because range_given = false
            .RandomInput(DT_FLOAT, dims)  // unused because range_given = false
            .Attr("signed_input", RandomBool())
            .Attr("num_bits", num_bits)
            .Attr("range_given", false)
            .Attr("round_mode", round_mode)
            .Attr("narrow_range", RandomBool())
            .Attr("axis", -1)
            .Attr("T", DT_FLOAT));
  });
}

TEST_F(OpTest, RandomShuffle) {
  // See b/209062491 as this test passes with --tf_xla_test_device=CPU:0
  if (tensorflow::tf_xla_test_use_mlir) GTEST_SKIP() << "b/201095155";
  if (!tensorflow::tf_xla_test_use_mlir) GTEST_SKIP() << "b/197140886";
  Repeatedly([this]() {  // NOLINT: due to GTEST_SKIP
    auto type = Choose<DataType>(kAllXlaTypes);
    return ExpectTfAndXlaOutputsAreClose(OpTestBuilder("RandomShuffle")
                                             .RandomInput(type, RandomDims(1))
                                             .Attr("seed", RandomSeed())
                                             .Attr("seed2", RandomSeed())
                                             .Attr("T", type));
  });
}

TEST_F(OpTest, RandomStandardNormal) {
  Repeatedly([this]() {
    auto shape_type = Choose<DataType>({DT_INT32, DT_INT64});
    return ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("RandomStandardNormal")
            .Input(AsIntTensor(shape_type, RandomDims()))
            .Attr("seed", RandomSeed())
            .Attr("seed2", RandomSeed())
            .Attr("T", shape_type)
            .Attr("dtype", DT_FLOAT));
  });
}

TEST_F(OpTest, RandomUniform) {
  Repeatedly([this]() {
    auto shape_type = Choose<DataType>({DT_INT32, DT_INT64});
    return ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("RandomStandardNormal")
            .Input(AsIntTensor(shape_type, RandomDims()))
            .Attr("seed", RandomSeed())
            .Attr("seed2", RandomSeed())
            .Attr("T", shape_type)
            .Attr("dtype", DT_FLOAT));
  });
}

TEST_F(OpTest, Range) {
  Repeatedly([this]() {
    auto ToScalar = [](DataType type, int x) {
      if (type == DT_INT32) return test::AsScalar<int32>(x);
      if (type == DT_INT64) return test::AsScalar<int64_t>(x);
      if (type == DT_FLOAT) return test::AsScalar<float>(x);
      if (type == DT_DOUBLE) return test::AsScalar<double>(x);
      LOG(FATAL) << "Unknown type " << DataTypeString(type);
    };
    std::uniform_int_distribution<int> distribution(-50, 50);
    DataType tidx = Choose<DataType>({DT_INT32, DT_INT64, DT_FLOAT, DT_DOUBLE});
    return ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("Range")
            .Input(ToScalar(tidx, distribution(generator())))
            .Input(ToScalar(tidx, distribution(generator())))
            .Input(ToScalar(tidx, distribution(generator())))
            .Attr("Tidx", tidx));
  });
}

TEST_F(OpTest, Rank) {
  Repeatedly([this]() {
    auto type = Choose<DataType>({DT_INT32, DT_FLOAT, DT_COMPLEX64});
    return ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("Rank").RandomInput(type).Attr("T", type));
  });
}

TEST_F(OpTest, Real) {
  Repeatedly([this]() {
    return ExpectTfAndXlaOutputsAreClose(OpTestBuilder("Real")
                                             .RandomInput(DT_COMPLEX64)
                                             .Attr("T", DT_COMPLEX64));
  });
}

TEST_F(OpTest, RealDiv) {
  Repeatedly([this]() {
    auto type = Choose<DataType>({DT_FLOAT, DT_COMPLEX64});
    auto dims = BroadcastableDims();
    return ExpectTfAndXlaOutputsAreClose(OpTestBuilder("RealDiv")
                                             .RandomInput(type, dims.first)
                                             .RandomInput(type, dims.second)
                                             .Attr("T", type));
  });
}

TEST_F(OpTest, Reciprocal) {
  Repeatedly([this]() {
    auto type = Choose<DataType>({DT_INT32, DT_FLOAT, DT_COMPLEX64});
    return ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("Reciprocal").RandomInput(type).Attr("T", type));
  });
}

TEST_F(OpTest, ReciprocalGrad) {
  Repeatedly([this]() {
    std::vector<int64_t> dims = RandomDims();
    auto type = Choose<DataType>({DT_INT32, DT_FLOAT, DT_COMPLEX64});
    return ExpectTfAndXlaOutputsAreClose(OpTestBuilder("ReciprocalGrad")
                                             .RandomInput(type, dims)
                                             .RandomInput(type, dims)
                                             .Attr("T", type));
  });
}
TEST_F(OpTest, Relu) {
  Repeatedly([this]() {
    return ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("Relu").RandomInput(DT_FLOAT).Attr("T", DT_FLOAT));
  });
}

TEST_F(OpTest, Relu6) {
  Repeatedly([this]() {
    return ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("Relu6").RandomInput(DT_FLOAT).Attr("T", DT_FLOAT));
  });
}

TEST_F(OpTest, Relu6Grad) {
  Repeatedly([this]() {
    auto dims = RandomDims(1);
    return ExpectTfAndXlaOutputsAreClose(OpTestBuilder("Relu6Grad")
                                             .RandomInput(DT_FLOAT, dims)
                                             .RandomInput(DT_FLOAT, dims)
                                             .Attr("T", DT_FLOAT));
  });
}

TEST_F(OpTest, ReluGrad) {
  Repeatedly([this]() {
    auto dims = RandomDims(1);
    return ExpectTfAndXlaOutputsAreClose(OpTestBuilder("ReluGrad")
                                             .RandomInput(DT_FLOAT, dims)
                                             .RandomInput(DT_FLOAT, dims)
                                             .Attr("T", DT_FLOAT));
  });
}

TEST_F(OpTest, Reshape) {
  Repeatedly([this]() {
    auto type = Choose<DataType>(kAllXlaTypes);
    std::vector<int64_t> dims = RandomDims();
    std::bernoulli_distribution random_bool;
    std::vector<int64_t> dims_before, dims_after;
    for (std::vector<int64_t>* out : {&dims_before, &dims_after}) {
      std::shuffle(dims.begin(), dims.end(), generator());
      for (int64_t dim : dims) {
        // Either add the dimension as a new dimension or merge it with the
        // previous dimension.
        if (out->empty() || random_bool(generator())) {
          out->push_back(dim);
        } else {
          out->back() *= dim;
        }
      }
    }
    return ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("Reshape")
            .RandomInput(type, dims_before)
            .Input(test::AsTensor<int32>(
                std::vector<int32>(dims_after.begin(), dims_after.end())))
            .Attr("T", type));
  });
}

TEST_F(OpTest, ResizeNearestNeighbor) {
  Repeatedly([this]() {
    auto type = Choose<DataType>({DT_FLOAT, DT_INT32, DT_INT64});
    return ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("ResizeNearestNeighbor")
            .RandomInput(type, RandomDims(4, 4, 1))
            .Input(AsIntTensor(DT_INT32, RandomDims(2, kDefaultMaxRank, 1)))
            .Attr("align_corners", RandomBool())
            .Attr("half_pixel_centers", RandomBool())
            .Attr("T", type));
  });
}

TEST_F(OpTest, ResizeBilinear) {
  Repeatedly([this]() {
    std::vector<int64_t> in_dims = RandomDims(4, 4);
    std::vector<int64_t> out_dims = RandomDims(2, 2);

    return ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("ResizeBilinear")
            .RandomInput(DT_FLOAT, in_dims)
            .Input(test::AsTensor<int32>(
                std::vector<int32>(out_dims.begin(), out_dims.end())))
            .Attr("T", DT_FLOAT)
            .Attr("align_corners", true));
  });
}

TEST_F(OpTest, ResizeBilinearGrad) {
  Repeatedly([this]() {
    std::vector<int64_t> in_dims = RandomDims(4, 4);
    std::vector<int64_t> out_dims = RandomDims(2, 2);

    return ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("ResizeBilinearGrad")
            .RandomInput(DT_FLOAT, in_dims)
            .RandomInput(DT_FLOAT,
                         {in_dims[0], out_dims[0], out_dims[1], in_dims[3]})
            .Attr("T", DT_FLOAT)
            .Attr("align_corners", true));
  });
}

TEST_F(OpTest, Reverse) {
  if (tensorflow::tf_xla_test_use_mlir) GTEST_SKIP() << "b/201095155";
  Repeatedly([this]() {
    std::vector<int64_t> dims = RandomDims(1);
    auto type = Choose<DataType>(kAllXlaTypes);
    int64_t rank = dims.size();
    return ExpectTfAndXlaOutputsAreClose(OpTestBuilder("Reverse")
                                             .RandomInput(type, dims)
                                             .RandomInput(DT_BOOL, {rank})
                                             .Attr("T", type));
  });
}

TEST_F(OpTest, ReverseSequence) {
  if (tensorflow::tf_xla_test_use_mlir) GTEST_SKIP() << "b/201095155";
  if (!tensorflow::tf_xla_test_use_mlir) GTEST_SKIP() << "b/197140886";
  Repeatedly([this]() {
    std::vector<int64_t> dims = RandomDims(/*min_rank=*/2);
    auto type = Choose<DataType>(kAllXlaTypes);
    int64_t rank = dims.size();

    // Choose random batch and sequence dimensions.
    std::vector<int> shuffled_dim_ids(rank);
    absl::c_iota(shuffled_dim_ids, 0);
    absl::c_shuffle(shuffled_dim_ids, generator());
    shuffled_dim_ids.resize(2);
    int batch_dim = shuffled_dim_ids[0];
    int seq_dim = shuffled_dim_ids[1];

    int batch_size = dims[batch_dim];
    int max_seq_len = dims[seq_dim];
    std::vector<int32> seq_lens(batch_size);
    std::uniform_int_distribution<int32> d(0, max_seq_len);
    absl::c_generate(seq_lens, [&]() { return d(generator()); });

    return ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("ReverseSequence")
            .RandomInput(type, dims)
            .Input(test::AsTensor<int32>(seq_lens))
            .Attr("seq_dim", seq_dim)
            .Attr("batch_dim", batch_dim)
            .Attr("T", type)
            .Attr("Tlen", DT_INT32));
  });
}

TEST_F(OpTest, ReverseV2) {
  Repeatedly([this]() {
    auto type = Choose<DataType>(kAllXlaTypes);
    std::vector<int64_t> data_dims = RandomDims();
    Tensor indices = RandomReductionIndices(data_dims.size());
    return ExpectTfAndXlaOutputsAreClose(OpTestBuilder("ReverseV2")
                                             .RandomInput(type, data_dims)
                                             .Input(indices)
                                             .Attr("T", type));
  });
}

TEST_F(OpTest, RightShift) {
  Repeatedly([this]() {
    bool is64 = RandomBool();
    auto dims = RandomDims();
    auto type = is64 ? DT_INT64 : DT_INT32;
    int max_shift = is64 ? 63 : 31;
    auto y = RandomBoundedTensor(type, 0, max_shift, false, dims);
    return ExpectTfAndXlaOutputsAreClose(OpTestBuilder("RightShift")
                                             .RandomInput(type, dims)
                                             .Input(y)
                                             .Attr("T", type));
  });
}

TEST_F(OpTest, Rint) {
  Repeatedly([this]() {
    return ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("Rint").RandomInput(DT_FLOAT).Attr("T", DT_FLOAT));
  });
}

TEST_F(OpTest, Roll) {
  Repeatedly([this]() {
    auto input_type = Choose<DataType>(kAllXlaTypes);
    auto axis_type = Choose<DataType>({DT_INT32, DT_INT64});
    // TODO(b/201095155,b/197140886): shift_type = DT_INT64 doesn't work.
    auto shift_type = DT_INT32;
    auto input_shape = RandomDims(1);
    int rank = input_shape.size();
    auto axis_shape = RandomDims(1, 1, 1, rank + 1);
    auto axis = RandomBoundedTensor(axis_type, 0, rank - 1, true, axis_shape);
    auto shift = RandomTensor(shift_type, false, axis_shape);
    return ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("Roll")
            .RandomInput(input_type, input_shape)
            .Input(shift)
            .Input(axis)
            .Attr("T", input_type)
            .Attr("Taxis", axis_type)
            .Attr("Tshift", shift_type));
  });
}

TEST_F(OpTest, Round) {
  Repeatedly([this]() {
    return ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("Round").RandomInput(DT_FLOAT).Attr("T", DT_FLOAT));
  });
}

TEST_F(OpTest, Rsqrt) {
  Repeatedly([this]() {
    auto type = Choose<DataType>({DT_FLOAT, DT_COMPLEX64});
    return ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("Rsqrt").RandomInput(type).Attr("T", type));
  });
}

TEST_F(OpTest, RsqrtGrad) {
  Repeatedly([this]() {
    auto dims = RandomDims();
    auto type = Choose<DataType>({DT_FLOAT, DT_COMPLEX64});
    return ExpectTfAndXlaOutputsAreClose(OpTestBuilder("RsqrtGrad")
                                             .RandomInput(type, dims)
                                             .RandomInput(type, dims)
                                             .Attr("T", type));
  });
}

TEST_F(OpTest, Select) {
  Repeatedly([this]() {
    auto type = Choose<DataType>(kAllXlaTypes);
    auto shape = RandomDims();
    return ExpectTfAndXlaOutputsAreClose(OpTestBuilder("Select")
                                             .RandomInput(DT_BOOL, shape)
                                             .RandomInput(type, shape)
                                             .RandomInput(type, shape)
                                             .Attr("T", type));
  });
}

TEST_F(OpTest, SelectV2) {
  Repeatedly([this]() {
    auto type = Choose<DataType>(kAllXlaTypes);
    auto shape = RandomDims();
    return ExpectTfAndXlaOutputsAreClose(OpTestBuilder("SelectV2")
                                             .RandomInput(DT_BOOL, shape)
                                             .RandomInput(type, shape)
                                             .RandomInput(type, shape)
                                             .Attr("T", type));
  });
}

TEST_F(OpTest, Shape) {
  Repeatedly([this]() {
    auto type = Choose<DataType>(kAllXlaTypes);
    return ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("Shape").RandomInput(type).Attr("T", type));
  });
}

TEST_F(OpTest, ShapeN) {
  Repeatedly([this]() {
    auto type = Choose<DataType>(kAllXlaTypes);
    int n = std::uniform_int_distribution<int>(1, 5)(generator());
    OpTestBuilder builder("ShapeN");
    builder.Attr("T", type);
    builder.Attr("N", n);
    for (int i = 0; i < n; ++i) {
      builder.RandomInput(type);
    }
    return ExpectTfAndXlaOutputsAreClose(builder);
  });
}

TEST_F(OpTest, Sigmoid) {
  Repeatedly([this]() {
    auto type = Choose<DataType>({DT_FLOAT, DT_COMPLEX64});
    return ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("Sigmoid").RandomInput(type).Attr("T", type));
  });
}

TEST_F(OpTest, SigmoidGrad) {
  Repeatedly([this]() {
    auto dims = RandomDims();
    auto type = Choose<DataType>({DT_FLOAT, DT_COMPLEX64});
    return ExpectTfAndXlaOutputsAreClose(OpTestBuilder("SigmoidGrad")
                                             .RandomInput(type, dims)
                                             .RandomInput(type, dims)
                                             .Attr("T", type));
  });
}

TEST_F(OpTest, Sign) {
  Repeatedly([this]() {
    auto type = Choose<DataType>({DT_INT32, DT_FLOAT, DT_COMPLEX64});
    return ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("Sign").RandomInput(type).Attr("T", type));
  });
}

TEST_F(OpTest, Sin) {
  Repeatedly([this]() {
    auto type = Choose<DataType>({DT_FLOAT, DT_COMPLEX64});
    return ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("Sin").RandomInput(type).Attr("T", type));
  });
}

TEST_F(OpTest, Sinh) {
  Repeatedly([this]() {
    auto type = Choose<DataType>({DT_FLOAT, DT_COMPLEX64});
    return ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("Sinh").RandomInput(type).Attr("T", type));
  });
}

TEST_F(OpTest, Size) {
  Repeatedly([this]() {
    auto type = Choose<DataType>(kAllXlaTypes);
    return ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("Size").RandomInput(type).Attr("T", type));
  });
}

TEST_F(OpTest, Slice) {
  Repeatedly([this]() {
    SliceArguments a = ChooseSliceArguments(true);
    std::vector<int32> size;
    size.insert(size.end(), a.size.begin(), a.size.end());
    return ExpectTfAndXlaOutputsAreClose(OpTestBuilder("Slice")
                                             .RandomInput(a.type, a.shape)
                                             .Input(a.indices)
                                             .Input(test::AsTensor<int32>(size))
                                             .Attr("T", a.type)
                                             .Attr("Index", a.indices_type));
  });
}

TEST_F(OpTest, Softmax) {
  Repeatedly([this]() {
    return ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("Softmax")
            .RandomInput(DT_FLOAT, RandomDims(2, 2))
            .Attr("T", DT_FLOAT));
  });
}

TEST_F(OpTest, SoftmaxCrossEntropyWithLogits) {
  Repeatedly([this]() {
    std::vector<int64_t> dims = RandomDims(2, 2, 1);
    return ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("SoftmaxCrossEntropyWithLogits")
            .RandomInput(DT_FLOAT, dims)
            .RandomInput(DT_FLOAT, dims)
            .Attr("T", DT_FLOAT));
  });
}

TEST_F(OpTest, Softplus) {
  Repeatedly([this]() {
    return ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("Softplus").RandomInput(DT_FLOAT).Attr("T", DT_FLOAT));
  });
}

TEST_F(OpTest, SoftplusGrad) {
  Repeatedly([this]() {
    std::vector<int64_t> dims = RandomDims();
    return ExpectTfAndXlaOutputsAreClose(OpTestBuilder("SoftplusGrad")
                                             .RandomInput(DT_FLOAT, dims)
                                             .RandomInput(DT_FLOAT, dims)
                                             .Attr("T", DT_FLOAT));
  });
}

TEST_F(OpTest, Softsign) {
  Repeatedly([this]() {
    return ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("Softsign").RandomInput(DT_FLOAT).Attr("T", DT_FLOAT));
  });
}

TEST_F(OpTest, SoftsignGrad) {
  Repeatedly([this]() {
    std::vector<int64_t> dims = RandomDims();
    return ExpectTfAndXlaOutputsAreClose(OpTestBuilder("SoftsignGrad")
                                             .RandomInput(DT_FLOAT, dims)
                                             .RandomInput(DT_FLOAT, dims)
                                             .Attr("T", DT_FLOAT));
  });
}

TEST_F(OpTest, SpaceToBatch) {
  Repeatedly([this]() {
    std::vector<int64_t> block_dims = RandomDims(4, 4, 0, 5);
    const int num_block_dims = 2;
    int64_t block_size = RandomDim(2, 5);

    std::vector<int64_t> input_dims(1 + num_block_dims + 1);
    input_dims[0] = RandomDim();
    for (int i = 0; i < num_block_dims; ++i) {
      input_dims[1 + i] = block_dims[i] * block_size;
    }
    input_dims[1 + num_block_dims] = RandomDim();

    std::vector<int64_t> padding_vals;
    std::uniform_int_distribution<int> distribution(0, 7);
    for (int i = 0; i < num_block_dims; ++i) {
      int64_t pad_before;
      int64_t pad_after;
      do {
        pad_before = distribution(generator());
        pad_after = distribution(generator());
      } while (pad_before + pad_after > input_dims[1 + i]);
      input_dims[1 + i] -= pad_before + pad_after;
      padding_vals.push_back(pad_before);
      padding_vals.push_back(pad_after);
    }
    Tensor paddings;
    CHECK(paddings.CopyFrom(AsIntTensor(DT_INT32, padding_vals),
                            TensorShape({num_block_dims, 2})));

    auto type = Choose<DataType>(kAllXlaTypes);
    return ExpectTfAndXlaOutputsAreClose(OpTestBuilder("SpaceToBatch")
                                             .RandomInput(type, input_dims)
                                             .Input(paddings)
                                             .Attr("T", type)
                                             .Attr("block_size", block_size));
  });
}

TEST_F(OpTest, SpaceToBatchND) {
  Repeatedly([this]() {
    std::vector<int64_t> block_dims = RandomDims(1, 3, 0, 5);
    int num_block_dims = block_dims.size();
    std::vector<int64_t> remaining_dims = RandomDims(0, 3);
    std::vector<int64_t> block_multipliers =
        RandomDims(block_dims.size(), block_dims.size(), 0, 4);

    std::vector<int64_t> input_dims(1 + num_block_dims + remaining_dims.size());
    input_dims[0] = RandomDim();
    for (int i = 0; i < num_block_dims; ++i) {
      input_dims[1 + i] = block_dims[i] * block_multipliers[i];
    }
    std::copy(remaining_dims.begin(), remaining_dims.end(),
              input_dims.begin() + 1 + num_block_dims);

    std::vector<int64_t> padding_vals;
    std::uniform_int_distribution<int> distribution(0, 7);
    for (int i = 0; i < num_block_dims; ++i) {
      int64_t pad_before;
      int64_t pad_after;
      do {
        pad_before = distribution(generator());
        pad_after = distribution(generator());
      } while (pad_before + pad_after > input_dims[1 + i]);
      input_dims[1 + i] -= pad_before + pad_after;
      padding_vals.push_back(pad_before);
      padding_vals.push_back(pad_after);
    }
    Tensor paddings;
    CHECK(paddings.CopyFrom(AsIntTensor(DT_INT32, padding_vals),
                            TensorShape({num_block_dims, 2})));

    auto type = Choose<DataType>(kAllXlaTypes);
    return ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("SpaceToBatchND")
            .RandomInput(type, input_dims)
            .Input(test::AsTensor<int32>(
                std::vector<int32>(block_dims.begin(), block_dims.end())))
            .Input(paddings)
            .Attr("T", type));
  });
}

TEST_F(OpTest, SpaceToDepth) {
  Repeatedly([this]() {
    int64_t block = RandomDim(2, 5);
    std::vector<int64_t> input_dims = RandomDims(4, 4);
    // Round spatial dimensions up to a multiple of the block size
    input_dims[1] = (input_dims[1] + (block - 1)) / block * block;
    input_dims[2] = (input_dims[2] + (block - 1)) / block * block;
    return ExpectTfAndXlaOutputsAreClose(OpTestBuilder("SpaceToDepth")
                                             .RandomInput(DT_FLOAT, input_dims)
                                             .Attr("T", DT_FLOAT)
                                             .Attr("block_size", block));
  });
}

TEST_F(OpTest, SparseMatMul) {
  if (tensorflow::tf_xla_test_use_mlir) GTEST_SKIP() << "b/201095155";
  if (!tensorflow::tf_xla_test_use_mlir) GTEST_SKIP() << "b/197140886";
  Repeatedly([this]() {
    int64_t x = RandomDim();
    int64_t y = RandomDim();
    int64_t z = RandomDim();

    std::vector<int64_t> a_dims = {x, y};
    std::vector<int64_t> b_dims = {y, z};

    std::bernoulli_distribution random_bool;
    bool transpose_a = random_bool(generator());
    bool transpose_b = random_bool(generator());
    if (transpose_a) {
      std::swap(a_dims[0], a_dims[1]);
    }
    if (transpose_b) {
      std::swap(b_dims[0], b_dims[1]);
    }

    return ExpectTfAndXlaOutputsAreClose(OpTestBuilder("SparseMatMul")
                                             .RandomInput(DT_FLOAT, a_dims)
                                             .RandomInput(DT_FLOAT, b_dims)
                                             .Attr("Ta", DT_FLOAT)
                                             .Attr("Tb", DT_FLOAT)
                                             .Attr("transpose_a", transpose_a)
                                             .Attr("transpose_b", transpose_b));
  });
}

TEST_F(OpTest, SparseSoftmaxCrossEntropyWithLogits) {
  Repeatedly([this]() {
    std::vector<int64_t> dims = RandomDims(2, 2, 1);
    int64_t batch_size = dims[0];
    int64_t num_classes = dims[1];

    std::vector<int32> indices(batch_size);
    for (int64_t i = 0; i < batch_size; ++i) {
      indices[i] =
          std::uniform_int_distribution<int32>(0, num_classes - 1)(generator());
    }

    return ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("SparseSoftmaxCrossEntropyWithLogits")
            .RandomInput(DT_FLOAT, dims)
            .Input(test::AsTensor<int32>(indices))
            .Attr("T", DT_FLOAT)
            .Attr("Tlabels", DT_INT32));
  });
}

TEST_F(OpTest, Split) {
  // See b/214080339#comment27 as this test causes Kokoro to crash.
  if (tensorflow::tf_xla_test_use_mlir) GTEST_SKIP() << "b/201095155";
  if (!tensorflow::tf_xla_test_use_mlir) GTEST_SKIP() << "b/197140886";
  Repeatedly([this]() {
    auto type = Choose<DataType>(kAllXlaTypes);
    std::vector<int64_t> dims = RandomDims(1);
    std::uniform_int_distribution<int> ud;
    int32_t dim = std::uniform_int_distribution<int32>(
        -static_cast<int32>(dims.size()),
        static_cast<int32>(dims.size()) - 1)(generator());
    int n = std::uniform_int_distribution<int>(1, 5)(generator());
    // Ensure 'dim' is evenly divisible by 'n'.
    dims[dim] /= n;
    dims[dim] *= n;
    return ExpectTfAndXlaOutputsAreClose(OpTestBuilder("Split")
                                             .Input(test::AsScalar<int32>(dim))
                                             .RandomInput(type, dims)
                                             .Attr("T", type)
                                             .Attr("num_split", n));
  });
}

TEST_F(OpTest, SplitV) {
  // Likely this only fails when dim is negative. Try type = DT_FLOAT first.
  if (tensorflow::tf_xla_test_use_mlir) GTEST_SKIP() << "b/201095155";
  if (!tensorflow::tf_xla_test_use_mlir) GTEST_SKIP() << "b/197140886";
  Repeatedly([this]() {  // NOLINT: due to GTEST_SKIP
    auto type = Choose<DataType>(kAllXlaTypes);
    std::vector<int64_t> dims = RandomDims(1, kDefaultMaxRank, 1);
    int32_t dim = std::uniform_int_distribution<int32>(
        -static_cast<int32>(dims.size()),
        static_cast<int32>(dims.size()) - 1)(generator());
    int n = std::uniform_int_distribution<int>(
        1, std::min(5, static_cast<int>(dims[dim])))(generator());
    std::vector<int32> size_splits(n);
    for (int i = 0; i < n - 1; ++i) {
      size_splits.push_back(dims[dim] / n);
    }
    size_splits.push_back(dims[dim] - (n - 1) * (dims[dim] / n));
    return ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("SplitV")
            .RandomInput(type, dims)
            .Input(test::AsTensor<int32>(size_splits))
            .Input(test::AsScalar<int32>(dim))
            .Attr("T", type)
            .Attr("num_split", n)
            .Attr("Tlen", DT_INT32));
  });
}

TEST_F(OpTest, Sqrt) {
  Repeatedly([this]() {
    auto type = Choose<DataType>({DT_FLOAT, DT_COMPLEX64});
    return ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("Sqrt").RandomInput(type).Attr("T", type));
  });
}

TEST_F(OpTest, StopGradient) {
  Repeatedly([this]() {
    auto type = Choose<DataType>(kAllXlaTypes);
    return ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("StopGradient").RandomInput(type).Attr("T", type));
  });
}

TEST_F(OpTest, SqrtGrad) {
  Repeatedly([this]() {
    auto dims = RandomDims();
    auto type = Choose<DataType>({DT_FLOAT, DT_COMPLEX64});
    return ExpectTfAndXlaOutputsAreClose(OpTestBuilder("SqrtGrad")
                                             .RandomInput(type, dims)
                                             .RandomInput(type, dims)
                                             .Attr("T", type));
  });
}

TEST_F(OpTest, SquaredDifference) {
  Repeatedly([this]() {
    auto dims = BroadcastableDims();
    return ExpectTfAndXlaOutputsAreClose(OpTestBuilder("SquaredDifference")
                                             .RandomInput(DT_FLOAT, dims.first)
                                             .RandomInput(DT_FLOAT, dims.second)
                                             .Attr("T", DT_FLOAT));
  });
}

TEST_F(OpTest, Square) {
  Repeatedly([this]() {
    auto type = Choose<DataType>({DT_INT32, DT_FLOAT, DT_COMPLEX64});
    return ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("Square").RandomInput(type).Attr("T", type));
  });
}

TEST_F(OpTest, Squeeze) {
  Repeatedly([this]() {
    auto type = Choose<DataType>(kAllXlaTypes);
    std::vector<int64_t> t_dims = RandomDims(0, kDefaultMaxRank, 0, 5);
    std::bernoulli_distribution random_bool;
    std::vector<int> squeeze_dims;
    for (int i = 0; i < t_dims.size(); ++i) {
      if (t_dims[i] == 1 && random_bool(generator())) {
        squeeze_dims.push_back(i);
      }
    }
    return ExpectTfAndXlaOutputsAreClose(OpTestBuilder("Squeeze")
                                             .RandomInput(type, t_dims)
                                             .Attr("squeeze_dims", squeeze_dims)
                                             .Attr("T", type));
  });
}

TEST_F(OpTest, Sub) {
  Repeatedly([this]() {
    auto type = Choose<DataType>({DT_INT32, DT_FLOAT, DT_COMPLEX64});
    auto dims = BroadcastableDims();
    return ExpectTfAndXlaOutputsAreClose(OpTestBuilder("Sub")
                                             .RandomInput(type, dims.first)
                                             .RandomInput(type, dims.second)
                                             .Attr("T", type));
  });
}

TEST_F(OpTest, Sum) {
  Repeatedly([this]() {
    auto type = Choose<DataType>({DT_INT32, DT_FLOAT, DT_COMPLEX64});
    std::vector<int64_t> data_dims = RandomDims();
    Tensor indices = RandomReductionIndices(data_dims.size());
    bool keep_dims = Choose<bool>({false, true});
    return ExpectTfAndXlaOutputsAreClose(OpTestBuilder("Sum")
                                             .RandomInput(type, data_dims)
                                             .Input(indices)
                                             .Attr("T", type)
                                             .Attr("keep_dims", keep_dims));
  });
}

TEST_F(OpTest, StridedSlice) {
  if (tensorflow::tf_xla_test_use_mlir) GTEST_SKIP() << "b/201095155";
  Repeatedly([this]() {
    auto type = Choose<DataType>(kAllXlaTypes);
    std::vector<int64_t> data_dims = RandomDims();
    std::vector<int32> begin(data_dims.size()), end(data_dims.size());
    std::vector<int32> strides(data_dims.size());
    for (int i = 0; i < data_dims.size(); ++i) {
      begin[i] = std::uniform_int_distribution<int32>(
          -2 * data_dims[i], 2 * data_dims[i])(generator());
      end[i] = std::uniform_int_distribution<int32>(
          -2 * data_dims[i], 2 * data_dims[i])(generator());
      // TODO(b/31360685): support strides other than 1 or -1
      strides[i] = std::bernoulli_distribution()(generator()) ? 1 : -1;
    }
    int64_t max_bitmask = (1LL << data_dims.size()) - 1;
    std::uniform_int_distribution<int64_t> bitmask_distribution(0, max_bitmask);
    int64_t begin_mask = bitmask_distribution(generator());
    int64_t end_mask = bitmask_distribution(generator());

    // Create a ellipsis bitmask with at most one 1 bit set.
    int64_t ellipsis_mask = 0;
    if (!data_dims.empty() && std::bernoulli_distribution()(generator())) {
      int ellipsis_pos = std::uniform_int_distribution<int>(
          0, data_dims.size() - 1)(generator());
      ellipsis_mask = 1LL << ellipsis_pos;
    }

    int64_t new_axis_mask = bitmask_distribution(generator());
    int64_t shrink_axis_mask = bitmask_distribution(generator());
    return ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("StridedSlice")
            .RandomInput(type, data_dims)
            .Input(test::AsTensor<int32>(begin))
            .Input(test::AsTensor<int32>(end))
            .Input(test::AsTensor<int32>(strides))
            .Attr("T", type)
            .Attr("Index", DT_INT32)
            .Attr("begin_mask", begin_mask)
            .Attr("end_mask", end_mask)
            .Attr("ellipsis_mask", ellipsis_mask)
            .Attr("new_axis_mask", new_axis_mask)
            .Attr("shrink_axis_mask", shrink_axis_mask));
  });
}

TEST_F(OpTest, StridedSliceGrad) {
  if (tensorflow::tf_xla_test_use_mlir) GTEST_SKIP() << "b/201095155";
  if (!tensorflow::tf_xla_test_use_mlir) GTEST_SKIP() << "b/197140886";
  Repeatedly([this]() {
    auto type = Choose<DataType>(kAllXlaTypes);

    // Dimensions of the forward input.
    std::vector<int64_t> dims = RandomDims();

    std::vector<int64_t> begin(dims.size()), end(dims.size());
    std::vector<int64_t> strides(dims.size());
    for (int i = 0; i < dims.size(); ++i) {
      begin[i] = std::uniform_int_distribution<int64_t>(
          -2 * dims[i], 2 * dims[i])(generator());
      end[i] = std::uniform_int_distribution<int64_t>(-2 * dims[i],
                                                      2 * dims[i])(generator());
      strides[i] = std::uniform_int_distribution<int64_t>(
          -2 * dims[i], 2 * dims[i])(generator());
    }
    int64_t max_bitmask = (1LL << dims.size()) - 1;
    std::uniform_int_distribution<int64_t> bitmask_distribution(0, max_bitmask);
    int64_t begin_mask = bitmask_distribution(generator());
    int64_t end_mask = bitmask_distribution(generator());

    // Create a ellipsis bitmask with at most one 1 bit set.
    int64_t ellipsis_mask = 0;
    if (!dims.empty() && std::bernoulli_distribution()(generator())) {
      int ellipsis_pos =
          std::uniform_int_distribution<int>(0, dims.size() - 1)(generator());
      ellipsis_mask = 1LL << ellipsis_pos;
    }

    int64_t new_axis_mask = bitmask_distribution(generator());
    int64_t shrink_axis_mask = bitmask_distribution(generator());

    // TODO(phawkins): use shape inference for the forward op to compute the
    // gradient shape for the backward op. At present, there is a low
    // probability of the golden op succeeding.
    return ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("StridedSliceGrad")
            .Input(test::AsTensor<int64_t>(dims))
            .Input(test::AsTensor<int64_t>(begin))
            .Input(test::AsTensor<int64_t>(end))
            .Input(test::AsTensor<int64_t>(strides))
            .RandomInput(type, RandomDims(1))
            .Attr("T", type)
            .Attr("Index", DT_INT64)
            .Attr("begin_mask", begin_mask)
            .Attr("end_mask", end_mask)
            .Attr("ellipsis_mask", ellipsis_mask)
            .Attr("new_axis_mask", new_axis_mask)
            .Attr("shrink_axis_mask", shrink_axis_mask));
  });
}

TEST_F(OpTest, Tan) {
  Repeatedly([this]() {
    auto type = Choose<DataType>({DT_FLOAT, DT_COMPLEX64});
    return ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("Tan").RandomInput(type).Attr("T", type));
  });
}

TEST_F(OpTest, Tanh) {
  Repeatedly([this]() {
    auto type = Choose<DataType>({DT_FLOAT, DT_COMPLEX64});
    return ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("Tanh").RandomInput(type).Attr("T", type));
  });
}

TEST_F(OpTest, TanhGrad) {
  Repeatedly([this]() {
    auto dims = RandomDims();
    auto type = Choose<DataType>({DT_FLOAT, DT_COMPLEX64});
    return ExpectTfAndXlaOutputsAreClose(OpTestBuilder("TanhGrad")
                                             .RandomInput(type, dims)
                                             .RandomInput(type, dims)
                                             .Attr("T", type));
  });
}

TEST_F(OpTest, TensorScatterUpdate) {
  if (tensorflow::tf_xla_test_use_mlir) GTEST_SKIP() << "b/201095155";
  if (!tensorflow::tf_xla_test_use_mlir) GTEST_SKIP() << "b/197140886";
  Repeatedly([this]() {  // NOLINT: due to GTEST_SKIP
    auto a = ChooseScatterArguments();
    return ExpectTfAndXlaOutputsAreClose(OpTestBuilder("TensorScatterUpdate")
                                             .RandomInput(a.type, a.shape)
                                             .Input(a.indices)
                                             .Input(a.updates)
                                             .Attr("T", a.type)
                                             .Attr("Tindices", a.indices_type));
  });
}

TEST_F(OpTest, Tile) {
  Repeatedly([this]() {
    auto type = Choose<DataType>(kAllXlaTypes);
    std::vector<int64_t> t_dims = RandomDims(1);
    std::vector<int32> multiples(t_dims.size());
    for (int i = 0; i < t_dims.size(); ++i) {
      multiples[i] = std::uniform_int_distribution<int>(1, 3)(generator());
    }
    return ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("Tile")
            .RandomInput(type, t_dims)
            .Input(test::AsTensor<int32>(multiples))
            .Attr("T", type));
  });
}

TEST_F(OpTest, TopKV2) {
  if (tensorflow::tf_xla_test_use_mlir) GTEST_SKIP() << "b/201095155";
  if (!tensorflow::tf_xla_test_use_mlir) GTEST_SKIP() << "b/197140886";
  Repeatedly([this]() {  // NOLINT: due to GTEST_SKIP
    auto type = Choose<DataType>({DT_INT32, DT_FLOAT, DT_INT64});
    auto shape = RandomDims(1);
    int32 k = std::uniform_int_distribution<int32>(1, shape[0])(generator());
    return ExpectTfAndXlaOutputsAreClose(OpTestBuilder("TopKV2")
                                             .RandomInput(type, shape)
                                             .Input(test::AsScalar<int32>(k))
                                             .Attr("sorted", RandomBool())
                                             .Attr("T", type));
  });
}

TEST_F(OpTest, Transpose) {
  Repeatedly([this]() {
    auto type = Choose<DataType>(kAllXlaTypes);
    std::vector<int64_t> data_dims = RandomDims();
    std::vector<int32> perm(data_dims.size());
    std::iota(perm.begin(), perm.end(), 0);
    std::shuffle(perm.begin(), perm.end(), generator());
    return ExpectTfAndXlaOutputsAreClose(OpTestBuilder("Transpose")
                                             .RandomInput(type, data_dims)
                                             .Input(test::AsTensor<int32>(perm))
                                             .Attr("T", type));
  });
}

TEST_F(OpTest, TruncateDiv) {
  if (tensorflow::tf_xla_test_use_mlir) GTEST_SKIP() << "b/201095155";
  if (!tensorflow::tf_xla_test_use_mlir) GTEST_SKIP() << "b/197140886";
  Repeatedly([this]() {
    DataType type = DT_INT32;
    auto dims = BroadcastableDims();
    return ExpectTfAndXlaOutputsAreClose(OpTestBuilder("TruncateDiv")
                                             .RandomInput(type, dims.first)
                                             .RandomInput(type, dims.second)
                                             .Attr("T", type));
  });
}

TEST_F(OpTest, TruncateMod) {
  Repeatedly([this]() {
    auto type = Choose<DataType>({DT_INT32, DT_FLOAT});
    auto dims = BroadcastableDims();
    return ExpectTfAndXlaOutputsAreClose(OpTestBuilder("TruncateMod")
                                             .RandomInput(type, dims.first)
                                             .RandomInput(type, dims.second)
                                             .Attr("T", type));
  });
}

TEST_F(OpTest, Unpack) {
  Repeatedly([this]() {
    auto type = Choose<DataType>(kAllXlaTypes);
    auto shape = RandomDims(1);
    int axis =
        std::uniform_int_distribution<int>(0, shape.size() - 1)(generator());
    return ExpectTfAndXlaOutputsAreClose(OpTestBuilder("Unpack")
                                             .RandomInput(type, shape)
                                             .Attr("axis", axis)
                                             .Attr("T", type)
                                             .Attr("num", shape[axis]));
  });
}

TEST_F(OpTest, Xdivy) {
  Repeatedly([this]() {
    auto type = Choose<DataType>({DT_FLOAT, DT_COMPLEX64});
    auto dims = BroadcastableDims();
    return ExpectTfAndXlaOutputsAreClose(OpTestBuilder("Xdivy")
                                             .RandomInput(type, dims.first)
                                             .RandomInput(type, dims.second)
                                             .Attr("T", type));
  });
}

TEST_F(OpTest, XlaDot) {
  Repeatedly([this]() {
    const XlaDotArguments& a = ChooseXlaDotArguments();
    return ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("XlaDot")
            .RandomInput(a.dtype, a.lhs_dims)
            .RandomInput(a.dtype, a.rhs_dims)
            .Attr("dimension_numbers", a.dnums_encoded)
            .Attr("precision_config", a.precision_config_encoded)
            .Attr("T", a.dtype));
  });
}

TEST_F(OpTest, XlaDotV2) {
  Repeatedly([this]() {
    const XlaDotArguments& a = ChooseXlaDotArguments();
    return ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("XlaDotV2")
            .RandomInput(a.dtype, a.lhs_dims)
            .RandomInput(a.dtype, a.rhs_dims)
            .Attr("dimension_numbers", a.dnums_encoded)
            .Attr("precision_config", a.precision_config_encoded)
            .Attr("LhsT", a.dtype)
            .Attr("RhsT", a.dtype)
            .Attr("preferred_element_type", a.dtype));
  });
}

TEST_F(OpTest, XlaDynamicUpdateSlice) {
  Repeatedly([this]() {
    SliceArguments a = ChooseSliceArguments(false);
    return ExpectTfAndXlaOutputsAreClose(OpTestBuilder("XlaDynamicUpdateSlice")
                                             .RandomInput(a.type, a.shape)
                                             .RandomInput(a.type, a.size)
                                             .Input(a.indices)
                                             .Attr("T", a.type)
                                             .Attr("Tindices", a.indices_type));
  });
}

TEST_F(OpTest, XlaEinsum) {
  Repeatedly([this]() {
    const EinsumArguments a = ChooseEinsumArguments();
    return ExpectTfAndXlaOutputsAreClose(OpTestBuilder("XlaEinsum")
                                             .RandomInput(a.type, a.lhs_dims)
                                             .RandomInput(a.type, a.rhs_dims)
                                             .Attr("equation", a.equation)
                                             .Attr("T", a.type));
  });
}

TEST_F(OpTest, XlaSort) {
  Repeatedly([this]() {
    auto type = Choose<DataType>(kAllXlaTypes);
    return ExpectTfAndXlaOutputsAreClose(OpTestBuilder("XlaSort")
                                             .RandomInput(type, RandomDims())
                                             .Attr("T", type));
  });
}

TEST_F(OpTest, Xlog1py) {
  Repeatedly([this]() {
    auto type = Choose<DataType>({DT_FLOAT, DT_COMPLEX64});
    auto dims = BroadcastableDims();
    return ExpectTfAndXlaOutputsAreClose(OpTestBuilder("Xlog1py")
                                             .RandomInput(type, dims.first)
                                             .RandomInput(type, dims.second)
                                             .Attr("T", type));
  });
}

TEST_F(OpTest, Xlogy) {
  Repeatedly([this]() {
    auto type = Choose<DataType>({DT_FLOAT, DT_COMPLEX64});
    auto dims = BroadcastableDims();
    return ExpectTfAndXlaOutputsAreClose(OpTestBuilder("Xlogy")
                                             .RandomInput(type, dims.first)
                                             .RandomInput(type, dims.second)
                                             .Attr("T", type));
  });
}

TEST_F(OpTest, ZerosLike) {
  if (tensorflow::tf_xla_test_use_mlir) GTEST_SKIP() << "b/201095155";
  Repeatedly([this]() {
    auto type = Choose<DataType>({DT_INT32, DT_FLOAT, DT_COMPLEX64});
    return ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("ZerosLike").RandomInput(type).Attr("T", type));
  });
}

TEST_F(OpTest, Zeta) {
  Repeatedly([this]() {
    auto dims = BroadcastableDims();
    return ExpectTfAndXlaOutputsAreClose(OpTestBuilder("Xlogy")
                                             .RandomInput(DT_FLOAT, dims.first)
                                             .RandomInput(DT_FLOAT, dims.second)
                                             .Attr("T", DT_FLOAT));
  });
}

// Example failing run:
//   --tf_xla_reference_device=GPU:0
//   --tf_xla_test_use_jit=true --tf_xla_test_device=GPU:0
//   --tf_xla_test_use_mlir=true
//   --tf_xla_test_repetitions=2
//   --gunit_filter='OpTest.FusedBatchNormTraining'
//   --tf_xla_random_seed=2838146746
TEST_F(OpTest, FusedBatchNormTraining) {
  if (tensorflow::tf_xla_test_use_mlir) GTEST_SKIP() << "b/201095155";
  if (!tensorflow::tf_xla_test_use_mlir) GTEST_SKIP() << "b/197140886";
  bool is_nhwc = RandomBool();
  std::vector<int64_t> x_dims = RandomDims(/*min_rank=*/4, /*max_rank=*/4,
                                           /*min_size=*/5, /*max_size=*/20);
  std::vector<int64_t> scale_dims = {x_dims[is_nhwc ? 3 : 1]};
  std::vector<int64_t> offset_dims = {x_dims[is_nhwc ? 3 : 1]};
  std::vector<int64_t> mean_dims = {0};
  std::vector<int64_t> variance_dims = {0};
  DataType type = DT_FLOAT;
  Repeatedly([&] {
    return ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("FusedBatchNorm")
            .RandomInput(type, x_dims)
            .RandomInput(type, scale_dims)
            .RandomInput(type, offset_dims)
            .RandomInput(type, mean_dims)
            .RandomInput(type, variance_dims)
            .Attr("T", type)
            .Attr("data_format", is_nhwc ? "NHWC" : "NCHW")
            .Attr("epsilon", static_cast<float>(1.001e-05))
            .Attr("is_training", true));
  });
}
}  // anonymous namespace
}  // namespace tensorflow

int main(int argc, char** argv) {
  tensorflow::tf_xla_test_device_ptr = new tensorflow::string("GPU:0");
  tensorflow::tf_xla_reference_device_ptr = new tensorflow::string("CPU:0");
  std::vector<tensorflow::Flag> flag_list = {
      tensorflow::Flag(
          "tf_xla_random_seed", &tensorflow::tf_xla_random_seed,
          "Random seed to use for XLA tests. <= 0 means choose a seed "
          "nondeterministically."),
      // TODO(phawkins): it might make more sense to run each test up to a
      // configurable time bound.
      tensorflow::Flag("tf_xla_test_repetitions",
                       &tensorflow::tf_xla_test_repetitions,
                       "Number of repetitions for each test."),
      tensorflow::Flag("tf_xla_max_tensor_size",
                       &tensorflow::tf_xla_max_tensor_size,
                       "Maximum number of elements for random input tensors."),
      tensorflow::Flag("tf_xla_test_device", tensorflow::tf_xla_test_device_ptr,
                       "Tensorflow device type to use for test"),
      tensorflow::Flag("tf_xla_reference_device",
                       tensorflow::tf_xla_reference_device_ptr,
                       "Tensorflow device type to use for reference"),
      tensorflow::Flag("tf_xla_test_use_jit", &tensorflow::tf_xla_test_use_jit,
                       "Use JIT compilation for the operator under test"),
      tensorflow::Flag(
          "tf_xla_test_use_mlir", &tensorflow::tf_xla_test_use_mlir,
          "Use MLIR legalization kernels for the operator under test"),
  };
  tensorflow::string usage = tensorflow::Flags::Usage(argv[0], flag_list);
  const bool parse_result = tensorflow::Flags::Parse(&argc, argv, flag_list);
  if (!parse_result) {
    LOG(ERROR) << "\n" << usage;
    return 2;
  }
  testing::InitGoogleTest(&argc, argv);
  if (argc > 1) {
    LOG(ERROR) << "Unknown argument " << argv[1] << "\n" << usage;
    return 2;
  }
  // XLA devices register kernels at construction time; create all known devices
  // to make sure the kernels are registered.
  std::vector<std::unique_ptr<tensorflow::Device>> devices;
  TF_CHECK_OK(tensorflow::DeviceFactory::AddDevices(
      tensorflow::SessionOptions(), "", &devices));
  tensorflow::StaticDeviceMgr device_mgr(std::move(devices));

  tensorflow::Device* ignored;
  TF_QCHECK_OK(
      device_mgr.LookupDevice(*tensorflow::tf_xla_test_device_ptr, &ignored))
      << "Unknown test device (" << *tensorflow::tf_xla_test_device_ptr
      << "). Did you build in the right configuration (e.g., is CUDA enabled)?";

  if (tensorflow::tf_xla_test_use_mlir)
    tensorflow::GetMlirCommonFlags()->tf_mlir_enable_mlir_bridge =
        tensorflow::ConfigProto::Experimental::MLIR_BRIDGE_ROLLOUT_ENABLED;
  return RUN_ALL_TESTS();
}
