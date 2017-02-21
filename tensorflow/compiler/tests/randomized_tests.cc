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
// * ArgMax
// * DepthwiseConv2DNative
// * Gather
// * InvertPermutation
// * MaxPoolGrad (requires implementation of forward operator)
// * Select
// * Unpack
//
// TODO(phawkins): improve tests for:
// * StridedSliceGrad (need to use shape function to compute sensible inputs)

#include <random>
#include <unordered_map>

#include "tensorflow/compiler/jit/defs.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/util/command_line_flags.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace tensorflow {
namespace {

// Command line flags: see main() below.
int64 tf_xla_random_seed = 0;
int32 tf_xla_test_repetitions = 20;
string* tf_xla_test_device_ptr;  // initial value set in main()
bool tf_xla_test_use_jit = true;

string LocalDeviceToFullDeviceName(const string& device) {
  return strings::StrCat("/job:localhost/replica:0/task:0/device:", device);
}

constexpr std::array<DataType, 3> kAllXlaTypes = {
    {DT_INT32, DT_FLOAT, DT_BOOL}};

// An OpTestBuilder is a graph builder class that takes as input an operator to
// test, its inputs and attributes, and builds a graph that executes the
// operator.
class OpTestBuilder {
 public:
  explicit OpTestBuilder(const string& op_name);

  // Adds an input 'tensor'.
  OpTestBuilder& Input(Tensor tensor);

  // Sets an attribute.
  template <class T>
  OpTestBuilder& Attr(StringPiece attr_name, T&& value);

  // Overload needed to allow {...} expressions for value.
  template <class T>
  OpTestBuilder& Attr(StringPiece attr_name, std::initializer_list<T> value);

  // Adds nodes that executes the operator under test on 'device' to 'graphdef'.
  // If 'use_jit' is true, marks the operator under test to be compiled by XLA.
  // The graph will consist of one Placeholder node per input, the operator
  // itself, and one Identity node per output. If 'test_node_def' is not null,
  // sets it to the NodeDef of the operator under test. Fills 'inputs' and
  // 'outputs' with the names of the input placeholder nodes and the output
  // identity nodes, respectively.
  Status BuildGraph(string name_prefix, string device, bool use_jit,
                    GraphDef* graphdef, NodeDef** test_node_def,
                    std::vector<string>* inputs,
                    std::vector<string>* outputs) const;

  const std::vector<Tensor>& inputs() const { return inputs_; }

 private:
  NodeDef node_def_;
  std::vector<Tensor> inputs_;
};

OpTestBuilder::OpTestBuilder(const string& op_name) {
  node_def_.set_op(op_name);
}

OpTestBuilder& OpTestBuilder::Input(Tensor tensor) {
  VLOG(1) << "Adding input: " << tensor.DebugString();
  inputs_.push_back(tensor);
  return *this;
}

template <class T>
OpTestBuilder& OpTestBuilder::Attr(StringPiece attr_name, T&& value) {
  AddNodeAttr(attr_name, std::forward<T>(value), &node_def_);
  return *this;
}

template <class T>
OpTestBuilder& OpTestBuilder::Attr(StringPiece attr_name,
                                   std::initializer_list<T> value) {
  Attr<std::initializer_list<T>>(attr_name, std::move(value));
  return *this;
}

Status OpTestBuilder::BuildGraph(string name_prefix, string device,
                                 bool use_jit, GraphDef* graphdef,
                                 NodeDef** test_node_def,
                                 std::vector<string>* inputs,
                                 std::vector<string>* outputs) const {
  OpRegistryInterface* op_registry = OpRegistry::Global();

  const OpDef* op_def;
  TF_RETURN_IF_ERROR(op_registry->LookUpOpDef(node_def_.op(), &op_def));

  NodeDef* test_def = graphdef->add_node();
  *test_def = node_def_;
  test_def->set_name(strings::StrCat(name_prefix, "_op_under_test"));
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
    string name = strings::StrCat(name_prefix, "_input_", i);
    TF_RETURN_IF_ERROR(NodeDefBuilder(name, "Placeholder")
                           .Device(device)
                           .Attr("dtype", input_types[i])
                           .Finalize(def));
    inputs->push_back(name);
    test_def->add_input(name);
  }

  for (int i = 0; i < output_types.size(); ++i) {
    NodeDef* def = graphdef->add_node();
    string name = strings::StrCat(name_prefix, "_output_", i);
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

  return Status::OK();
}

// Test fixture. The fixture manages the random number generator and its seed,
// and has a number of convenience methods for building random Tensors, shapes,
// etc.
class OpTest : public ::testing::Test {
 public:
  OpTest();

  // Runs 'fn' up to --tf_xla_test_repetitions times, or until a failure occurs;
  // whichever happens first.
  void Repeatedly(std::function<void(void)> fn);

  // Select a random element from 'candidates'.
  template <typename T>
  T Choose(gtl::ArraySlice<T> candidates);

  static constexpr int kDefaultMaxRank = 5;
  static constexpr int64 kDefaultMaxDimensionSize = 20LL;

  // Returns a random dimension size.
  int64 RandomDim(int64 min = 0, int64 max = kDefaultMaxDimensionSize);

  // Returns a random shape. The tensor has rank in the range [min_rank,
  // max_rank).
  // Each dimension has size [0, kDefaultMaxDimensionSize].
  std::vector<int64> RandomDims(int min_rank = 0,
                                int max_rank = kDefaultMaxRank,
                                int64 min_size = 0,
                                int64 max_size = kDefaultMaxDimensionSize);

  // Given a shape 'dims', build a pair of dimensions such that one broadcasts
  // to the other.
  std::pair<std::vector<int64>, std::vector<int64>> BroadcastableDims(
      std::vector<int64> dims);

  // Builds a random pair of broadcastable dims.
  // TODO(phawkins): currently the maximum rank is 3, because broadcasting > 3
  // dimensions is unimplemented by the Tensorflow Eigen code (b/29268487)
  std::pair<std::vector<int64>, std::vector<int64>> BroadcastableDims();

  // Returns a tensor filled with random but "reasonable" values from the middle
  // of the type's range. If the shape is omitted, a random shape is used.
  // TODO(phawkins): generalize this code to a caller-supplied distribution.
  Tensor RandomTensor(DataType dtype, gtl::ArraySlice<int64> shape);
  Tensor RandomTensor(DataType dtype);

  // Like RandomTensor, but uses values >= 0.
  Tensor RandomNonNegativeTensor(DataType dtype, gtl::ArraySlice<int64> shape);
  Tensor RandomNonNegativeTensor(DataType dtype);

  // Returns a random subset of the integers in the range [0, rank), suitable
  // for use as reduction indices.
  Tensor RandomReductionIndices(int rank);

  struct WindowedDims {
    Padding padding;
    int kernel_rows, kernel_cols;
    int stride_rows, stride_cols;
    int input_rows, input_cols;
    int64 output_rows, output_cols;
  };
  // Choose dimensions for a 2D windowed op such as pooling or convolution.
  // TODO(phawkins): currently this only produces spatial windows, in NHWC
  // format.
  WindowedDims ChooseWindowedDims();

  std::mt19937& generator() { return *generator_; }

  // Run the test case described by 'builder' with and without XLA and check
  // that the outputs are close. Tensors x and y are close if they have the same
  // type, same shape, and have close values. For floating-point tensors, the
  // element-wise difference between x and y must no more than
  // atol + rtol * abs(x); or both elements may be NaN or infinity. For
  // non-floating-point tensors the element values must match exactly.
  void ExpectTfAndXlaOutputsAreClose(const OpTestBuilder& builder,
                                     double atol = 1e-2, double rtol = 1e-2);

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
  int64 s = tf_xla_random_seed;
  unsigned int seed;
  if (s <= 0) {
    std::random_device random_device;
    seed = random_device();
  } else {
    seed = static_cast<unsigned int>(s);
  }
  LOG(INFO) << "Random seed for test case: " << seed
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

void OpTest::Repeatedly(std::function<void(void)> fn) {
  int const max_repetitions = tf_xla_test_repetitions;
  for (int i = 0; !HasFailure() && i < max_repetitions; ++i) {
    fn();
  }
}

template <typename T>
T OpTest::Choose(gtl::ArraySlice<T> candidates) {
  std::uniform_int_distribution<size_t> d(0, candidates.size() - 1);
  return candidates[d(generator())];
}

int64 OpTest::RandomDim(int64 min, int64 max) {
  std::uniform_int_distribution<int64> size_distribution(min, max - 1);
  return size_distribution(generator());
}

std::vector<int64> OpTest::RandomDims(int min_rank, int max_rank,
                                      int64 min_size, int64 max_size) {
  CHECK_LE(0, min_rank);
  CHECK_LE(min_rank, max_rank);
  std::uniform_int_distribution<int> rank_distribution(min_rank, max_rank);
  int rank = rank_distribution(generator());
  std::vector<int64> dims(rank);
  std::generate(dims.begin(), dims.end(), [this, min_size, max_size]() {
    return RandomDim(min_size, max_size);
  });
  return dims;
}

Tensor OpTest::RandomTensor(DataType dtype, gtl::ArraySlice<int64> shape) {
  Tensor tensor(dtype, TensorShape(shape));
  switch (dtype) {
    case DT_FLOAT: {
      std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
      test::FillFn<float>(&tensor, [this, &distribution](int i) -> float {
        return distribution(generator());
      });
      break;
    }
    case DT_DOUBLE: {
      std::uniform_real_distribution<double> distribution(-1.0, 1.0);
      test::FillFn<double>(&tensor, [this, &distribution](int i) -> double {
        return distribution(generator());
      });
      break;
    }
    case DT_INT32: {
      std::uniform_int_distribution<int32> distribution(-(1 << 20), 1 << 20);
      test::FillFn<int32>(&tensor, [this, &distribution](int i) -> int32 {
        return distribution(generator());
      });
      break;
    }
    case DT_INT64: {
      std::uniform_int_distribution<int64> distribution(-(1LL << 40),
                                                        1LL << 40);
      test::FillFn<int64>(&tensor, [this, &distribution](int i) -> int64 {
        return distribution(generator());
      });
      break;
    }
    case DT_BOOL: {
      std::bernoulli_distribution distribution;
      test::FillFn<bool>(&tensor, [this, &distribution](int i) -> bool {
        return distribution(generator());
      });
      break;
    }
    default:
      LOG(FATAL) << "Unimplemented type " << dtype << " in RandomTensor";
  }
  return tensor;
}

Tensor OpTest::RandomTensor(DataType dtype) {
  return RandomTensor(dtype, RandomDims());
}

Tensor OpTest::RandomNonNegativeTensor(DataType dtype,
                                       gtl::ArraySlice<int64> shape) {
  Tensor tensor(dtype, TensorShape(shape));
  switch (dtype) {
    case DT_FLOAT: {
      std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
      test::FillFn<float>(&tensor, [this, &distribution](int i) -> float {
        return distribution(generator());
      });
      break;
    }
    case DT_DOUBLE: {
      std::uniform_real_distribution<double> distribution(0.0, 1.0);
      test::FillFn<double>(&tensor, [this, &distribution](int i) -> double {
        return distribution(generator());
      });
      break;
    }
    case DT_INT32: {
      std::uniform_int_distribution<int32> distribution(0, 1 << 20);
      test::FillFn<int32>(&tensor, [this, &distribution](int i) -> int32 {
        return distribution(generator());
      });
      break;
    }
    case DT_INT64: {
      std::uniform_int_distribution<int64> distribution(0, 1LL << 40);
      test::FillFn<int64>(&tensor, [this, &distribution](int i) -> int64 {
        return distribution(generator());
      });
      break;
    }
    default:
      LOG(FATAL) << "Unimplemented type " << dtype
                 << " in RandomNonNegativeTensor";
  }
  return tensor;
}

Tensor OpTest::RandomNonNegativeTensor(DataType dtype) {
  return RandomNonNegativeTensor(dtype, RandomDims());
}

std::pair<std::vector<int64>, std::vector<int64>> OpTest::BroadcastableDims(
    std::vector<int64> dims) {
  if (dims.empty()) return {dims, dims};

  // Remove some dimensions from the front of 'dims'.
  size_t skip =
      std::uniform_int_distribution<size_t>(0, dims.size() - 1)(generator());

  std::vector<int64> bdims(dims.begin() + skip, dims.end());

  // Randomly replace some of the remaining dimensions of 'dims' with 1.
  std::bernoulli_distribution random_bool;

  for (int64& dim : bdims) {
    if (random_bool(generator())) {
      dim = 1LL;
    }
  }

  // Possibly swap the roles of 'dims' and 'bdims'.
  if (random_bool(generator())) {
    dims.swap(bdims);
  }
  return {dims, bdims};
}

std::pair<std::vector<int64>, std::vector<int64>> OpTest::BroadcastableDims() {
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

OpTest::WindowedDims OpTest::ChooseWindowedDims() {
  WindowedDims d;
  d.padding = Choose<Padding>({SAME, VALID});
  std::uniform_int_distribution<int> random_int(1, 5);
  Status s;
  // Repeatedly try different filter/stride sizes until we find a valid
  // combination.
  do {
    // CPU implementations require stride <= kernel size.
    d.kernel_rows = random_int(generator()),
    d.input_rows = RandomDim(d.kernel_rows);
    d.stride_rows =
        std::uniform_int_distribution<int>(1, d.kernel_rows)(generator());
    int64 pad_dummy;
    s = GetWindowedOutputSize(d.input_rows, d.kernel_rows, d.stride_rows,
                              d.padding, &d.output_rows, &pad_dummy);
  } while (!s.ok());
  do {
    d.kernel_cols = random_int(generator());
    d.input_cols = RandomDim(d.kernel_cols);
    d.stride_cols =
        std::uniform_int_distribution<int>(1, d.kernel_cols)(generator());
    int64 pad_dummy;
    s = GetWindowedOutputSize(d.input_cols, d.kernel_cols, d.stride_cols,
                              d.padding, &d.output_cols, &pad_dummy);
  } while (!s.ok());
  return d;
}

// Functions for comparing tensors.

template <typename T>
bool IsClose(const T& x, const T& y, double atol, double rtol) {
  if (std::isnan(x) && std::isnan(y)) return true;
  if (x == y) return true;  // Allow inf == inf.
  return fabs(x - y) < atol + rtol * fabs(x);
}

template <typename T>
Status TensorsAreCloseImpl(const Tensor& x, const Tensor& y, double atol,
                           double rtol) {
  auto Tx = x.flat<T>();
  auto Ty = y.flat<T>();
  for (int i = 0; i < Tx.size(); ++i) {
    if (!IsClose(Tx(i), Ty(i), atol, rtol)) {
      return errors::InvalidArgument(strings::StrCat(
          i, "-th tensor element isn't close: ", Tx(i), " vs. ", Ty(i),
          ". x = ", x.DebugString(), "y = ", y.DebugString(), "atol = ", atol,
          " rtol = ", rtol, " tol = ", atol + rtol * std::fabs(Tx(i))));
    }
  }
  return Status::OK();
}

template <typename T>
Status TensorsAreEqualImpl(const Tensor& x, const Tensor& y) {
  auto Tx = x.flat<T>();
  auto Ty = y.flat<T>();
  for (int i = 0; i < Tx.size(); ++i) {
    if (Tx(i) != Ty(i)) {
      return errors::InvalidArgument(strings::StrCat(
          i, "-th tensor element isn't equal: ", Tx(i), " vs. ", Ty(i),
          ". x = ", x.DebugString(), "y = ", y.DebugString()));
    }
  }
  return Status::OK();
}

// Tests if "x" and "y" are tensors of the same type, same shape, and with
// close values. For floating-point tensors, the element-wise difference between
// x and y must no more than atol + rtol * abs(x). For non-floating-point
// tensors the values must match exactly.
Status TensorsAreClose(const Tensor& a, const Tensor& b, double atol,
                       double rtol) {
  if (a.dtype() != b.dtype()) {
    return errors::InvalidArgument(strings::StrCat(
        "Tensors have different types: ", DataTypeString(a.dtype()), " and ",
        DataTypeString(b.dtype())));
  }
  if (!a.IsSameSize(b)) {
    return errors::InvalidArgument(strings::StrCat(
        "Tensors have different shapes: ", a.shape().DebugString(), " and ",
        b.shape().DebugString()));
  }

  switch (a.dtype()) {
    case DT_FLOAT:
      return TensorsAreCloseImpl<float>(a, b, atol, rtol);
    case DT_DOUBLE:
      return TensorsAreCloseImpl<double>(a, b, atol, rtol);
    case DT_INT32:
      return TensorsAreEqualImpl<int32>(a, b);
    case DT_INT64:
      return TensorsAreEqualImpl<int64>(a, b);
    case DT_BOOL:
      return TensorsAreEqualImpl<bool>(a, b);
    default:
      LOG(FATAL) << "Unexpected type : " << DataTypeString(a.dtype());
  }
}

void OpTest::ExpectTfAndXlaOutputsAreClose(const OpTestBuilder& builder,
                                           double atol, double rtol) {
  string cpu_device =
      LocalDeviceToFullDeviceName(strings::StrCat(DEVICE_CPU, ":0"));
  string test_device = LocalDeviceToFullDeviceName(*tf_xla_test_device_ptr);

  DeviceNameUtils::ParsedName parsed_name;
  ASSERT_TRUE(
      DeviceNameUtils::ParseLocalName(*tf_xla_test_device_ptr, &parsed_name));
  DeviceType test_device_type(parsed_name.type);
  ++num_tests_;

  GraphDef graph;
  std::vector<string> expected_inputs, test_inputs;
  std::vector<string> expected_fetches, test_fetches;
  TF_ASSERT_OK(builder.BuildGraph(
      strings::StrCat("test", num_tests_, "_expected"), cpu_device,
      /* use_jit= */ false, &graph, /* test_node_def= */ nullptr,
      &expected_inputs, &expected_fetches));

  NodeDef* node_def;
  TF_ASSERT_OK(builder.BuildGraph(strings::StrCat("test", num_tests_, "_test"),
                                  test_device, tf_xla_test_use_jit, &graph,
                                  &node_def, &test_inputs, &test_fetches));

  // Check that there's a kernel corresponding to 'node_def' on the device under
  // test.
  Status status = FindKernelDef(test_device_type, *node_def, nullptr, nullptr);
  if (!status.ok()) {
    VLOG(1) << "Skipping test because there is no corresponding registered "
            << "kernel on the test device: " << status;
    return;
  }

  TF_ASSERT_OK(session_->Extend(graph));

  const std::vector<Tensor>& input_tensors = builder.inputs();
  if (VLOG_IS_ON(1)) {
    for (const Tensor& input : input_tensors) {
      VLOG(1) << "Input: " << input.DebugString();
    }
  }

  std::vector<std::pair<string, Tensor>> expected_feeds(expected_inputs.size());
  std::vector<std::pair<string, Tensor>> test_feeds(test_inputs.size());
  ASSERT_EQ(input_tensors.size(), expected_inputs.size());
  ASSERT_EQ(input_tensors.size(), test_inputs.size());

  for (int i = 0; i < input_tensors.size(); ++i) {
    expected_feeds[i] = {expected_inputs[i], input_tensors[i]};
    test_feeds[i] = {test_inputs[i], input_tensors[i]};
  }

  std::vector<Tensor> expected_outputs, test_outputs;
  VLOG(1) << "Running expected graph";
  Status s =
      session_->Run(expected_feeds, expected_fetches, {}, &expected_outputs);
  if (!s.ok()) {
    VLOG(1) << "Expected graph failed with status: " << s << ". Skipping test";
    return;
  }

  VLOG(1) << "Running test graph";
  TF_ASSERT_OK(session_->Run(test_feeds, test_fetches, {}, &test_outputs));

  ASSERT_EQ(expected_outputs.size(), test_outputs.size());
  for (int j = 0; s.ok() && j < test_outputs.size(); ++j) {
    s = TensorsAreClose(expected_outputs[j], test_outputs[j], atol, rtol);
  }
  TF_EXPECT_OK(s);
}

// Helper that converts 'values' to an int32 or int64 Tensor.
Tensor AsIntTensor(DataType dtype, const std::vector<int64>& values) {
  switch (dtype) {
    case DT_INT32: {
      std::vector<int32> values32(values.begin(), values.end());
      return test::AsTensor<int32>(values32);
    }
    case DT_INT64:
      return test::AsTensor<int64>(values);
    default:
      CHECK(false);
  }
}

TEST_F(OpTest, Abs) {
  Repeatedly([this]() {
    DataType type = Choose<DataType>({DT_INT32, DT_FLOAT});
    ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("Abs").Input(RandomTensor(type)).Attr("T", type));
  });
}

TEST_F(OpTest, Add) {
  Repeatedly([this]() {
    DataType type = Choose<DataType>({DT_INT32, DT_FLOAT});
    auto dims = BroadcastableDims();
    ExpectTfAndXlaOutputsAreClose(OpTestBuilder("Add")
                                      .Input(RandomTensor(type, dims.first))
                                      .Input(RandomTensor(type, dims.second))
                                      .Attr("T", type));
  });
}

TEST_F(OpTest, AddN) {
  Repeatedly([this]() {
    DataType type = Choose<DataType>({DT_INT32, DT_FLOAT});
    int n = std::uniform_int_distribution<int>(1, 5)(generator());

    auto shape = RandomDims();

    OpTestBuilder builder("AddN");
    builder.Attr("T", type);
    builder.Attr("N", n);
    for (int i = 0; i < n; ++i) {
      builder.Input(RandomTensor(type, shape));
    }
    ExpectTfAndXlaOutputsAreClose(builder);
  });
}

TEST_F(OpTest, All) {
  Repeatedly([this]() {
    Tensor data = RandomTensor(DT_BOOL);
    Tensor indices = RandomReductionIndices(data.dims());
    bool keep_dims = Choose<bool>({false, true});
    ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("All").Input(data).Input(indices).Attr("keep_dims",
                                                             keep_dims));
  });
}

TEST_F(OpTest, Any) {
  Repeatedly([this]() {
    Tensor data = RandomTensor(DT_BOOL);
    Tensor indices = RandomReductionIndices(data.dims());
    bool keep_dims = Choose<bool>({false, true});
    ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("Any").Input(data).Input(indices).Attr("keep_dims",
                                                             keep_dims));
  });
}

TEST_F(OpTest, AvgPool) {
  Repeatedly([this]() {
    std::uniform_int_distribution<int> random_int(1, 5);
    int kernel_rows = random_int(generator()),
        kernel_cols = random_int(generator());
    int stride_rows = random_int(generator()),
        stride_cols = random_int(generator());
    string padding = Choose<string>({"SAME", "VALID"});
    ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("AvgPool")
            .Input(
                RandomTensor(DT_FLOAT, {RandomDim(1), RandomDim(kernel_rows),
                                        RandomDim(kernel_cols), RandomDim(1)}))
            .Attr("T", DT_FLOAT)
            .Attr("ksize", {1, kernel_rows, kernel_cols, 1})
            .Attr("strides", {1, stride_rows, stride_cols, 1})
            .Attr("padding", padding)
            .Attr("data_format", "NHWC"));
  });
  // TODO(phawkins): the CPU device only implements spatial pooling. Add tests
  // for batch pooling when supported.
}

TEST_F(OpTest, AvgPoolGrad) {
  Repeatedly([this]() {
    int batch = RandomDim(1), features = RandomDim(1);
    WindowedDims d = ChooseWindowedDims();
    ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("AvgPoolGrad")
            .Input(test::AsTensor<int32>(
                {batch, d.input_rows, d.input_cols, features}))
            .Input(RandomTensor(
                DT_FLOAT, {batch, d.output_rows, d.output_cols, features}))
            .Attr("T", DT_FLOAT)
            .Attr("ksize", {1, d.kernel_rows, d.kernel_cols, 1})
            .Attr("strides", {1, d.stride_rows, d.stride_cols, 1})
            .Attr("padding", d.padding == SAME ? "SAME" : "VALID")
            .Attr("data_format", "NHWC"));
  });
}

TEST_F(OpTest, BatchMatMul) {
  Repeatedly([this]() {
    std::vector<int64> output_dims = RandomDims(2, 5, 0, 7);
    int64 ndims = output_dims.size();
    int64 inner_dim = RandomDim();
    std::vector<int64> x_dims(output_dims), y_dims(output_dims);
    x_dims[ndims - 1] = inner_dim;
    y_dims[ndims - 2] = inner_dim;
    ExpectTfAndXlaOutputsAreClose(OpTestBuilder("BatchMatMul")
                                      .Input(RandomTensor(DT_FLOAT, x_dims))
                                      .Input(RandomTensor(DT_FLOAT, y_dims))
                                      .Attr("T", DT_FLOAT));

    std::swap(x_dims[ndims - 1], x_dims[ndims - 2]);
    ExpectTfAndXlaOutputsAreClose(OpTestBuilder("BatchMatMul")
                                      .Input(RandomTensor(DT_FLOAT, x_dims))
                                      .Input(RandomTensor(DT_FLOAT, y_dims))
                                      .Attr("T", DT_FLOAT)
                                      .Attr("adj_x", true));

    std::swap(y_dims[ndims - 1], y_dims[ndims - 2]);
    ExpectTfAndXlaOutputsAreClose(OpTestBuilder("BatchMatMul")
                                      .Input(RandomTensor(DT_FLOAT, x_dims))
                                      .Input(RandomTensor(DT_FLOAT, y_dims))
                                      .Attr("T", DT_FLOAT)
                                      .Attr("adj_x", true)
                                      .Attr("adj_y", true));

    std::swap(x_dims[ndims - 1], x_dims[ndims - 2]);
    ExpectTfAndXlaOutputsAreClose(OpTestBuilder("BatchMatMul")
                                      .Input(RandomTensor(DT_FLOAT, x_dims))
                                      .Input(RandomTensor(DT_FLOAT, y_dims))
                                      .Attr("T", DT_FLOAT)
                                      .Attr("adj_y", true));
  });
}

TEST_F(OpTest, BiasAdd) {
  Repeatedly([this]() {
    auto x = RandomTensor(DT_FLOAT, RandomDims(2, kDefaultMaxRank));
    auto y = RandomTensor(DT_FLOAT, {x.dim_size(x.dims() - 1)});
    // TODO(phawkins): test both data formats.
    ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("BiasAdd").Input(x).Input(y).Attr("T", DT_FLOAT));
  });
}

TEST_F(OpTest, BiasAddGrad) {
  Repeatedly([this]() {
    auto x = RandomTensor(DT_FLOAT);
    // TODO(phawkins): test both data formats.
    ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("BiasAddGrad").Input(x).Attr("T", DT_FLOAT));
  });
}

TEST_F(OpTest, BiasAddV1) {
  Repeatedly([this]() {
    auto x = RandomTensor(DT_FLOAT, RandomDims(2, kDefaultMaxRank));
    auto y = RandomTensor(DT_FLOAT, {x.dim_size(x.dims() - 1)});
    ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("BiasAddV1").Input(x).Input(y).Attr("T", DT_FLOAT));
  });
}

TEST_F(OpTest, BroadcastGradientArgs) {
  Repeatedly([this]() {
    // TODO(phawkins): only int32 seems to be implemented in Tensorflow.
    // DataType type = Choose<DataType>({DT_INT32, DT_INT64});
    DataType type = DT_INT32;
    auto dims = BroadcastableDims();
    ExpectTfAndXlaOutputsAreClose(OpTestBuilder("BroadcastGradientArgs")
                                      .Input(AsIntTensor(type, dims.first))
                                      .Input(AsIntTensor(type, dims.second))
                                      .Attr("T", type));
  });
}

TEST_F(OpTest, Cast) {
  Repeatedly([this]() {
    DataType src_type, dst_type;
    src_type = Choose<DataType>({DT_INT32, DT_FLOAT, DT_BOOL});
    dst_type = Choose<DataType>({DT_INT32, DT_FLOAT, DT_BOOL});
    ExpectTfAndXlaOutputsAreClose(OpTestBuilder("Cast")
                                      .Input(RandomTensor(src_type))
                                      .Attr("SrcT", src_type)
                                      .Attr("DstT", dst_type));
  });
}

TEST_F(OpTest, Ceil) {
  Repeatedly([this]() {
    ExpectTfAndXlaOutputsAreClose(OpTestBuilder("Ceil")
                                      .Input(RandomTensor(DT_FLOAT))
                                      .Attr("T", DT_FLOAT));
  });
}

TEST_F(OpTest, Concat) {
  Repeatedly([this]() {
    DataType type = Choose<DataType>(kAllXlaTypes);
    int n = std::uniform_int_distribution<int>(2, 5)(generator());

    std::vector<int64> dims = RandomDims(1);
    int concat_dim =
        std::uniform_int_distribution<int32>(0, dims.size() - 1)(generator());

    OpTestBuilder builder("Concat");
    builder.Input(test::AsScalar<int32>(concat_dim));
    builder.Attr("T", type);
    builder.Attr("N", n);
    for (int i = 0; i < n; ++i) {
      std::vector<int64> shape = dims;
      shape[concat_dim] = RandomDim();
      builder.Input(RandomTensor(type, shape));
    }
    ExpectTfAndXlaOutputsAreClose(builder);
  });
}

TEST_F(OpTest, ConcatOffset) {
  Repeatedly([this]() {
    int n = std::uniform_int_distribution<int>(2, 5)(generator());

    std::vector<int64> dims = RandomDims(1);
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
    ExpectTfAndXlaOutputsAreClose(builder);
  });
}

TEST_F(OpTest, Conv2D) {
  Repeatedly([this]() {
    WindowedDims d = ChooseWindowedDims();
    std::uniform_int_distribution<int> random_int(1, 5);
    int features_in = random_int(generator());
    int features_out = random_int(generator());
    Tensor data = RandomTensor(
        DT_FLOAT, {RandomDim(), d.input_rows, d.input_cols, features_in});

    Tensor kernel = RandomTensor(
        DT_FLOAT, {d.kernel_rows, d.kernel_cols, features_in, features_out});
    ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("Conv2D")
            .Input(data)
            .Input(kernel)
            .Attr("T", DT_FLOAT)
            .Attr("strides", {1, d.stride_rows, d.stride_cols, 1})
            .Attr("padding", d.padding == SAME ? "SAME" : "VALID")
            .Attr("data_format", "NHWC"));
  });
}

TEST_F(OpTest, Conv2DBackpropFilter) {
  Repeatedly([this]() {
    WindowedDims d = ChooseWindowedDims();
    std::uniform_int_distribution<int> random_int(1, 5);
    int features_in = random_int(generator());
    int features_out = random_int(generator());
    int32 batch = RandomDim();
    Tensor activations = RandomTensor(
        DT_FLOAT, {batch, d.input_rows, d.input_cols, features_in});
    Tensor backprop = RandomTensor(
        DT_FLOAT, {batch, d.output_rows, d.output_cols, features_out});
    Tensor kernel_shape = test::AsTensor<int32>(
        {d.kernel_rows, d.kernel_cols, features_in, features_out});
    ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("Conv2DBackpropFilter")
            .Input(activations)
            .Input(kernel_shape)
            .Input(backprop)
            .Attr("T", DT_FLOAT)
            .Attr("strides", {1, d.stride_rows, d.stride_cols, 1})
            .Attr("padding", d.padding == SAME ? "SAME" : "VALID")
            .Attr("data_format", "NHWC"));
  });
}

TEST_F(OpTest, Conv2DBackpropInput) {
  Repeatedly([this]() {
    WindowedDims d = ChooseWindowedDims();
    std::uniform_int_distribution<int> random_int(1, 5);
    int features_in = random_int(generator());
    int features_out = random_int(generator());
    int32 batch = RandomDim();
    Tensor in_shape =
        test::AsTensor<int32>({batch, d.input_rows, d.input_cols, features_in});
    Tensor backprop = RandomTensor(
        DT_FLOAT, {batch, d.output_rows, d.output_cols, features_out});
    Tensor kernel = RandomTensor(
        DT_FLOAT, {d.kernel_rows, d.kernel_cols, features_in, features_out});
    ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("Conv2DBackpropInput")
            .Input(in_shape)
            .Input(kernel)
            .Input(backprop)
            .Attr("T", DT_FLOAT)
            .Attr("strides", {1, d.stride_rows, d.stride_cols, 1})
            .Attr("padding", d.padding == SAME ? "SAME" : "VALID")
            .Attr("data_format", "NHWC"));
  });
}

TEST_F(OpTest, Diag) {
  Repeatedly([this]() {
    DataType type = Choose<DataType>({DT_INT32, DT_FLOAT});
    ExpectTfAndXlaOutputsAreClose(OpTestBuilder("Diag")
                                      .Input(RandomTensor(type, RandomDims(1)))
                                      .Attr("T", type));
  });
}

TEST_F(OpTest, DiagPart) {
  Repeatedly([this]() {
    DataType type = Choose<DataType>({DT_INT32, DT_FLOAT});
    auto dims = RandomDims(1, 3);
    // Duplicate the random dims.
    std::vector<int64> doubled_dims(dims.size() * 2);
    std::copy(dims.begin(), dims.end(), doubled_dims.begin());
    std::copy(dims.begin(), dims.end(), doubled_dims.begin() + dims.size());
    ExpectTfAndXlaOutputsAreClose(OpTestBuilder("DiagPart")
                                      .Input(RandomTensor(type, doubled_dims))
                                      .Attr("T", type));
  });
}

TEST_F(OpTest, Div) {
  Repeatedly([this]() {
    DataType type = Choose<DataType>({DT_INT32, DT_FLOAT});
    auto dims = BroadcastableDims();
    ExpectTfAndXlaOutputsAreClose(OpTestBuilder("Div")
                                      .Input(RandomTensor(type, dims.first))
                                      .Input(RandomTensor(type, dims.second))
                                      .Attr("T", type));
  });
}

TEST_F(OpTest, DynamicStitch) {
  Repeatedly([this]() {
    DataType type = Choose<DataType>(kAllXlaTypes);
    int n = std::uniform_int_distribution<int>(2, 5)(generator());
    OpTestBuilder builder("DynamicStitch");
    builder.Attr("T", type);
    builder.Attr("N", n);
    std::vector<std::vector<int64>> index_dims;
    int size = 0;
    // TODO(phawkins): the XLA implementation of DynamicStitch does not
    // accept an empty set of indices.
    do {
      size = 0;
      index_dims.clear();
      for (int i = 0; i < n; ++i) {
        std::vector<int64> dims = RandomDims(0, 3, 0, 5);
        size += TensorShape(dims).num_elements();
        index_dims.push_back(dims);
      }
    } while (size == 0);

    // Shuffle the range of indices that cover the output.
    // TODO(phawkins): The documentation for DynamicStitch doesn't require that
    // the indices cover all positions of the output. The XLA implementation
    // does so require. However, the native TF implementation leaves undefined
    // values if we don't cover everything, so we can't really test that case
    // anyway.
    std::vector<int32> indices(size);
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), generator());

    int pos = 0;
    for (int i = 0; i < n; ++i) {
      TensorShape shape(index_dims[i]);
      Tensor t = test::AsTensor<int32>(
          gtl::ArraySlice<int32>(indices, pos, shape.num_elements()), shape);
      builder.Input(t);
      pos += t.NumElements();
    }

    std::vector<int64> constant_dims = RandomDims(0, 3, 0, 5);
    for (int i = 0; i < n; ++i) {
      std::vector<int64> dims(index_dims[i].begin(), index_dims[i].end());
      std::copy(constant_dims.begin(), constant_dims.end(),
                std::back_inserter(dims));
      Tensor t = RandomTensor(type, dims);
      builder.Input(t);
    }
    ExpectTfAndXlaOutputsAreClose(builder);
  });
}

TEST_F(OpTest, Equal) {
  Repeatedly([this]() {
    DataType type = Choose<DataType>({DT_INT32, DT_FLOAT});
    auto dims = BroadcastableDims();
    ExpectTfAndXlaOutputsAreClose(OpTestBuilder("Equal")
                                      .Input(RandomTensor(type, dims.first))
                                      .Input(RandomTensor(type, dims.second))
                                      .Attr("T", type));
  });
}

TEST_F(OpTest, Exp) {
  Repeatedly([this]() {
    ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("Exp").Input(RandomTensor(DT_FLOAT)).Attr("T", DT_FLOAT));
  });
}

TEST_F(OpTest, ExpandDims) {
  Repeatedly([this]() {
    DataType type = Choose<DataType>(kAllXlaTypes);
    Tensor in = RandomTensor(type);
    Tensor dim(DT_INT32, TensorShape());
    std::uniform_int_distribution<int32> d(-1 - in.dims(), in.dims());
    dim.scalar<int32>()() = d(generator());
    ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("ExpandDims").Input(in).Input(dim).Attr("T", type));
  });
}

TEST_F(OpTest, Fill) {
  Repeatedly([this]() {
    DataType type = Choose<DataType>(kAllXlaTypes);
    Tensor scalar = RandomTensor(type, {});
    std::vector<int64> dims = RandomDims();
    std::vector<int32> shape(dims.begin(), dims.end());
    ExpectTfAndXlaOutputsAreClose(OpTestBuilder("Fill")
                                      .Input(test::AsTensor<int32>(shape))
                                      .Input(scalar)
                                      .Attr("T", type));
  });
}

TEST_F(OpTest, Floor) {
  Repeatedly([this]() {
    ExpectTfAndXlaOutputsAreClose(OpTestBuilder("Floor")
                                      .Input(RandomTensor(DT_FLOAT))
                                      .Attr("T", DT_FLOAT));
  });
}

TEST_F(OpTest, FloorDiv) {
  Repeatedly([this]() {
    DataType type = DT_INT32;
    auto dims = BroadcastableDims();
    ExpectTfAndXlaOutputsAreClose(OpTestBuilder("FloorDiv")
                                      .Input(RandomTensor(type, dims.first))
                                      .Input(RandomTensor(type, dims.second))
                                      .Attr("T", type));
  });
}

TEST_F(OpTest, FloorMod) {
  Repeatedly([this]() {
    DataType type = Choose<DataType>({DT_INT32, DT_FLOAT});
    auto dims = BroadcastableDims();
    ExpectTfAndXlaOutputsAreClose(OpTestBuilder("FloorMod")
                                      .Input(RandomTensor(type, dims.first))
                                      .Input(RandomTensor(type, dims.second))
                                      .Attr("T", type));
  });
}

TEST_F(OpTest, Greater) {
  Repeatedly([this]() {
    DataType type = Choose<DataType>({DT_INT32, DT_FLOAT});
    auto dims = BroadcastableDims();
    ExpectTfAndXlaOutputsAreClose(OpTestBuilder("Greater")
                                      .Input(RandomTensor(type, dims.first))
                                      .Input(RandomTensor(type, dims.second))
                                      .Attr("T", type));
  });
}

TEST_F(OpTest, GreaterEqual) {
  Repeatedly([this]() {
    DataType type = Choose<DataType>({DT_INT32, DT_FLOAT});
    auto dims = BroadcastableDims();
    ExpectTfAndXlaOutputsAreClose(OpTestBuilder("GreaterEqual")
                                      .Input(RandomTensor(type, dims.first))
                                      .Input(RandomTensor(type, dims.second))
                                      .Attr("T", type));
  });
}

TEST_F(OpTest, Reciprocal) {
  Repeatedly([this]() {
    ExpectTfAndXlaOutputsAreClose(OpTestBuilder("Reciprocal")
                                      .Input(RandomTensor(DT_FLOAT))
                                      .Attr("T", DT_FLOAT));
  });
}

TEST_F(OpTest, L2Loss) {
  Repeatedly([this]() {
    DataType type = Choose<DataType>({DT_INT32, DT_FLOAT});
    // TODO(b/31644876): scalars currently crash.
    ExpectTfAndXlaOutputsAreClose(OpTestBuilder("L2Loss")
                                      .Input(RandomTensor(type, RandomDims(1)))
                                      .Attr("T", type));
  });
}

TEST_F(OpTest, Less) {
  Repeatedly([this]() {
    DataType type = Choose<DataType>({DT_INT32, DT_FLOAT});
    auto dims = BroadcastableDims();
    ExpectTfAndXlaOutputsAreClose(OpTestBuilder("Less")
                                      .Input(RandomTensor(type, dims.first))
                                      .Input(RandomTensor(type, dims.second))
                                      .Attr("T", type));
  });
}

TEST_F(OpTest, LessEqual) {
  Repeatedly([this]() {
    DataType type = Choose<DataType>({DT_INT32, DT_FLOAT});
    auto dims = BroadcastableDims();
    ExpectTfAndXlaOutputsAreClose(OpTestBuilder("LessEqual")
                                      .Input(RandomTensor(type, dims.first))
                                      .Input(RandomTensor(type, dims.second))
                                      .Attr("T", type));
  });
}

TEST_F(OpTest, LinSpace) {
  Repeatedly([this]() {
    auto ToScalar = [](DataType type, int x) {
      if (type == DT_INT32) return test::AsScalar<int32>(x);
      return test::AsScalar<int64>(x);
    };
    std::uniform_int_distribution<int> distribution(-50, 50);
    DataType type = Choose<DataType>({DT_INT32, DT_INT64});
    ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("LinSpace")
            .Input(RandomTensor(DT_FLOAT, {}))
            .Input(RandomTensor(DT_FLOAT, {}))
            .Input(ToScalar(type, distribution(generator())))
            .Attr("T", DT_FLOAT)
            .Attr("Tidx", type));
  });
}

TEST_F(OpTest, Log) {
  Repeatedly([this]() {
    ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("Log").Input(RandomTensor(DT_FLOAT)).Attr("T", DT_FLOAT));
  });
}

TEST_F(OpTest, LogicalAnd) {
  Repeatedly([this]() {
    auto dims = BroadcastableDims();
    ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("LogicalAnd")
            .Input(RandomTensor(DT_BOOL, dims.first))
            .Input(RandomTensor(DT_BOOL, dims.second)));
  });
}

TEST_F(OpTest, LogicalNot) {
  Repeatedly([this]() {
    ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("LogicalNot").Input(RandomTensor(DT_BOOL)));
  });
}

TEST_F(OpTest, LogicalOr) {
  Repeatedly([this]() {
    auto dims = BroadcastableDims();
    ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("LogicalOr")
            .Input(RandomTensor(DT_BOOL, dims.first))
            .Input(RandomTensor(DT_BOOL, dims.second)));
  });
}

TEST_F(OpTest, LogSoftmax) {
  Repeatedly([this]() {
    ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("LogSoftmax")
            .Input(RandomTensor(DT_FLOAT, RandomDims(2, 2)))
            .Attr("T", DT_FLOAT));
  });
}

TEST_F(OpTest, LRN) {
  Repeatedly([this]() {
    Tensor data;
    // TODO(b/31362467): Crashes with 0 dims on GPU. Re-enable when fixed.
    data = RandomTensor(DT_FLOAT, RandomDims(4, 4, 1, 8));
    // CuDNN requires depth_radius > 0.
    std::uniform_int_distribution<int> radius(1, data.dim_size(3));
    std::uniform_real_distribution<float> coeff(0.01, 2.0);
    ExpectTfAndXlaOutputsAreClose(OpTestBuilder("LRN")
                                      .Input(data)
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
    std::vector<int64> dims = RandomDims(4, 4, 1, 8);
    Tensor input_grads = RandomTensor(DT_FLOAT, dims);
    Tensor input_image = RandomTensor(DT_FLOAT, dims);
    Tensor output_image = RandomTensor(DT_FLOAT, dims);
    // CuDNN requires depth_radius > 0.
    std::uniform_int_distribution<int> radius(1, input_grads.dim_size(3));
    std::uniform_real_distribution<float> coeff(0.0, 2.0);
    ExpectTfAndXlaOutputsAreClose(OpTestBuilder("LRNGrad")
                                      .Input(input_grads)
                                      .Input(input_image)
                                      .Input(output_image)
                                      .Attr("T", DT_FLOAT)
                                      .Attr("depth_radius", radius(generator()))
                                      .Attr("bias", coeff(generator()))
                                      .Attr("alpha", coeff(generator()))
                                      .Attr("beta", coeff(generator())));
  });
}

TEST_F(OpTest, MatMul) {
  Repeatedly([this]() {
    int64 x = RandomDim();
    int64 y = RandomDim();
    int64 z = RandomDim();

    ExpectTfAndXlaOutputsAreClose(OpTestBuilder("MatMul")
                                      .Input(RandomTensor(DT_FLOAT, {x, y}))
                                      .Input(RandomTensor(DT_FLOAT, {y, z}))
                                      .Attr("T", DT_FLOAT));

    ExpectTfAndXlaOutputsAreClose(OpTestBuilder("MatMul")
                                      .Input(RandomTensor(DT_FLOAT, {y, x}))
                                      .Input(RandomTensor(DT_FLOAT, {y, z}))
                                      .Attr("T", DT_FLOAT)
                                      .Attr("transpose_a", true));

    ExpectTfAndXlaOutputsAreClose(OpTestBuilder("MatMul")
                                      .Input(RandomTensor(DT_FLOAT, {x, y}))
                                      .Input(RandomTensor(DT_FLOAT, {z, y}))
                                      .Attr("T", DT_FLOAT)
                                      .Attr("transpose_b", true));

    ExpectTfAndXlaOutputsAreClose(OpTestBuilder("MatMul")
                                      .Input(RandomTensor(DT_FLOAT, {y, x}))
                                      .Input(RandomTensor(DT_FLOAT, {z, y}))
                                      .Attr("T", DT_FLOAT)
                                      .Attr("transpose_a", true)
                                      .Attr("transpose_b", true));
  });
}

TEST_F(OpTest, MatrixDiag) {
  Repeatedly([this]() {
    DataType type = Choose<DataType>({DT_BOOL, DT_INT32, DT_FLOAT});
    ExpectTfAndXlaOutputsAreClose(OpTestBuilder("MatrixDiag")
                                      .Input(RandomTensor(type, RandomDims(1)))
                                      .Attr("T", type));
  });
}

TEST_F(OpTest, MatrixDiagPart) {
  Repeatedly([this]() {
    DataType type = Choose<DataType>({DT_BOOL, DT_INT32, DT_FLOAT});
    ExpectTfAndXlaOutputsAreClose(OpTestBuilder("MatrixDiagPart")
                                      .Input(RandomTensor(type, RandomDims(2)))
                                      .Attr("T", type));
  });
}

TEST_F(OpTest, Max) {
  Repeatedly([this]() {
    DataType type = Choose<DataType>({DT_INT32, DT_FLOAT});
    Tensor data = RandomTensor(type);
    Tensor indices = RandomReductionIndices(data.dims());
    bool keep_dims = Choose<bool>({false, true});
    ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("Max").Input(data).Input(indices).Attr("T", type).Attr(
            "keep_dims", keep_dims));
  });
}

TEST_F(OpTest, Maximum) {
  Repeatedly([this]() {
    DataType type = Choose<DataType>({DT_INT32, DT_FLOAT});
    auto dims = BroadcastableDims();
    ExpectTfAndXlaOutputsAreClose(OpTestBuilder("Maximum")
                                      .Input(RandomTensor(type, dims.first))
                                      .Input(RandomTensor(type, dims.second))
                                      .Attr("T", type));
  });
}

TEST_F(OpTest, MaxPool) {
  Repeatedly([this]() {
    std::uniform_int_distribution<int> random_int(1, 5);
    int kernel_rows = random_int(generator()),
        kernel_cols = random_int(generator());
    int stride_rows = random_int(generator()),
        stride_cols = random_int(generator());
    string padding = Choose<string>({"SAME", "VALID"});
    ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("MaxPool")
            .Input(
                RandomTensor(DT_FLOAT, {RandomDim(1), RandomDim(kernel_rows),
                                        RandomDim(kernel_cols), RandomDim(1)}))
            .Attr("T", DT_FLOAT)
            .Attr("ksize", {1, kernel_rows, kernel_cols, 1})
            .Attr("strides", {1, stride_rows, stride_cols, 1})
            .Attr("padding", padding)
            .Attr("data_format", "NHWC"));
  });
  // TODO(phawkins): test NCHW format (not supported by CPU)
}

TEST_F(OpTest, Mean) {
  Repeatedly([this]() {
    DataType type = Choose<DataType>({DT_INT32, DT_FLOAT});
    // TODO(phawkins): CPU and XLA differ output for reducing across a
    // size-0 dimension (nan vs 0). For now, require size >= 1.
    Tensor data = RandomTensor(type, RandomDims(0, kDefaultMaxRank, 1));
    Tensor indices = RandomReductionIndices(data.dims());
    bool keep_dims = Choose<bool>({false, true});
    ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("Mean").Input(data).Input(indices).Attr("T", type).Attr(
            "keep_dims", keep_dims));
  });
}

TEST_F(OpTest, Min) {
  Repeatedly([this]() {
    DataType type = Choose<DataType>({DT_INT32, DT_FLOAT});
    Tensor data = RandomTensor(type);
    Tensor indices = RandomReductionIndices(data.dims());
    bool keep_dims = Choose<bool>({false, true});
    ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("Min").Input(data).Input(indices).Attr("T", type).Attr(
            "keep_dims", keep_dims));
  });
}

TEST_F(OpTest, Minimum) {
  Repeatedly([this]() {
    DataType type = Choose<DataType>({DT_INT32, DT_FLOAT});
    auto dims = BroadcastableDims();
    ExpectTfAndXlaOutputsAreClose(OpTestBuilder("Minimum")
                                      .Input(RandomTensor(type, dims.first))
                                      .Input(RandomTensor(type, dims.second))
                                      .Attr("T", type));
  });
}

TEST_F(OpTest, Mod) {
  Repeatedly([this]() {
    auto dims = BroadcastableDims();
    ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("Mod")
            .Input(RandomTensor(DT_INT32, dims.first))
            .Input(RandomTensor(DT_INT32, dims.second))
            .Attr("T", DT_INT32));
  });
}

TEST_F(OpTest, Mul) {
  Repeatedly([this]() {
    DataType type = Choose<DataType>({DT_INT32, DT_FLOAT});
    auto dims = BroadcastableDims();
    ExpectTfAndXlaOutputsAreClose(OpTestBuilder("Mul")
                                      .Input(RandomTensor(type, dims.first))
                                      .Input(RandomTensor(type, dims.second))
                                      .Attr("T", type));
  });
}

TEST_F(OpTest, Neg) {
  Repeatedly([this]() {
    DataType type = Choose<DataType>({DT_INT32, DT_FLOAT});
    ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("Neg").Input(RandomTensor(type)).Attr("T", type));
  });
}

TEST_F(OpTest, NotEqual) {
  Repeatedly([this]() {
    DataType type = Choose<DataType>({DT_INT32, DT_FLOAT});
    auto dims = BroadcastableDims();
    ExpectTfAndXlaOutputsAreClose(OpTestBuilder("NotEqual")
                                      .Input(RandomTensor(type, dims.first))
                                      .Input(RandomTensor(type, dims.second))
                                      .Attr("T", type));
  });
}

TEST_F(OpTest, Pack) {
  Repeatedly([this]() {
    DataType type = Choose<DataType>(kAllXlaTypes);
    int n = std::uniform_int_distribution<int>(1, 5)(generator());

    std::vector<int64> dims = RandomDims();
    int num_dims = dims.size();
    int axis = std::uniform_int_distribution<int32>(-num_dims - 1,
                                                    num_dims)(generator());

    OpTestBuilder builder("Pack");
    builder.Attr("T", type);
    builder.Attr("N", n);
    builder.Attr("axis", axis);
    for (int i = 0; i < n; ++i) {
      builder.Input(RandomTensor(type, dims));
    }
    ExpectTfAndXlaOutputsAreClose(builder);
  });
}

// TODO(b/31741898): crashes on GPU.
TEST_F(OpTest, Pad) {
  Repeatedly([this]() {
    DataType type = Choose<DataType>(kAllXlaTypes);
    Tensor t = RandomTensor(type);

    // TODO(b/31741996): re-enable DT_INT64 when bug is fixed.
    // DataType tpaddings = Choose<DataType>({DT_INT32, DT_INT64});
    DataType tpaddings = DT_INT32;
    std::vector<int64> paddings_vec;
    std::uniform_int_distribution<int> distribution(0, 7);
    for (int i = 0; i < t.dims(); ++i) {
      paddings_vec.push_back(distribution(generator()));
      paddings_vec.push_back(distribution(generator()));
    }
    Tensor paddings;
    CHECK(paddings.CopyFrom(AsIntTensor(tpaddings, paddings_vec),
                            TensorShape({t.dims(), 2})));
    ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("Pad").Input(t).Input(paddings).Attr("T", type).Attr(
            "Tpaddings", tpaddings));
  });
}

TEST_F(OpTest, Pow) {
  // TODO(phawkins): Feeding large DT_INT32 values to Pow() leads to
  // nontermination.
  Repeatedly([this]() {
    auto dims = BroadcastableDims();
    ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("Pow")
            .Input(RandomTensor(DT_FLOAT, dims.first))
            .Input(RandomTensor(DT_FLOAT, dims.second))
            .Attr("T", DT_FLOAT));
  });
}

TEST_F(OpTest, Prod) {
  Repeatedly([this]() {
    DataType type = Choose<DataType>({DT_INT32, DT_FLOAT});
    Tensor data = RandomTensor(type);
    Tensor indices = RandomReductionIndices(data.dims());
    bool keep_dims = Choose<bool>({false, true});
    ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("Prod").Input(data).Input(indices).Attr("T", type).Attr(
            "keep_dims", keep_dims));
  });
}

TEST_F(OpTest, Range) {
  Repeatedly([this]() {
    auto ToScalar = [](DataType type, int x) {
      if (type == DT_INT32) return test::AsScalar<int32>(x);
      if (type == DT_INT64) return test::AsScalar<int64>(x);
      if (type == DT_FLOAT) return test::AsScalar<float>(x);
      if (type == DT_DOUBLE) return test::AsScalar<double>(x);
      LOG(FATAL) << "Unknown type " << DataTypeString(type);
    };
    std::uniform_int_distribution<int> distribution(-50, 50);
    DataType tidx = Choose<DataType>({DT_INT32, DT_INT64, DT_FLOAT, DT_DOUBLE});
    ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("Range")
            .Input(ToScalar(tidx, distribution(generator())))
            .Input(ToScalar(tidx, distribution(generator())))
            .Input(ToScalar(tidx, distribution(generator())))
            .Attr("Tidx", tidx));
  });
}

TEST_F(OpTest, Rank) {
  Repeatedly([this]() {
    DataType type = Choose<DataType>({DT_INT32, DT_FLOAT});
    ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("Rank").Input(RandomTensor(type)).Attr("T", type));
  });
}

TEST_F(OpTest, RealDiv) {
  Repeatedly([this]() {
    DataType type = DT_FLOAT;
    auto dims = BroadcastableDims();
    ExpectTfAndXlaOutputsAreClose(OpTestBuilder("RealDiv")
                                      .Input(RandomTensor(type, dims.first))
                                      .Input(RandomTensor(type, dims.second))
                                      .Attr("T", type));
  });
}

TEST_F(OpTest, Relu) {
  Repeatedly([this]() {
    ExpectTfAndXlaOutputsAreClose(OpTestBuilder("Relu")
                                      .Input(RandomTensor(DT_FLOAT))
                                      .Attr("T", DT_FLOAT));
  });
}

TEST_F(OpTest, Relu6) {
  Repeatedly([this]() {
    ExpectTfAndXlaOutputsAreClose(OpTestBuilder("Relu6")
                                      .Input(RandomTensor(DT_FLOAT))
                                      .Attr("T", DT_FLOAT));
  });
}

TEST_F(OpTest, Relu6Grad) {
  Repeatedly([this]() {
    auto dims = RandomDims(1);
    ExpectTfAndXlaOutputsAreClose(OpTestBuilder("Relu6Grad")
                                      .Input(RandomTensor(DT_FLOAT, dims))
                                      .Input(RandomTensor(DT_FLOAT, dims))
                                      .Attr("T", DT_FLOAT));
  });
}

TEST_F(OpTest, ReluGrad) {
  Repeatedly([this]() {
    auto dims = RandomDims(1);
    ExpectTfAndXlaOutputsAreClose(OpTestBuilder("ReluGrad")
                                      .Input(RandomTensor(DT_FLOAT, dims))
                                      .Input(RandomTensor(DT_FLOAT, dims))
                                      .Attr("T", DT_FLOAT));
  });
}

TEST_F(OpTest, Reshape) {
  Repeatedly([this]() {
    DataType type = Choose<DataType>(kAllXlaTypes);
    std::vector<int64> dims = RandomDims();
    std::bernoulli_distribution random_bool;
    std::vector<int64> dims_before, dims_after;
    for (std::vector<int64>* out : {&dims_before, &dims_after}) {
      std::shuffle(dims.begin(), dims.end(), generator());
      for (int64 dim : dims) {
        // Either add the dimension as a new dimension or merge it with the
        // previous dimension.
        if (out->empty() || random_bool(generator())) {
          out->push_back(dim);
        } else {
          out->back() *= dim;
        }
      }
    }
    Tensor data = RandomTensor(type, dims_before);
    ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("Reshape")
            .Input(data)
            .Input(test::AsTensor<int32>(
                std::vector<int32>(dims_after.begin(), dims_after.end())))
            .Attr("T", type));
  });
}

TEST_F(OpTest, Rsqrt) {
  Repeatedly([this]() {
    ExpectTfAndXlaOutputsAreClose(OpTestBuilder("Rsqrt")
                                      .Input(RandomTensor(DT_FLOAT))
                                      .Attr("T", DT_FLOAT));
  });
}

TEST_F(OpTest, RsqrtGrad) {
  Repeatedly([this]() {
    auto dims = RandomDims();
    ExpectTfAndXlaOutputsAreClose(OpTestBuilder("RsqrtGrad")
                                      .Input(RandomTensor(DT_FLOAT, dims))
                                      .Input(RandomTensor(DT_FLOAT, dims))
                                      .Attr("T", DT_FLOAT));
  });
}

TEST_F(OpTest, Shape) {
  Repeatedly([this]() {
    DataType type = Choose<DataType>(kAllXlaTypes);
    ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("Shape").Input(RandomTensor(type)).Attr("T", type));
  });
}

TEST_F(OpTest, ShapeN) {
  Repeatedly([this]() {
    DataType type = Choose<DataType>(kAllXlaTypes);
    int n = std::uniform_int_distribution<int>(1, 5)(generator());
    OpTestBuilder builder("ShapeN");
    builder.Attr("T", type);
    builder.Attr("N", n);
    for (int i = 0; i < n; ++i) {
      builder.Input(RandomTensor(type));
    }
    ExpectTfAndXlaOutputsAreClose(builder);
  });
}

TEST_F(OpTest, Sigmoid) {
  Repeatedly([this]() {
    ExpectTfAndXlaOutputsAreClose(OpTestBuilder("Sigmoid")
                                      .Input(RandomTensor(DT_FLOAT))
                                      .Attr("T", DT_FLOAT));
  });
}

TEST_F(OpTest, SigmoidGrad) {
  Repeatedly([this]() {
    auto dims = RandomDims();
    ExpectTfAndXlaOutputsAreClose(OpTestBuilder("SigmoidGrad")
                                      .Input(RandomTensor(DT_FLOAT, dims))
                                      .Input(RandomTensor(DT_FLOAT, dims))
                                      .Attr("T", DT_FLOAT));
  });
}

TEST_F(OpTest, Sign) {
  Repeatedly([this]() {
    DataType type = Choose<DataType>({DT_INT32, DT_FLOAT});
    ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("Sign").Input(RandomTensor(type)).Attr("T", type));
  });
}

TEST_F(OpTest, Size) {
  Repeatedly([this]() {
    DataType type = Choose<DataType>({DT_INT32, DT_FLOAT});
    ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("Size").Input(RandomTensor(type)).Attr("T", type));
  });
}

TEST_F(OpTest, Slice) {
  Repeatedly([this]() {
    DataType type = Choose<DataType>(kAllXlaTypes);
    Tensor data = RandomTensor(type);

    std::vector<int32> begin(data.dims()), size(data.dims());
    for (int i = 0; i < data.dims(); ++i) {
      begin[i] = std::uniform_int_distribution<int32>(
          0, data.dim_size(i))(generator());
      size[i] = std::uniform_int_distribution<int32>(
          -1, data.dim_size(i) - begin[i])(generator());
    }
    ExpectTfAndXlaOutputsAreClose(OpTestBuilder("Slice")
                                      .Input(data)
                                      .Input(test::AsTensor<int32>(begin))
                                      .Input(test::AsTensor<int32>(size))
                                      .Attr("T", type)
                                      .Attr("Index", DT_INT32));
  });
}

TEST_F(OpTest, Softmax) {
  Repeatedly([this]() {
    ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("Softmax")
            .Input(RandomTensor(DT_FLOAT, RandomDims(2, 2)))
            .Attr("T", DT_FLOAT));
  });
}

TEST_F(OpTest, Split) {
  Repeatedly([this]() {
    DataType type = Choose<DataType>(kAllXlaTypes);
    std::vector<int64> dims = RandomDims(1);
    std::uniform_int_distribution<int> ud;
    int32 dim = std::uniform_int_distribution<int32>(
        0, static_cast<int32>(dims.size()) - 1)(generator());
    int n = std::uniform_int_distribution<int>(1, 5)(generator());
    // Ensure 'dim' is evenly divisible by 'n'.
    dims[dim] /= n;
    dims[dim] *= n;
    ExpectTfAndXlaOutputsAreClose(OpTestBuilder("Split")
                                      .Input(test::AsScalar<int32>(dim))
                                      .Input(RandomTensor(type, dims))
                                      .Attr("T", type)
                                      .Attr("num_split", n));
  });
}

TEST_F(OpTest, Softplus) {
  Repeatedly([this]() {
    ExpectTfAndXlaOutputsAreClose(OpTestBuilder("Softplus")
                                      .Input(RandomTensor(DT_FLOAT))
                                      .Attr("T", DT_FLOAT));
  });
}

TEST_F(OpTest, SoftplusGrad) {
  Repeatedly([this]() {
    std::vector<int64> dims = RandomDims();
    ExpectTfAndXlaOutputsAreClose(OpTestBuilder("SoftplusGrad")
                                      .Input(RandomTensor(DT_FLOAT, dims))
                                      .Input(RandomTensor(DT_FLOAT, dims))
                                      .Attr("T", DT_FLOAT));
  });
}

TEST_F(OpTest, SparseMatMul) {
  Repeatedly([this]() {
    int64 x = RandomDim();
    int64 y = RandomDim();
    int64 z = RandomDim();

    ExpectTfAndXlaOutputsAreClose(OpTestBuilder("SparseMatMul")
                                      .Input(RandomTensor(DT_FLOAT, {x, y}))
                                      .Input(RandomTensor(DT_FLOAT, {y, z}))
                                      .Attr("Ta", DT_FLOAT)
                                      .Attr("Tb", DT_FLOAT));

    ExpectTfAndXlaOutputsAreClose(OpTestBuilder("SparseMatMul")
                                      .Input(RandomTensor(DT_FLOAT, {y, x}))
                                      .Input(RandomTensor(DT_FLOAT, {y, z}))
                                      .Attr("Ta", DT_FLOAT)
                                      .Attr("Tb", DT_FLOAT)
                                      .Attr("transpose_a", true));

    ExpectTfAndXlaOutputsAreClose(OpTestBuilder("SparseMatMul")
                                      .Input(RandomTensor(DT_FLOAT, {x, y}))
                                      .Input(RandomTensor(DT_FLOAT, {z, y}))
                                      .Attr("Ta", DT_FLOAT)
                                      .Attr("Tb", DT_FLOAT)
                                      .Attr("transpose_b", true));

    ExpectTfAndXlaOutputsAreClose(OpTestBuilder("SparseMatMul")
                                      .Input(RandomTensor(DT_FLOAT, {y, x}))
                                      .Input(RandomTensor(DT_FLOAT, {z, y}))
                                      .Attr("Ta", DT_FLOAT)
                                      .Attr("Tb", DT_FLOAT)
                                      .Attr("transpose_a", true)
                                      .Attr("transpose_b", true));
  });
}

TEST_F(OpTest, Sqrt) {
  Repeatedly([this]() {
    ExpectTfAndXlaOutputsAreClose(OpTestBuilder("Sqrt")
                                      .Input(RandomTensor(DT_FLOAT))
                                      .Attr("T", DT_FLOAT));
  });
}

TEST_F(OpTest, SquaredDifference) {
  Repeatedly([this]() {
    auto dims = BroadcastableDims();
    ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("SquaredDifference")
            .Input(RandomTensor(DT_FLOAT, dims.first))
            .Input(RandomTensor(DT_FLOAT, dims.second))
            .Attr("T", DT_FLOAT));
  });
}

TEST_F(OpTest, Square) {
  Repeatedly([this]() {
    DataType type = Choose<DataType>({DT_INT32, DT_FLOAT});
    ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("Square").Input(RandomTensor(type)).Attr("T", type));
  });
}

TEST_F(OpTest, Squeeze) {
  Repeatedly([this]() {
    DataType type = Choose<DataType>(kAllXlaTypes);
    Tensor t = RandomTensor(type, RandomDims(0, kDefaultMaxRank, 0, 5));
    std::bernoulli_distribution random_bool;
    std::vector<int> squeeze_dims;
    for (int i = 0; i < t.dims(); ++i) {
      if (t.dim_size(i) == 1 && random_bool(generator())) {
        squeeze_dims.push_back(i);
      }
    }
    ExpectTfAndXlaOutputsAreClose(OpTestBuilder("Squeeze")
                                      .Input(t)
                                      .Attr("squeeze_dims", squeeze_dims)
                                      .Attr("T", type));
  });
}

TEST_F(OpTest, Sub) {
  Repeatedly([this]() {
    DataType type = Choose<DataType>({DT_INT32, DT_FLOAT});
    auto dims = BroadcastableDims();
    ExpectTfAndXlaOutputsAreClose(OpTestBuilder("Sub")
                                      .Input(RandomTensor(type, dims.first))
                                      .Input(RandomTensor(type, dims.second))
                                      .Attr("T", type));
  });
}

TEST_F(OpTest, Sum) {
  Repeatedly([this]() {
    DataType type = Choose<DataType>({DT_INT32, DT_FLOAT});
    Tensor data = RandomTensor(type);
    Tensor indices = RandomReductionIndices(data.dims());
    bool keep_dims = Choose<bool>({false, true});
    ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("Sum").Input(data).Input(indices).Attr("T", type).Attr(
            "keep_dims", keep_dims));
  });
}

TEST_F(OpTest, StridedSlice) {
  Repeatedly([this]() {
    DataType type = Choose<DataType>(kAllXlaTypes);
    Tensor data = RandomTensor(type);

    std::vector<int32> begin(data.dims()), end(data.dims());
    std::vector<int32> strides(data.dims());
    for (int i = 0; i < data.dims(); ++i) {
      begin[i] = std::uniform_int_distribution<int32>(
          -2 * data.dim_size(i), 2 * data.dim_size(i))(generator());
      end[i] = std::uniform_int_distribution<int32>(
          -2 * data.dim_size(i), 2 * data.dim_size(i))(generator());
      // TODO(b/31360685): support strides other than 1 or -1
      strides[i] = std::bernoulli_distribution()(generator()) ? 1 : -1;
    }
    int64 max_bitmask = (1LL << data.dims()) - 1;
    std::uniform_int_distribution<int64> bitmask_distribution(0, max_bitmask);
    int64 begin_mask = bitmask_distribution(generator());
    int64 end_mask = bitmask_distribution(generator());

    // Create a ellipsis bitmask with at most one 1 bit set.
    int64 ellipsis_mask = 0;
    if (data.dims() > 0 && std::bernoulli_distribution()(generator())) {
      int ellipsis_pos =
          std::uniform_int_distribution<int>(0, data.dims() - 1)(generator());
      ellipsis_mask = 1LL << ellipsis_pos;
    }

    int64 new_axis_mask = bitmask_distribution(generator());
    int64 shrink_axis_mask = bitmask_distribution(generator());
    ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("StridedSlice")
            .Input(data)
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
  Repeatedly([this]() {
    DataType type = Choose<DataType>(kAllXlaTypes);

    // Dimensions of the forward input.
    std::vector<int64> dims = RandomDims();

    std::vector<int64> begin(dims.size()), end(dims.size());
    std::vector<int64> strides(dims.size());
    for (int i = 0; i < dims.size(); ++i) {
      begin[i] = std::uniform_int_distribution<int64>(-2 * dims[i],
                                                      2 * dims[i])(generator());
      end[i] = std::uniform_int_distribution<int64>(-2 * dims[i],
                                                    2 * dims[i])(generator());
      strides[i] = std::uniform_int_distribution<int64>(
          -2 * dims[i], 2 * dims[i])(generator());
    }
    int64 max_bitmask = (1LL << dims.size()) - 1;
    std::uniform_int_distribution<int64> bitmask_distribution(0, max_bitmask);
    int64 begin_mask = bitmask_distribution(generator());
    int64 end_mask = bitmask_distribution(generator());

    // Create a ellipsis bitmask with at most one 1 bit set.
    int64 ellipsis_mask = 0;
    if (!dims.empty() && std::bernoulli_distribution()(generator())) {
      int ellipsis_pos =
          std::uniform_int_distribution<int>(0, dims.size() - 1)(generator());
      ellipsis_mask = 1LL << ellipsis_pos;
    }

    int64 new_axis_mask = bitmask_distribution(generator());
    int64 shrink_axis_mask = bitmask_distribution(generator());

    // TODO(phawkins): use shape inference for the forward op to compute the
    // gradient shape for the backward op. At present, there is a low
    // probability of the golden op succeeding.
    ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("StridedSliceGrad")
            .Input(test::AsTensor<int64>(dims))
            .Input(test::AsTensor<int64>(begin))
            .Input(test::AsTensor<int64>(end))
            .Input(test::AsTensor<int64>(strides))
            .Input(RandomTensor(type, RandomDims(1)))
            .Attr("T", type)
            .Attr("Index", DT_INT64)
            .Attr("begin_mask", begin_mask)
            .Attr("end_mask", end_mask)
            .Attr("ellipsis_mask", ellipsis_mask)
            .Attr("new_axis_mask", new_axis_mask)
            .Attr("shrink_axis_mask", shrink_axis_mask));
  });
}

TEST_F(OpTest, Tanh) {
  Repeatedly([this]() {
    ExpectTfAndXlaOutputsAreClose(OpTestBuilder("Tanh")
                                      .Input(RandomTensor(DT_FLOAT))
                                      .Attr("T", DT_FLOAT));
  });
}

TEST_F(OpTest, TanhGrad) {
  Repeatedly([this]() {
    auto dims = RandomDims();
    ExpectTfAndXlaOutputsAreClose(OpTestBuilder("TanhGrad")
                                      .Input(RandomTensor(DT_FLOAT, dims))
                                      .Input(RandomTensor(DT_FLOAT, dims))
                                      .Attr("T", DT_FLOAT));
  });
}

TEST_F(OpTest, Tile) {
  Repeatedly([this]() {
    DataType type = Choose<DataType>(kAllXlaTypes);
    Tensor t = RandomTensor(type, RandomDims(1));
    std::vector<int32> multiples(t.dims());
    for (int i = 0; i < t.dims(); ++i) {
      multiples[i] = std::uniform_int_distribution<int>(1, 3)(generator());
    }
    ExpectTfAndXlaOutputsAreClose(OpTestBuilder("Tile")
                                      .Input(t)
                                      .Input(test::AsTensor<int32>(multiples))
                                      .Attr("T", type));
  });
}

TEST_F(OpTest, Transpose) {
  Repeatedly([this]() {
    DataType type = Choose<DataType>(kAllXlaTypes);
    Tensor data = RandomTensor(type);
    std::vector<int32> perm(data.dims());
    std::iota(perm.begin(), perm.end(), 0);
    std::shuffle(perm.begin(), perm.end(), generator());
    ExpectTfAndXlaOutputsAreClose(OpTestBuilder("Transpose")
                                      .Input(data)
                                      .Input(test::AsTensor<int32>(perm))
                                      .Attr("T", type));
  });
}

TEST_F(OpTest, TruncateDiv) {
  Repeatedly([this]() {
    DataType type = DT_INT32;
    auto dims = BroadcastableDims();
    ExpectTfAndXlaOutputsAreClose(OpTestBuilder("TruncateDiv")
                                      .Input(RandomTensor(type, dims.first))
                                      .Input(RandomTensor(type, dims.second))
                                      .Attr("T", type));
  });
}

TEST_F(OpTest, TruncateMod) {
  Repeatedly([this]() {
    DataType type = Choose<DataType>({DT_INT32, DT_FLOAT});
    auto dims = BroadcastableDims();
    ExpectTfAndXlaOutputsAreClose(OpTestBuilder("TruncateMod")
                                      .Input(RandomTensor(type, dims.first))
                                      .Input(RandomTensor(type, dims.second))
                                      .Attr("T", type));
  });
}

TEST_F(OpTest, ZerosLike) {
  Repeatedly([this]() {
    DataType type = Choose<DataType>({DT_INT32, DT_FLOAT});
    ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("ZerosLike").Input(RandomTensor(type)).Attr("T", type));
  });
}

}  // anonymous namespace
}  // namespace tensorflow

int main(int argc, char** argv) {
  tensorflow::tf_xla_test_device_ptr = new tensorflow::string("GPU:0");
  std::vector<tensorflow::Flag> flag_list = {
      tensorflow::Flag(
          "tf_xla_random_seed", &tensorflow::tf_xla_random_seed,
          "Random seed to use for XLA tests. <= 0 means choose a seed "
          "nondetermistically."),
      // TODO(phawkins): it might make more sense to run each test up to a
      // configurable time bound.
      tensorflow::Flag("tf_xla_test_repetitions",
                       &tensorflow::tf_xla_test_repetitions,
                       "Number of repetitions for each test."),
      tensorflow::Flag("tf_xla_test_device", tensorflow::tf_xla_test_device_ptr,
                       "Tensorflow device type to use for test"),
      tensorflow::Flag("tf_xla_test_use_jit", &tensorflow::tf_xla_test_use_jit,
                       "Use JIT compilation for the operator under test"),
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
  std::vector<tensorflow::Device*> devices;
  TF_CHECK_OK(tensorflow::DeviceFactory::AddDevices(
      tensorflow::SessionOptions(), "", &devices));
  tensorflow::DeviceMgr device_mgr(devices);

  tensorflow::Device* ignored;
  TF_QCHECK_OK(
      device_mgr.LookupDevice(*tensorflow::tf_xla_test_device_ptr, &ignored))
      << "Unknown test device (" << *tensorflow::tf_xla_test_device_ptr
      << "). Did you build in the right configuration (e.g., is CUDA enabled)?";

  return RUN_ALL_TESTS();
}
