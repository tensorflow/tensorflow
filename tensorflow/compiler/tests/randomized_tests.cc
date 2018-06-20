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
#include "tensorflow/core/util/tensor_format.h"

namespace tensorflow {
namespace {

// Command line flags: see main() below.
int64 tf_xla_random_seed = 0;
int32 tf_xla_test_repetitions = 20;
int64 tf_xla_max_tensor_size = 10000LL;
string* tf_xla_test_device_ptr;  // initial value set in main()
bool tf_xla_test_use_jit = true;

string LocalDeviceToFullDeviceName(const string& device) {
  return strings::StrCat("/job:localhost/replica:0/task:0/device:", device);
}

constexpr std::array<DataType, 5> kAllXlaTypes = {
    {DT_INT32, DT_FLOAT, DT_BOOL, DT_COMPLEX64, DT_INT64}};

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
  OpTestBuilder& RandomInput(DataType type, std::vector<int64> dims);

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
  Status BuildGraph(const string& name_prefix, const string& device,
                    bool use_jit, GraphDef* graphdef, NodeDef** test_node_def,
                    std::vector<string>* inputs,
                    std::vector<string>* outputs) const;

  struct InputDescription {
    Tensor tensor;

    DataType type = DT_INVALID;
    bool has_dims = false;
    std::vector<int64> dims;
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
                                          std::vector<int64> dims) {
  VLOG(1) << "Adding input: " << type << " " << TensorShape(dims).DebugString();
  InputDescription input;
  input.type = type;
  input.has_dims = true;
  input.dims = std::move(dims);
  inputs_.push_back(input);
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
  T Choose(gtl::ArraySlice<T> candidates);

  static constexpr int kDefaultMaxRank = 5;
  static constexpr int64 kDefaultMaxDimensionSize = 256LL;

  // Returns true if 'dims' have a size less than tf_xla_max_tensor_size.
  bool TensorSizeIsOk(gtl::ArraySlice<int64> dims);

  // Returns a random dimension size, in the range [min, max).
  int64 RandomDim(int64 min = 0, int64 max = kDefaultMaxDimensionSize);

  // Returns a random shape. The tensor has rank in the range [min_rank,
  // max_rank). Each dimension has size [min_size, max_size).
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

  struct WindowedSpatialDims {
    Padding padding;
    std::vector<int64> kernel_dims;
    std::vector<int64> stride_dims;
    std::vector<int64> input_dims;
    std::vector<int64> output_dims;
  };
  // Choose spatial dimensions for a windowed op such as pooling or convolution.
  WindowedSpatialDims ChooseWindowedSpatialDims(int num_spatial_dims);

  // Builds dimensions for a windowed op such as pooling or convolution,
  // including a batch and feature dimension.
  std::vector<int64> ImageDims(TensorFormat format, int batch, int feature,
                               const std::vector<int64>& spatial_dims);

  // Converts an int64 vector to an int32 vector.
  std::vector<int32> AsInt32s(const std::vector<int64>& int64s);

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
  int64 s = tf_xla_random_seed;
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
T OpTest::Choose(gtl::ArraySlice<T> candidates) {
  std::uniform_int_distribution<size_t> d(0, candidates.size() - 1);
  return candidates[d(generator())];
}

int64 OpTest::RandomDim(int64 min, int64 max) {
  std::uniform_int_distribution<int64> size_distribution(min, max - 1);
  return size_distribution(generator());
}

bool OpTest::TensorSizeIsOk(gtl::ArraySlice<int64> dims) {
  int64 size = 1LL;
  for (int64 dim : dims) {
    size *= dim;
  }
  return size < tf_xla_max_tensor_size;
}

std::vector<int64> OpTest::RandomDims(int min_rank, int max_rank,
                                      int64 min_size, int64 max_size) {
  CHECK_LE(0, min_rank);
  CHECK_LE(min_rank, max_rank);
  std::uniform_int_distribution<int> rank_distribution(min_rank, max_rank);
  int rank = rank_distribution(generator());
  std::vector<int64> dims(rank);
  // TODO(phawkins): too small a maximum tensor size could lead to an infinite
  // loop here.
  do {
    std::generate(dims.begin(), dims.end(), [this, min_size, max_size]() {
      return RandomDim(min_size, max_size);
    });
  } while (!TensorSizeIsOk(dims));
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
    case DT_COMPLEX64: {
      std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
      test::FillFn<complex64>(&tensor, [this, &distribution](int i) {
        return complex64(distribution(generator()), distribution(generator()));
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
      int64 pad_dummy;
      s = GetWindowedOutputSize(d.input_dims[i], d.kernel_dims[i],
                                d.stride_dims[i], d.padding, &d.output_dims[i],
                                &pad_dummy);
    } while (!s.ok());
  }
  return d;
}

std::vector<int64> OpTest::ImageDims(TensorFormat format, int batch,
                                     int feature,
                                     const std::vector<int64>& spatial_dims) {
  std::vector<int64> dims;
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

std::vector<int32> OpTest::AsInt32s(const std::vector<int64>& int64s) {
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
  return strings::StrCat(x);
}
template <>
string Str<complex64>(complex64 x) {
  return strings::StrCat("(", x.real(), ", ", x.imag(), ")");
}

template <typename T>
Status TensorsAreCloseImpl(const Tensor& x, const Tensor& y, double atol,
                           double rtol) {
  auto Tx = x.flat<T>();
  auto Ty = y.flat<T>();
  for (int i = 0; i < Tx.size(); ++i) {
    if (!IsClose(Tx(i), Ty(i), atol, rtol)) {
      return errors::InvalidArgument(strings::StrCat(
          i, "-th tensor element isn't close: ", Str(Tx(i)), " vs. ",
          Str(Ty(i)), ". x = ", x.DebugString(), "y = ", y.DebugString(),
          "atol = ", atol, " rtol = ", rtol,
          " tol = ", atol + rtol * Abs(Tx(i))));
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
    case DT_COMPLEX64:
      return TensorsAreCloseImpl<complex64>(a, b, atol, rtol);
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

OpTest::TestResult OpTest::ExpectTfAndXlaOutputsAreClose(
    const OpTestBuilder& builder, double atol, double rtol) {
  const std::vector<OpTestBuilder::InputDescription>& inputs = builder.inputs();
  std::vector<Tensor> input_tensors;
  input_tensors.reserve(inputs.size());
  for (const OpTestBuilder::InputDescription& input : inputs) {
    if (input.type == DT_INVALID) {
      input_tensors.push_back(input.tensor);
    } else {
      std::vector<int64> dims;
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
      input_tensors.push_back(RandomTensor(input.type, dims));
    }
    VLOG(1) << "Input: " << input_tensors.back().DebugString();
  }

  string cpu_device =
      LocalDeviceToFullDeviceName(strings::StrCat(DEVICE_CPU, ":0"));
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
      strings::StrCat("test", num_tests_, "_expected"), cpu_device,
      /* use_jit= */ false, &graph, /* test_node_def= */ nullptr,
      &expected_inputs, &expected_fetches);
  if (!status.ok()) {
    LOG(ERROR) << "Expected graph construction failed: " << status;
    return kFatalError;
  }

  NodeDef* node_def;
  status = builder.BuildGraph(strings::StrCat("test", num_tests_, "_test"),
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
      LOG(FATAL);
  }
}

TEST_F(OpTest, Abs) {
  Repeatedly([this]() {
    auto type = Choose<DataType>({DT_INT32, DT_FLOAT, DT_COMPLEX64});
    return ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("Abs").RandomInput(type).Attr("T", type));
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

TEST_F(OpTest, All) {
  Repeatedly([this]() {
    std::vector<int64> data_dims = RandomDims();
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
    std::vector<int64> data_dims = RandomDims();
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
    std::vector<int64> dims = RandomDims(1, 5, 1);
    int num_dims = dims.size();
    int reduce_dim =
        std::uniform_int_distribution<int32>(-num_dims, num_dims)(generator());
    return ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("ArgMax")
            .RandomInput(DT_FLOAT, dims)
            .Input(test::AsScalar<int32>(reduce_dim))
            .Attr("T", DT_FLOAT)
            .Attr("Tidx", DT_INT32)
            .Attr("output_type", DT_INT32));
  });
}

TEST_F(OpTest, ArgMin) {
  Repeatedly([this]() {
    std::vector<int64> dims = RandomDims(1, 5, 1);
    int num_dims = dims.size();
    int reduce_dim =
        std::uniform_int_distribution<int32>(-num_dims, num_dims)(generator());
    return ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("ArgMin")
            .RandomInput(DT_FLOAT, dims)
            .Input(test::AsScalar<int32>(reduce_dim))
            .Attr("T", DT_FLOAT)
            .Attr("Tidx", DT_INT32)
            .Attr("output_type", DT_INT32));
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
    std::vector<int64> dims = RandomDims(4, 4, 1);
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
  Repeatedly([this]() {
    std::uniform_int_distribution<int> random_int(1, 5);
    std::vector<int64> dims = RandomDims(5, 5, 1);

    std::vector<int64> input_dims, kernel_dims, stride_dims;
    for (int i = 0; i < 3; ++i) {
      kernel_dims.push_back(
          std::uniform_int_distribution<int>(1, dims[i])(generator()));
      input_dims.push_back(dims[i]);
      stride_dims.push_back(random_int(generator()));
    }
    int64 batch = dims[3];
    int64 feature = dims[4];

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
  Repeatedly([this]() {
    int batch = RandomDim(1), features = RandomDim(1);
    WindowedSpatialDims d = ChooseWindowedSpatialDims(2);
    std::vector<int32> input_dims =
        AsInt32s(ImageDims(FORMAT_NHWC, batch, features, d.input_dims));
    std::vector<int64> output_dims =
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
  Repeatedly([this]() {
    int batch = RandomDim(1), features = RandomDim(1);
    WindowedSpatialDims d = ChooseWindowedSpatialDims(3);
    std::vector<int32> input_dims =
        AsInt32s(ImageDims(FORMAT_NHWC, batch, features, d.input_dims));
    std::vector<int64> output_dims =
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
  Repeatedly([this]() {
    auto type = Choose<DataType>({DT_FLOAT, DT_COMPLEX64});
    std::vector<int64> output_dims = RandomDims(2, 5, 0, 7);
    int64 ndims = output_dims.size();
    int64 inner_dim = RandomDim();
    std::vector<int64> x_dims(output_dims), y_dims(output_dims);
    x_dims[ndims - 1] = inner_dim;
    y_dims[ndims - 2] = inner_dim;

    std::bernoulli_distribution random_bool;
    bool adj_x = random_bool(generator());
    bool adj_y = random_bool(generator());
    if (adj_x) {
      std::swap(x_dims[ndims - 1], x_dims[ndims - 2]);
    }
    if (adj_y) {
      std::swap(y_dims[ndims - 1], y_dims[ndims - 2]);
    }

    return ExpectTfAndXlaOutputsAreClose(OpTestBuilder("BatchMatMul")
                                             .RandomInput(type, x_dims)
                                             .RandomInput(type, y_dims)
                                             .Attr("T", type)
                                             .Attr("adj_x", adj_x)
                                             .Attr("adj_y", adj_y));
  });
}

TEST_F(OpTest, BatchToSpace) {
  Repeatedly([this]() {
    const int num_block_dims = 2;
    std::vector<int64> block_dims =
        RandomDims(num_block_dims, num_block_dims, 0, 5);
    int64 block_size = RandomDim(2, 5);

    std::vector<int64> input_dims(1 + num_block_dims + 1);
    input_dims[0] = RandomDim();
    for (int i = 0; i < num_block_dims; ++i) {
      input_dims[0] *= block_size;
      input_dims[1 + i] = block_dims[i];
    }
    input_dims[1 + num_block_dims] = RandomDim();

    std::vector<int64> crop_vals;
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
    std::vector<int64> block_dims = RandomDims(1, 3, 0, 5);
    int num_block_dims = block_dims.size();
    std::vector<int64> remaining_dims = RandomDims(0, 3);
    std::vector<int64> block_multipliers =
        RandomDims(block_dims.size(), block_dims.size(), 0, 4);

    std::vector<int64> input_dims(1 + num_block_dims + remaining_dims.size());
    input_dims[0] = RandomDim();
    for (int i = 0; i < num_block_dims; ++i) {
      input_dims[0] *= block_dims[i];
    }
    std::copy(block_multipliers.begin(), block_multipliers.end(),
              input_dims.begin() + 1);
    std::copy(remaining_dims.begin(), remaining_dims.end(),
              input_dims.begin() + 1 + num_block_dims);

    std::vector<int64> crop_vals;
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

TEST_F(OpTest, Ceil) {
  Repeatedly([this]() {
    return ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("Ceil").RandomInput(DT_FLOAT).Attr("T", DT_FLOAT));
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
  Repeatedly([this]() {
    auto type = Choose<DataType>(kAllXlaTypes);
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
      builder.RandomInput(type, shape);
    }
    return ExpectTfAndXlaOutputsAreClose(builder);
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

TEST_F(OpTest, FFT) {
  Repeatedly([this]() {
    std::vector<int64> dims = RandomDims(1, kDefaultMaxRank);
    return ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("FFT").RandomInput(DT_COMPLEX64, dims));
  });
}

TEST_F(OpTest, FFT2D) {
  Repeatedly([this]() {
    std::vector<int64> dims = RandomDims(2, kDefaultMaxRank);
    return ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("FFT2D").RandomInput(DT_COMPLEX64, dims));
  });
}

TEST_F(OpTest, FFT3D) {
  Repeatedly([this]() {
    std::vector<int64> dims = RandomDims(3, kDefaultMaxRank);
    return ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("FFT3D").RandomInput(DT_COMPLEX64, dims));
  });
}

TEST_F(OpTest, IFFT) {
  Repeatedly([this]() {
    std::vector<int64> dims = RandomDims(1, kDefaultMaxRank);
    return ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("IFFT").RandomInput(DT_COMPLEX64, dims));
  });
}

TEST_F(OpTest, IFFT2D) {
  Repeatedly([this]() {
    std::vector<int64> dims = RandomDims(2, kDefaultMaxRank);
    return ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("IFFT2D").RandomInput(DT_COMPLEX64, dims));
  });
}

TEST_F(OpTest, IFFT3D) {
  Repeatedly([this]() {
    std::vector<int64> dims = RandomDims(3, kDefaultMaxRank);
    return ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("IFFT3D").RandomInput(DT_COMPLEX64, dims));
  });
}

TEST_F(OpTest, RFFT) {
  Repeatedly([this]() {
    std::vector<int64> dims = RandomDims(1, kDefaultMaxRank, 3);
    Tensor fft_shape = test::AsTensor<int32>(AsInt32s({dims[dims.size() - 1]}));
    return ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("RFFT").RandomInput(DT_FLOAT, dims).Input(fft_shape));
  });
}

TEST_F(OpTest, RFFT2D) {
  Repeatedly([this]() {
    std::vector<int64> dims = RandomDims(2, kDefaultMaxRank, 3);
    Tensor fft_shape = test::AsTensor<int32>(
        AsInt32s({dims[dims.size() - 2], dims[dims.size() - 1]}));
    return ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("RFFT2D").RandomInput(DT_FLOAT, dims).Input(fft_shape));
  });
}

TEST_F(OpTest, RFFT3D) {
  Repeatedly([this]() {
    std::vector<int64> dims = RandomDims(3, kDefaultMaxRank, 3);
    Tensor fft_shape = test::AsTensor<int32>(AsInt32s(
        {dims[dims.size() - 3], dims[dims.size() - 2], dims[dims.size() - 1]}));
    return ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("RFFT3D").RandomInput(DT_FLOAT, dims).Input(fft_shape));
  });
}

TEST_F(OpTest, IRFFT) {
  Repeatedly([this]() {
    std::vector<int64> dims = RandomDims(1, kDefaultMaxRank, 3);
    int64 orig_size = dims[dims.size() - 1];
    dims[dims.size() - 1] = dims[dims.size() - 1] / 2 + 1;
    Tensor fft_shape = test::AsTensor<int32>(AsInt32s({orig_size}));
    return ExpectTfAndXlaOutputsAreClose(OpTestBuilder("IRFFT")
                                             .RandomInput(DT_COMPLEX64, dims)
                                             .Input(fft_shape));
  });
}

TEST_F(OpTest, IRFFT2D) {
  Repeatedly([this]() {
    std::vector<int64> dims = RandomDims(2, kDefaultMaxRank, 3);
    std::vector<int64> orig_size = {dims[dims.size() - 2],
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
    std::vector<int64> dims = RandomDims(3, kDefaultMaxRank, 3);
    std::vector<int64> orig_size = {
        dims[dims.size() - 3], dims[dims.size() - 2], dims[dims.size() - 1]};
    dims[dims.size() - 1] = dims[dims.size() - 1] / 2 + 1;
    Tensor fft_shape = test::AsTensor<int32>(AsInt32s({orig_size}));
    return ExpectTfAndXlaOutputsAreClose(OpTestBuilder("IRFFT3D")
                                             .RandomInput(DT_COMPLEX64, dims)
                                             .Input(fft_shape));
  });
}

TEST_F(OpTest, Conv2D) {
  Repeatedly([this]() {
    WindowedSpatialDims d = ChooseWindowedSpatialDims(2);
    std::uniform_int_distribution<int> random_int(1, 5);
    int features_in = random_int(generator());
    int features_out = random_int(generator());

    int64 batch = RandomDim();

    std::vector<int64> data_dims =
        ImageDims(FORMAT_NHWC, batch, features_in, d.input_dims);

    std::vector<int64> kernel_dims = {d.kernel_dims[0], d.kernel_dims[1],
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
    int32 batch = RandomDim();
    std::vector<int64> activations =
        ImageDims(FORMAT_NHWC, batch, features_in, d.input_dims);
    std::vector<int64> backprop =
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
    int32 batch = RandomDim();
    Tensor in_shape = test::AsTensor<int32>(
        AsInt32s(ImageDims(FORMAT_NHWC, batch, features_in, d.input_dims)));
    std::vector<int64> backprop =
        ImageDims(FORMAT_NHWC, batch, features_out, d.output_dims);
    std::vector<int64> kernel = {d.kernel_dims[0], d.kernel_dims[1],
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
  Repeatedly([this]() {
    WindowedSpatialDims d = ChooseWindowedSpatialDims(3);
    std::uniform_int_distribution<int> random_int(1, 5);
    int features_in = random_int(generator());
    int features_out = random_int(generator());
    std::vector<int64> data = {RandomDim(), d.input_dims[0], d.input_dims[1],
                               d.input_dims[2], features_in};

    std::vector<int64> kernel = {d.kernel_dims[0], d.kernel_dims[1],
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
  Repeatedly([this]() {
    WindowedSpatialDims d = ChooseWindowedSpatialDims(3);
    std::uniform_int_distribution<int> random_int(1, 5);
    int features_in = random_int(generator());
    int features_out = random_int(generator());
    int32 batch = RandomDim(1);
    std::vector<int64> activations =
        ImageDims(FORMAT_NHWC, batch, features_in, d.input_dims);
    std::vector<int64> backprop =
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
  Repeatedly([this]() {
    WindowedSpatialDims d = ChooseWindowedSpatialDims(3);
    std::uniform_int_distribution<int> random_int(1, 5);
    int features_in = random_int(generator());
    int features_out = random_int(generator());
    int32 batch = RandomDim(1);
    Tensor in_shape = test::AsTensor<int32>(
        AsInt32s(ImageDims(FORMAT_NHWC, batch, features_in, d.input_dims)));
    std::vector<int64> backprop =
        ImageDims(FORMAT_NHWC, batch, features_out, d.output_dims);
    std::vector<int64> kernel = {d.kernel_dims[0], d.kernel_dims[1],
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
    int64 block = RandomDim(2, 5);
    std::vector<int64> input_dims = RandomDims(4, 4);
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
  Repeatedly([this]() {
    WindowedSpatialDims d = ChooseWindowedSpatialDims(2);
    std::uniform_int_distribution<int> random_int(1, 5);
    int features_in = random_int(generator());
    int depth_multiplier = random_int(generator());
    std::vector<int64> input_dims = {RandomDim(), d.input_dims[0],
                                     d.input_dims[1], features_in};

    std::vector<int64> kernel_dims = {d.kernel_dims[0], d.kernel_dims[1],
                                      features_in, depth_multiplier};
    std::vector<int64> strides = ImageDims(FORMAT_NHWC, 1, 1, d.stride_dims);
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

TEST_F(OpTest, DepthwiseConv2DBackpropFilter) {
  Repeatedly([this]() {
    WindowedSpatialDims d = ChooseWindowedSpatialDims(2);
    std::uniform_int_distribution<int> random_int(1, 5);
    int features_in = random_int(generator());
    int depth_multiplier = random_int(generator());
    int32 batch = RandomDim();
    std::vector<int64> activations =
        ImageDims(FORMAT_NHWC, batch, features_in, d.input_dims);
    std::vector<int64> backprop = ImageDims(
        FORMAT_NHWC, batch, features_in * depth_multiplier, d.output_dims);
    Tensor kernel_shape = test::AsTensor<int32>(AsInt32s(
        {d.kernel_dims[0], d.kernel_dims[1], features_in, depth_multiplier}));
    std::vector<int64> strides = ImageDims(FORMAT_NHWC, 1, 1, d.stride_dims);
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
  Repeatedly([this]() {
    WindowedSpatialDims d = ChooseWindowedSpatialDims(2);
    std::uniform_int_distribution<int> random_int(1, 5);
    int features_in = random_int(generator());
    int depth_multiplier = random_int(generator());
    int32 batch = RandomDim();
    Tensor in_shape = test::AsTensor<int32>(
        AsInt32s(ImageDims(FORMAT_NHWC, batch, features_in, d.input_dims)));
    std::vector<int64> backprop = ImageDims(
        FORMAT_NHWC, batch, features_in * depth_multiplier, d.output_dims);
    std::vector<int64> kernel = {d.kernel_dims[0], d.kernel_dims[1],
                                 features_in, depth_multiplier};
    std::vector<int64> strides = ImageDims(FORMAT_NHWC, 1, 1, d.stride_dims);
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
    std::vector<int64> dims;
    // Diag causes a quadratic blowup in output size.
    int64 size;
    do {
      dims = RandomDims(1);
      size = TensorShape(dims).num_elements();
    } while (size * size < tf_xla_max_tensor_size);
    return ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("Diag").RandomInput(type, dims).Attr("T", type));
  });
}

TEST_F(OpTest, DiagPart) {
  Repeatedly([this]() {
    auto type = Choose<DataType>(kAllXlaTypes);
    auto dims = RandomDims(1, 3);
    // Duplicate the random dims.
    std::vector<int64> doubled_dims(dims.size() * 2);
    std::copy(dims.begin(), dims.end(), doubled_dims.begin());
    std::copy(dims.begin(), dims.end(), doubled_dims.begin() + dims.size());
    return ExpectTfAndXlaOutputsAreClose(OpTestBuilder("DiagPart")
                                             .RandomInput(type, doubled_dims)
                                             .Attr("T", type));
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

TEST_F(OpTest, DynamicStitch) {
  Repeatedly([this]() {
    auto type = Choose<DataType>(kAllXlaTypes);
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
          gtl::ArraySlice<int32>(indices, pos, shape.num_elements()), shape);
      builder.Input(t);
      pos += t.NumElements();
    }

    std::vector<int64> constant_dims = RandomDims(0, 3, 0, 5);
    for (int i = 0; i < n; ++i) {
      std::vector<int64> dims(index_dims[i].begin(), index_dims[i].end());
      std::copy(constant_dims.begin(), constant_dims.end(),
                std::back_inserter(dims));
      builder.RandomInput(type, dims);
    }
    return ExpectTfAndXlaOutputsAreClose(builder);
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
    std::vector<int64> in_dims = RandomDims();
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
    std::vector<int64> dims = RandomDims();
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

TEST_F(OpTest, Imag) {
  Repeatedly([this]() {
    return ExpectTfAndXlaOutputsAreClose(OpTestBuilder("Imag")
                                             .RandomInput(DT_COMPLEX64)
                                             .Attr("T", DT_COMPLEX64));
  });
}

TEST_F(OpTest, Invert) {
  Repeatedly([this]() {
    DataType type = DT_INT32;
    return ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("Invert").RandomInput(type).Attr("T", type));
  });
}

TEST_F(OpTest, L2Loss) {
  Repeatedly([this]() {
    DataType type = DT_FLOAT;
    return ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("L2Loss").RandomInput(type).Attr("T", type));
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

TEST_F(OpTest, LinSpace) {
  Repeatedly([this]() {
    auto ToScalar = [](DataType type, int x) {
      if (type == DT_INT32) return test::AsScalar<int32>(x);
      return test::AsScalar<int64>(x);
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
    std::vector<int64> data_dims = RandomDims(4, 4, 1, 8);
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
    std::vector<int64> dims = RandomDims(4, 4, 1, 8);
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
    int64 x = RandomDim();
    int64 y = RandomDim();
    int64 z = RandomDim();

    std::vector<int64> a_dims = {x, y};
    std::vector<int64> b_dims = {y, z};

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

TEST_F(OpTest, MatrixDiag) {
  Repeatedly([this]() {
    auto type = Choose<DataType>({DT_INT32, DT_FLOAT, DT_COMPLEX64});
    return ExpectTfAndXlaOutputsAreClose(OpTestBuilder("MatrixDiag")
                                             .RandomInput(type, RandomDims(1))
                                             .Attr("T", type));
  });
}

TEST_F(OpTest, MatrixDiagPart) {
  Repeatedly([this]() {
    auto type = Choose<DataType>({DT_INT32, DT_FLOAT, DT_COMPLEX64});
    return ExpectTfAndXlaOutputsAreClose(OpTestBuilder("MatrixDiagPart")
                                             .RandomInput(type, RandomDims(2))
                                             .Attr("T", type));
  });
}

TEST_F(OpTest, Max) {
  Repeatedly([this]() {
    auto type = Choose<DataType>({DT_INT32, DT_FLOAT});
    std::vector<int64> data_dims = RandomDims();
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
    std::vector<int64> dims = RandomDims(4, 4, 1);
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
  Repeatedly([this]() {
    std::uniform_int_distribution<int> random_int(1, 5);
    std::vector<int64> dims = RandomDims(5, 5, 1);

    std::vector<int64> input_dims, kernel_dims, stride_dims;
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
    int64 batch = dims[3];
    int64 feature = dims[4];

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
    std::vector<int64> data_dims = RandomDims(0, kDefaultMaxRank, 1);
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
    std::vector<int64> data_dims = RandomDims();
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

TEST_F(OpTest, Neg) {
  Repeatedly([this]() {
    auto type = Choose<DataType>({DT_INT32, DT_FLOAT, DT_COMPLEX64});
    return ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("Neg").RandomInput(type).Attr("T", type));
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

    std::vector<int64> dims = RandomDims();
    int num_dims = dims.size();

    int32 depth = RandomDim();

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

    std::vector<int64> dims = RandomDims();
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

// TODO(b/31741898): crashes on GPU.
TEST_F(OpTest, Pad) {
  Repeatedly([this]() {
    auto type = Choose<DataType>(kAllXlaTypes);
    std::vector<int64> t_dims = RandomDims();

    // TODO(b/31741996): re-enable DT_INT64 when bug is fixed.
    // DataType tpaddings = Choose<DataType>({DT_INT32, DT_INT64});
    DataType tpaddings = DT_INT32;
    std::vector<int64> paddings_vec;
    std::uniform_int_distribution<int> distribution(0, 7);
    for (int i = 0; i < t_dims.size(); ++i) {
      paddings_vec.push_back(distribution(generator()));
      paddings_vec.push_back(distribution(generator()));
    }
    Tensor paddings;
    CHECK(
        paddings.CopyFrom(AsIntTensor(tpaddings, paddings_vec),
                          TensorShape({static_cast<int64>(t_dims.size()), 2})));
    return ExpectTfAndXlaOutputsAreClose(OpTestBuilder("Pad")
                                             .RandomInput(type, t_dims)
                                             .Input(paddings)
                                             .Attr("T", type)
                                             .Attr("Tpaddings", tpaddings));
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
    std::vector<int64> data_dims = RandomDims();
    Tensor indices = RandomReductionIndices(data_dims.size());
    bool keep_dims = Choose<bool>({false, true});
    return ExpectTfAndXlaOutputsAreClose(OpTestBuilder("Prod")
                                             .RandomInput(type, data_dims)
                                             .Input(indices)
                                             .Attr("T", type)
                                             .Attr("keep_dims", keep_dims));
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
    std::vector<int64> dims = RandomDims();
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
    return ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("Reshape")
            .RandomInput(type, dims_before)
            .Input(test::AsTensor<int32>(
                std::vector<int32>(dims_after.begin(), dims_after.end())))
            .Attr("T", type));
  });
}

TEST_F(OpTest, ResizeBilinear) {
  Repeatedly([this]() {
    std::vector<int64> in_dims = RandomDims(4, 4);
    std::vector<int64> out_dims = RandomDims(2, 2);

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
    std::vector<int64> in_dims = RandomDims(4, 4);
    std::vector<int64> out_dims = RandomDims(2, 2);

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
  Repeatedly([this]() {
    std::vector<int64> dims = RandomDims(1);
    auto type = Choose<DataType>(kAllXlaTypes);
    int64 rank = dims.size();
    return ExpectTfAndXlaOutputsAreClose(OpTestBuilder("Reverse")
                                             .RandomInput(type, dims)
                                             .RandomInput(DT_BOOL, {rank})
                                             .Attr("T", type));
  });
}

TEST_F(OpTest, ReverseV2) {
  Repeatedly([this]() {
    auto type = Choose<DataType>(kAllXlaTypes);
    std::vector<int64> data_dims = RandomDims();
    Tensor indices = RandomReductionIndices(data_dims.size());
    return ExpectTfAndXlaOutputsAreClose(OpTestBuilder("ReverseV2")
                                             .RandomInput(type, data_dims)
                                             .Input(indices)
                                             .Attr("T", type));
  });
}

TEST_F(OpTest, Rint) {
  Repeatedly([this]() {
    return ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("Rint").RandomInput(DT_FLOAT).Attr("T", DT_FLOAT));
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
    auto type = Choose<DataType>(kAllXlaTypes);
    std::vector<int64> data_dims = RandomDims();

    std::vector<int32> begin(data_dims.size()), size(data_dims.size());
    for (int i = 0; i < data_dims.size(); ++i) {
      begin[i] =
          std::uniform_int_distribution<int32>(0, data_dims[i])(generator());
      size[i] = std::uniform_int_distribution<int32>(
          -1, data_dims[i] - begin[i])(generator());
    }
    return ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("Slice")
            .RandomInput(type, data_dims)
            .Input(test::AsTensor<int32>(begin))
            .Input(test::AsTensor<int32>(size))
            .Attr("T", type)
            .Attr("Index", DT_INT32));
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
    std::vector<int64> dims = RandomDims(2, 2, 1);
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
    std::vector<int64> dims = RandomDims();
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
    std::vector<int64> dims = RandomDims();
    return ExpectTfAndXlaOutputsAreClose(OpTestBuilder("SoftsignGrad")
                                             .RandomInput(DT_FLOAT, dims)
                                             .RandomInput(DT_FLOAT, dims)
                                             .Attr("T", DT_FLOAT));
  });
}

TEST_F(OpTest, SpaceToBatch) {
  Repeatedly([this]() {
    std::vector<int64> block_dims = RandomDims(4, 4, 0, 5);
    const int num_block_dims = 2;
    int64 block_size = RandomDim(2, 5);

    std::vector<int64> input_dims(1 + num_block_dims + 1);
    input_dims[0] = RandomDim();
    for (int i = 0; i < num_block_dims; ++i) {
      input_dims[1 + i] = block_dims[i] * block_size;
    }
    input_dims[1 + num_block_dims] = RandomDim();

    std::vector<int64> padding_vals;
    std::uniform_int_distribution<int> distribution(0, 7);
    for (int i = 0; i < num_block_dims; ++i) {
      int64 pad_before;
      int64 pad_after;
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
    std::vector<int64> block_dims = RandomDims(1, 3, 0, 5);
    int num_block_dims = block_dims.size();
    std::vector<int64> remaining_dims = RandomDims(0, 3);
    std::vector<int64> block_multipliers =
        RandomDims(block_dims.size(), block_dims.size(), 0, 4);

    std::vector<int64> input_dims(1 + num_block_dims + remaining_dims.size());
    input_dims[0] = RandomDim();
    for (int i = 0; i < num_block_dims; ++i) {
      input_dims[1 + i] = block_dims[i] * block_multipliers[i];
    }
    std::copy(remaining_dims.begin(), remaining_dims.end(),
              input_dims.begin() + 1 + num_block_dims);

    std::vector<int64> padding_vals;
    std::uniform_int_distribution<int> distribution(0, 7);
    for (int i = 0; i < num_block_dims; ++i) {
      int64 pad_before;
      int64 pad_after;
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
    int64 block = RandomDim(2, 5);
    std::vector<int64> input_dims = RandomDims(4, 4);
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
  Repeatedly([this]() {
    int64 x = RandomDim();
    int64 y = RandomDim();
    int64 z = RandomDim();

    std::vector<int64> a_dims = {x, y};
    std::vector<int64> b_dims = {y, z};

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
    std::vector<int64> dims = RandomDims(2, 2, 1);
    int64 batch_size = dims[0];
    int64 num_classes = dims[1];

    std::vector<int32> indices(batch_size);
    for (int64 i = 0; i < batch_size; ++i) {
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
  Repeatedly([this]() {
    auto type = Choose<DataType>(kAllXlaTypes);
    std::vector<int64> dims = RandomDims(1);
    std::uniform_int_distribution<int> ud;
    int32 dim = std::uniform_int_distribution<int32>(
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

TEST_F(OpTest, Sqrt) {
  Repeatedly([this]() {
    auto type = Choose<DataType>({DT_FLOAT, DT_COMPLEX64});
    return ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("Sqrt").RandomInput(type).Attr("T", type));
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
    std::vector<int64> t_dims = RandomDims(0, kDefaultMaxRank, 0, 5);
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
    std::vector<int64> data_dims = RandomDims();
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
  Repeatedly([this]() {
    auto type = Choose<DataType>(kAllXlaTypes);
    std::vector<int64> data_dims = RandomDims();
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
    int64 max_bitmask = (1LL << data_dims.size()) - 1;
    std::uniform_int_distribution<int64> bitmask_distribution(0, max_bitmask);
    int64 begin_mask = bitmask_distribution(generator());
    int64 end_mask = bitmask_distribution(generator());

    // Create a ellipsis bitmask with at most one 1 bit set.
    int64 ellipsis_mask = 0;
    if (!data_dims.empty() && std::bernoulli_distribution()(generator())) {
      int ellipsis_pos = std::uniform_int_distribution<int>(
          0, data_dims.size() - 1)(generator());
      ellipsis_mask = 1LL << ellipsis_pos;
    }

    int64 new_axis_mask = bitmask_distribution(generator());
    int64 shrink_axis_mask = bitmask_distribution(generator());
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
  Repeatedly([this]() {
    auto type = Choose<DataType>(kAllXlaTypes);

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
    return ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("StridedSliceGrad")
            .Input(test::AsTensor<int64>(dims))
            .Input(test::AsTensor<int64>(begin))
            .Input(test::AsTensor<int64>(end))
            .Input(test::AsTensor<int64>(strides))
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

TEST_F(OpTest, Tile) {
  Repeatedly([this]() {
    auto type = Choose<DataType>(kAllXlaTypes);
    std::vector<int64> t_dims = RandomDims(1);
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

TEST_F(OpTest, Transpose) {
  Repeatedly([this]() {
    auto type = Choose<DataType>(kAllXlaTypes);
    std::vector<int64> data_dims = RandomDims();
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

TEST_F(OpTest, ZerosLike) {
  Repeatedly([this]() {
    auto type = Choose<DataType>({DT_INT32, DT_FLOAT, DT_COMPLEX64});
    return ExpectTfAndXlaOutputsAreClose(
        OpTestBuilder("ZerosLike").RandomInput(type).Attr("T", type));
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
      tensorflow::Flag("tf_xla_max_tensor_size",
                       &tensorflow::tf_xla_max_tensor_size,
                       "Maximum number of elements for random input tensors."),
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
