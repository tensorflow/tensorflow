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

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#include "tensorflow/core/kernels/gpu_prim_helpers.h"

#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

template <typename Tkey, typename Tindex>
class TestGpuRadixSortKernel : public tensorflow::OpKernel {
 public:
  explicit TestGpuRadixSortKernel(tensorflow::OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("need_keys_out", &need_keys_out_));
    OP_REQUIRES_OK(context, context->GetAttr("num_bits", &num_bits_));
    if (num_bits_ == -1) {
      num_bits_ = sizeof(Tkey) * 8;
    }
  }

  void Compute(tensorflow::OpKernelContext* context) override {
    const Tensor& keys_in = context->input(0);
    const Tensor& indices_in = context->input(1);

    const Tkey* keys_in_data = keys_in.flat<Tkey>().data();
    const Tindex* indices_in_data = indices_in.NumElements() == 0
                                        ? nullptr
                                        : indices_in.flat<Tindex>().data();

    int64 size = keys_in.NumElements();

    Tkey* keys_out_data = nullptr;
    if (need_keys_out_) {
      Tensor* keys_out = nullptr;
      OP_REQUIRES_OK(
          context, context->allocate_output(0, TensorShape({size}), &keys_out));
      keys_out_data = keys_out->flat<Tkey>().data();
    }

    Tensor* indices_out;
    OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape({size}),
                                                     &indices_out));
    Tindex* indices_out_data = indices_out->flat<Tindex>().data();

    OP_REQUIRES_OK(context,
                   GpuRadixSort(context, size, keys_in_data, keys_out_data,
                                indices_in_data, indices_out_data, num_bits_));
  }

 private:
  bool need_keys_out_;
  int num_bits_;
};

REGISTER_OP("TestGpuRadixSort")
    .Input("keys_in: Tkey")
    .Input("indices_in: Tindex")
    .Output("keys_out: Tkey")
    .Output("indices_out: Tindex")
    .Attr("need_keys_out: bool = true")
    .Attr("num_bits: int = -1")
    .Attr("Tkey: type")
    .Attr("Tindex: type");
#define REGISTER_KERNELS(Tkey, Tindex)                           \
  REGISTER_KERNEL_BUILDER(Name("TestGpuRadixSort")               \
                              .Device(tensorflow::DEVICE_GPU)    \
                              .TypeConstraint<Tkey>("Tkey")      \
                              .TypeConstraint<Tindex>("Tindex"), \
                          TestGpuRadixSortKernel<Tkey, Tindex>)
REGISTER_KERNELS(float, int32);
REGISTER_KERNELS(int32, int32);
#undef REGISTER_KERNELS

template <typename T>
class TestGpuInclusivePrefixSumKernel : public tensorflow::OpKernel {
 public:
  explicit TestGpuInclusivePrefixSumKernel(
      tensorflow::OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(tensorflow::OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    const T* input_data = input.flat<T>().data();
    int64 size = input.NumElements();

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, TensorShape({size}), &output));
    T* output_data = output->flat<T>().data();

    OP_REQUIRES_OK(
        context, GpuInclusivePrefixSum(context, size, input_data, output_data));
  }
};

REGISTER_OP("TestGpuInclusivePrefixSum")
    .Input("input: T")
    .Output("output: T")
    .Attr("T: type");
#define REGISTER_KERNELS(T)                                   \
  REGISTER_KERNEL_BUILDER(Name("TestGpuInclusivePrefixSum")   \
                              .Device(tensorflow::DEVICE_GPU) \
                              .TypeConstraint<T>("T"),        \
                          TestGpuInclusivePrefixSumKernel<T>)
REGISTER_KERNELS(int32);
#undef REGISTER_KERNELS

template <typename T, typename Toffset, typename ReduceOp>
class TestGpuSegmentedReduceKernel : public tensorflow::OpKernel {
 public:
  explicit TestGpuSegmentedReduceKernel(
      tensorflow::OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(tensorflow::OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    const T* input_data = input.flat<T>().data();
    const Tensor& segment_offsets = context->input(1);
    const Toffset* segment_offsets_data =
        segment_offsets.flat<Toffset>().data();
    int num_segments = segment_offsets.NumElements() - 1;
    const Tensor& initial_value_tensor = context->input(2);
    T initial_value = initial_value_tensor.scalar<T>()();

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(
                                0, TensorShape({num_segments}), &output));
    T* output_data = output->flat<T>().data();

    OP_REQUIRES_OK(
        context,
        GpuSegmentedReduce(context, num_segments, ReduceOp(), initial_value,
                           input_data, segment_offsets_data, output_data));
  }

 private:
  T initial_value_;
};

REGISTER_OP("TestGpuSegmentedSum")
    .Input("input: T")
    .Input("segment_offsets: Toffset")
    .Input("initial_value: T")
    .Output("output: T")
    .Attr("T: type")
    .Attr("Toffset: type");
#define REGISTER_KERNELS(T, Toffset)           \
  REGISTER_KERNEL_BUILDER(                     \
      Name("TestGpuSegmentedSum")              \
          .Device(tensorflow::DEVICE_GPU)      \
          .HostMemory("initial_value")         \
          .TypeConstraint<T>("T")              \
          .TypeConstraint<Toffset>("Toffset"), \
      TestGpuSegmentedReduceKernel<T, Toffset, gpuprim::Sum>)
REGISTER_KERNELS(int32, int32);
#undef REGISTER_KERNELS

template <typename T>
class TestGpuSelectFlaggedKernel : public tensorflow::OpKernel {
 public:
  explicit TestGpuSelectFlaggedKernel(tensorflow::OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("output_size", &output_size_));
  }

  void Compute(tensorflow::OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    const T* input_data = input.flat<T>().data();
    const Tensor& flags = context->input(1);
    const bool* flags_data = flags.flat<bool>().data();

    int64_t input_size = input.dim_size(0);

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(
                                0, TensorShape({output_size_}), &output));
    T* output_data = output->flat<T>().data();

    Tensor output_size_t;
    OP_REQUIRES_OK(context, context->allocate_temp(DT_INT64, TensorShape({}),
                                                   &output_size_t));
    int64_t* output_size_data = output_size_t.scalar<int64_t>().data();

    OP_REQUIRES_OK(context,
                   GpuSelectFlagged(context, input_size, input_data, flags_data,
                                    output_data, output_size_data));

    // Copy the computed output size to host and ensure it matches.
    se::Stream* stream = context->op_device_context()->stream();
    int64_t output_size_host;
    OP_REQUIRES_OK(
        context, stream->Memcpy(&output_size_host,
                                se::DeviceMemoryBase(output_size_data,
                                                     sizeof(output_size_data)),
                                sizeof(output_size_host)));
    OP_REQUIRES_OK(context, stream->BlockHostUntilDone());
    OP_REQUIRES(context, output_size_host == output_size_,
                errors::Internal("Incorrect output size: expected ",
                                 output_size_, ", got ", output_size_host));
  }

 private:
  int64_t output_size_;
};

REGISTER_OP("TestGpuSelectFlagged")
    .Input("input: T")
    .Input("flags: bool")
    .Output("output: T")
    .Attr("T: type")
    .Attr("output_size: int");
#define REGISTER_KERNELS(T)                                   \
  REGISTER_KERNEL_BUILDER(Name("TestGpuSelectFlagged")        \
                              .Device(tensorflow::DEVICE_GPU) \
                              .TypeConstraint<T>("T"),        \
                          TestGpuSelectFlaggedKernel<T>)
REGISTER_KERNELS(int32);
#undef REGISTER_KERNELS

class GpuPrimHelpersTest : public OpsTestBase {
 protected:
  GpuPrimHelpersTest() {
    SetDevice(DEVICE_GPU,
              std::unique_ptr<tensorflow::Device>(DeviceFactory::NewDevice(
                  "GPU", {}, "/job:a/replica:0/task:0")));
  }

  void MakeRadixSort(DataType key_type, DataType index_type,
                     bool need_keys_out = true, int num_bits = -1) {
    TF_ASSERT_OK(NodeDefBuilder("test_op", "TestGpuRadixSort")
                     .Input(FakeInput(key_type))
                     .Input(FakeInput(index_type))
                     .Attr("need_keys_out", need_keys_out)
                     .Attr("num_bits", num_bits)
                     .Finalize(node_def()));
    TF_ASSERT_OK(InitOp());
  }

  void MakeInclusivePrefixSum(DataType type) {
    TF_ASSERT_OK(NodeDefBuilder("test_op", "TestGpuInclusivePrefixSum")
                     .Input(FakeInput(type))
                     .Finalize(node_def()));
    TF_ASSERT_OK(InitOp());
  }

  void MakeSegmentedSum(DataType type, DataType offset_type) {
    TF_ASSERT_OK(NodeDefBuilder("test_op", "TestGpuSegmentedSum")
                     .Input(FakeInput(type))
                     .Input(FakeInput(offset_type))
                     .Input(FakeInput(type))
                     .Finalize(node_def()));
    TF_ASSERT_OK(InitOp());
  }

  void MakeSelectFlagged(DataType type, int64 output_size) {
    TF_ASSERT_OK(NodeDefBuilder("test_op", "TestGpuSelectFlagged")
                     .Input(FakeInput(type))
                     .Input(FakeInput(DT_BOOL))
                     .Attr("output_size", output_size)
                     .Finalize(node_def()));
    TF_ASSERT_OK(InitOp());
  }
};

TEST_F(GpuPrimHelpersTest, GpuRadixSort_Keys) {
  MakeRadixSort(DT_FLOAT, DT_INT32);
  AddInputFromArray<float>(TensorShape({8}), {4, 2, 6, 7, 1, 3, 0, 5});  // keys
  AddInputFromArray<int32>(TensorShape({0}), {});                        // inds
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected_keys_out(allocator(), DT_FLOAT, TensorShape({8}));
  test::FillValues<float>(&expected_keys_out, {0, 1, 2, 3, 4, 5, 6, 7});
  test::ExpectTensorEqual<float>(expected_keys_out, *GetOutput(0));

  Tensor expected_indices_out(allocator(), DT_INT32, TensorShape({8}));
  test::FillValues<int32>(&expected_indices_out, {6, 4, 1, 5, 0, 7, 2, 3});
  test::ExpectTensorEqual<int32>(expected_indices_out, *GetOutput(1));
}

TEST_F(GpuPrimHelpersTest, GpuRadixSort_KeysAndIndices) {
  MakeRadixSort(DT_FLOAT, DT_INT32);
  AddInputFromArray<float>(TensorShape({8}), {4, 2, 6, 7, 1, 3, 0, 5});  // keys
  AddInputFromArray<int32>(TensorShape({8}), {7, 6, 5, 4, 3, 2, 1, 0});  // inds
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected_keys_out(allocator(), DT_FLOAT, TensorShape({8}));
  test::FillValues<float>(&expected_keys_out, {0, 1, 2, 3, 4, 5, 6, 7});
  test::ExpectTensorEqual<float>(expected_keys_out, *GetOutput(0));

  Tensor expected_indices_out(allocator(), DT_INT32, TensorShape({8}));
  test::FillValues<int32>(&expected_indices_out, {1, 3, 6, 2, 7, 0, 5, 4});
  test::ExpectTensorEqual<int32>(expected_indices_out, *GetOutput(1));
}

TEST_F(GpuPrimHelpersTest, GpuRadixSort_NoKeysOut) {
  MakeRadixSort(DT_FLOAT, DT_INT32, /*need_keys_out=*/false);
  AddInputFromArray<float>(TensorShape({8}), {4, 2, 6, 7, 1, 3, 0, 5});  // keys
  AddInputFromArray<int32>(TensorShape({0}), {});                        // inds
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected_indices_out(allocator(), DT_INT32, TensorShape({8}));
  test::FillValues<int32>(&expected_indices_out, {6, 4, 1, 5, 0, 7, 2, 3});
  test::ExpectTensorEqual<int32>(expected_indices_out, *GetOutput(1));
}

TEST_F(GpuPrimHelpersTest, GpuRadixSort_WithNumBits) {
  // Only sort by the lowest 2 bits, otherwise keep input order (stable sort).
  MakeRadixSort(DT_INT32, DT_INT32, /*need_keys_out=*/true, /*num_bits=*/2);
  AddInputFromArray<int32>(TensorShape({8}), {4, 2, 6, 7, 1, 3, 0, 5});  // keys
  AddInputFromArray<int32>(TensorShape({0}), {});                        // inds
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected_keys_out(allocator(), DT_INT32, TensorShape({8}));
  test::FillValues<int32>(&expected_keys_out, {4, 0, 1, 5, 2, 6, 7, 3});
  test::ExpectTensorEqual<int32>(expected_keys_out, *GetOutput(0));

  Tensor expected_indices_out(allocator(), DT_INT32, TensorShape({8}));
  test::FillValues<int32>(&expected_indices_out, {0, 6, 4, 7, 1, 2, 3, 5});
  test::ExpectTensorEqual<int32>(expected_indices_out, *GetOutput(1));
}

TEST_F(GpuPrimHelpersTest, GpuRadixSort_WithNumBitsZero) {
  // Check that num_bits=0 is handled correctly.
  MakeRadixSort(DT_INT32, DT_INT32, /*need_keys_out=*/true, /*num_bits=*/0);
  AddInputFromArray<int32>(TensorShape({8}), {4, 2, 6, 7, 1, 3, 0, 5});  // keys
  AddInputFromArray<int32>(TensorShape({0}), {});                        // inds
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected_keys_out(allocator(), DT_INT32, TensorShape({8}));
  test::FillValues<int32>(&expected_keys_out, {4, 2, 6, 7, 1, 3, 0, 5});
  test::ExpectTensorEqual<int32>(expected_keys_out, *GetOutput(0));

  Tensor expected_indices_out(allocator(), DT_INT32, TensorShape({8}));
  test::FillValues<int32>(&expected_indices_out, {0, 1, 2, 3, 4, 5, 6, 7});
  test::ExpectTensorEqual<int32>(expected_indices_out, *GetOutput(1));
}

TEST_F(GpuPrimHelpersTest, GpuRadixSort_KeysAndIndices_WithNumBitsZero) {
  // Check that num_bits=0 is handled correctly (with indices_in).
  MakeRadixSort(DT_INT32, DT_INT32, /*need_keys_out=*/true, /*num_bits=*/0);
  AddInputFromArray<int32>(TensorShape({8}), {4, 2, 6, 7, 1, 3, 0, 5});  // keys
  AddInputFromArray<int32>(TensorShape({8}), {7, 6, 5, 4, 3, 2, 1, 0});  // inds
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected_keys_out(allocator(), DT_INT32, TensorShape({8}));
  test::FillValues<int32>(&expected_keys_out, {4, 2, 6, 7, 1, 3, 0, 5});
  test::ExpectTensorEqual<int32>(expected_keys_out, *GetOutput(0));

  Tensor expected_indices_out(allocator(), DT_INT32, TensorShape({8}));
  test::FillValues<int32>(&expected_indices_out, {7, 6, 5, 4, 3, 2, 1, 0});
  test::ExpectTensorEqual<int32>(expected_indices_out, *GetOutput(1));
}

TEST_F(GpuPrimHelpersTest, GpuInclusivePrefixSum) {
  MakeInclusivePrefixSum(DT_INT32);
  AddInputFromArray<int32>(TensorShape({8}), {4, 2, 6, 7, 1, 3, 0, 5});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected_output(allocator(), DT_INT32, TensorShape({8}));
  test::FillValues<int32>(&expected_output, {4, 6, 12, 19, 20, 23, 23, 28});
  test::ExpectTensorEqual<int32>(expected_output, *GetOutput(0));
}

TEST_F(GpuPrimHelpersTest, GpuSegmentedReduce_Sum) {
  MakeSegmentedSum(DT_INT32, DT_INT32);
  // Input.
  AddInputFromArray<int32>(TensorShape({10}), {0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
  // Segment IDs.
  AddInputFromArray<int32>(TensorShape({6}), {1, 3, 4, 4, 8, 10});
  // Initial value.
  AddInputFromArray<int32>(TensorShape({}), {0});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected_output(allocator(), DT_INT32, TensorShape({5}));
  test::FillValues<int32>(&expected_output, {3, 3, 0, 22, 17});
  test::ExpectTensorEqual<int32>(expected_output, *GetOutput(0));
}

TEST_F(GpuPrimHelpersTest, GpuSelectFlagged) {
  MakeSelectFlagged(DT_INT32, 3);
  // Input.
  AddInputFromArray<int32>(TensorShape({10}), {0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
  // Flags.
  AddInputFromArray<bool>(TensorShape({10}), {0, 0, 1, 0, 1, 0, 0, 1, 0, 0});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected_output(allocator(), DT_INT32, TensorShape({3}));
  test::FillValues<int32>(&expected_output, {2, 4, 7});
  test::ExpectTensorEqual<int32>(expected_output, *GetOutput(0));
}

TEST_F(GpuPrimHelpersTest, GpuSelectFlagged_Empty) {
  MakeSelectFlagged(DT_INT32, 0);
  // Input.
  AddInputFromArray<int32>(TensorShape({0}), {});
  // Flags.
  AddInputFromArray<bool>(TensorShape({0}), {});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected_output(allocator(), DT_INT32, TensorShape({0}));
  test::FillValues<int32>(&expected_output, {});
  test::ExpectTensorEqual<int32>(expected_output, *GetOutput(0));
}

}  // namespace
}  // namespace tensorflow

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
