/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

// See docs in ../ops/array_ops.cc.

#ifdef INTEL_MKL

#include "mkldnn.hpp"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/prefetch.h"
#include "tensorflow/core/util/mkl_types.h"
#include "tensorflow/core/util/mkl_util.h"

using mkldnn::stream;
#ifndef ENABLE_MKLDNN_V1
using mkldnn::view;
#endif

namespace tensorflow {

namespace {

gtl::InlinedVector<int64, 4> IntTensorToInt64Vec(const Tensor& tensor) {
  gtl::InlinedVector<int64, 4> out;
  if (tensor.dtype() == DT_INT32) {
    for (int64 i = 0; i < tensor.NumElements(); ++i) {
      out.push_back(tensor.flat<int32>()(i));
    }
  } else if (tensor.dtype() == DT_INT64) {
    for (int64 i = 0; i < tensor.NumElements(); ++i) {
      out.push_back(tensor.flat<int64>()(i));
    }
  } else {
    // tensor must be either int32 or int64
    DCHECK(false);
  }
  return out;
}

}  // namespace

typedef Eigen::ThreadPoolDevice CPUDevice;

// A version of SharedValidation (slice_op.h) written for input that is in
// either Mkl layout or Tensorflow layout. A shared code to validate input
// shapes and check for identity, which is not dependent on the type of T.
// We do this to reduce code size by not duplicating all this for all T
// (float, double, int32, etc.)
static void ValidateMklInputs(OpKernelContext* context, bool* is_identity,
                              gtl::InlinedVector<int64, 4>* begin,
                              gtl::InlinedVector<int64, 4>* size) {
  const int kInputTensorIndex = 0;
  const int kInputBeginIndex = 1;
  const int kInputSizeIndex = 2;
  const Tensor& input = MklGetInput(context, kInputTensorIndex);
  const Tensor& begin_tensor = MklGetInput(context, kInputBeginIndex);
  const Tensor& size_tensor = MklGetInput(context, kInputSizeIndex);

  MklDnnShape input_mkl_shape, begin_mkl_shape, size_mkl_shape;
  GetMklShape(context, kInputTensorIndex, &input_mkl_shape);
  GetMklShape(context, kInputBeginIndex, &begin_mkl_shape);
  GetMklShape(context, kInputSizeIndex, &size_mkl_shape);

  // Begin and size tensors cannot be in MklDnn layout.
  DCHECK_EQ(begin_mkl_shape.IsMklTensor(), false);
  DCHECK_EQ(size_mkl_shape.IsMklTensor(), false);

  TensorShape input_tf_shape = input_mkl_shape.IsMklTensor()
                                   ? input_mkl_shape.GetTfShape()
                                   : input.shape();
  const int input_dims = input_tf_shape.dims();

  OP_REQUIRES(
      context,
      TensorShapeUtils::IsVector(begin_tensor.shape()) &&
          TensorShapeUtils::IsVector(size_tensor.shape()) &&
          begin_tensor.NumElements() == input_dims &&
          size_tensor.NumElements() == input_dims,
      errors::InvalidArgument(
          "Expected begin and size arguments to be 1-D tensors of size ",
          input_dims, ", but got shapes ", begin_tensor.shape().DebugString(),
          " and ", size_tensor.shape().DebugString(), " instead."));

  *begin = IntTensorToInt64Vec(begin_tensor);
  *size = IntTensorToInt64Vec(size_tensor);
  for (int i = 0; i < input_dims; ++i) {
    if ((*size)[i] == -1) {
      // A size[i] of -1 means "all elements from begin[i] to dim_size(i)".
      (*size)[i] = input_tf_shape.dim_size(i) - (*begin)[i];
    }
  }

  *is_identity = true;
  for (int i = 0; i < input_dims; ++i) {
    int64 b = (*begin)[i];
    int64 s = (*size)[i];
    if (input_tf_shape.dim_size(i) == 0) {
      OP_REQUIRES(
          context, b == 0 && s == 0,
          errors::InvalidArgument("Expected begin[", i, "] == 0 (got ", b,
                                  ") and size[", i, "] == 0 ", "(got ", s,
                                  ") when ", "input.dim_size(", i, ") == 0"));
    } else {
      OP_REQUIRES(context, 0 <= b && b <= input_tf_shape.dim_size(i),
                  errors::InvalidArgument("Expected begin[", i, "] in [0, ",
                                          input_tf_shape.dim_size(i),
                                          "], but got ", b));
      OP_REQUIRES(context, 0 <= s && b + s <= input_tf_shape.dim_size(i),
                  errors::InvalidArgument("Expected size[", i, "] in [0, ",
                                          input_tf_shape.dim_size(i) - b,
                                          "], but ", "got ", s));
    }
    const bool take_all = (b == 0) && (s == input_tf_shape.dim_size(i));
    (*is_identity) &= take_all;
  }
}

// A version of SharedSliceCommonCases function written for input tensor
// that may be in MklDnn layout or in Tensorflow layout.
template <typename T>
static void CheckCommonCasesForMklInputs(OpKernelContext* context,
                                         gtl::InlinedVector<int64, 4>* begin,
                                         gtl::InlinedVector<int64, 4>* size,
                                         bool* done) {
  bool is_identity = true;
  *done = false;

  ValidateMklInputs(context, &is_identity, begin, size);
  if (!context->status().ok()) return;

  const Tensor& input = MklGetInput(context, 0);
  MklDnnShape input_mkl_shape;
  GetMklShape(context, 0, &input_mkl_shape);

  if (is_identity) {
    VLOG(1) << "Slice identity";
    context->set_output(0, input);
    // Mkl metadata tensor in this case can just be forwarded from input to
    // output.
    AllocateOutputSetMklShape(context, 0, input_mkl_shape);
    *done = true;
  }
}

// This structure aggregates multiple inputs to Slice methods.
struct MklSliceParams {
  // Parameters from & to represents memory pointing to reorder.
  const memory* from;
  const memory* to;

  // Parameters begin_dims & size_dims represents offset and length
  // passed to view primitive.
  memory::dims begin_dims;
  memory::dims size_dims;

  MklSliceParams(const memory* from, const memory* to, memory::dims begin_dims,
                 memory::dims size_dims)
      : from(from), to(to), begin_dims(begin_dims), size_dims(size_dims) {}
};

// This implements the shared interface of Slice reorders.
template <typename T>
class MklSlicePrimitive : public MklPrimitive {
 public:
  explicit MklSlicePrimitive(const MklSliceParams& sliceParams)
      : MklPrimitive(engine(ENGINE_CPU, 0)) {
    Setup(sliceParams);
  }

  ~MklSlicePrimitive() {}

  void Execute(const MklSliceParams& sliceParams, std::shared_ptr<stream> slice_stream) {
    context_.src_mem->set_data_handle(sliceParams.from->get_data_handle());
    context_.dst_mem->set_data_handle(sliceParams.to->get_data_handle());

#ifdef ENABLE_MKLDNN_V1
    execute_primitives(context_.slice_primitives, slice_stream,
                       context_.slice_primitives_args);
#else
    slice_stream->submit(context_.slice_primitives);
#endif

    // We should set it back to DummyData so as to make the primitive
    // in cache pool stateless. Otherwise, if the result for previous
    // iteration is kept, problems of current iteration won't be
    // thrown immediately, and wrong data would be reused.
    context_.src_mem->set_data_handle(DummyData);
    context_.dst_mem->set_data_handle(DummyData);
    return;
  }

  std::shared_ptr<primitive> GetPrimitive() { return context_.reorder_prim; }

 private:
  struct SliceContext {
    std::shared_ptr<mkldnn::memory> src_mem;
    std::shared_ptr<mkldnn::memory> dst_mem;
    std::shared_ptr<primitive> reorder_prim;
    std::shared_ptr<reorder::primitive_desc> reorder_pd;
    std::shared_ptr<mkldnn::stream> slice_stream;
    std::vector<mkldnn::primitive> slice_primitives;
#ifdef ENABLE_MKLDNN_V1
    std::shared_ptr<mkldnn::memory> src_sub_mem;
    std::vector<std::unordered_map<int, memory>> slice_primitives_args;
#else
    std::shared_ptr<view::primitive_desc> view_pd;
#endif  // ENABLE_MKLDNN_V1
    SliceContext()
        : src_mem(nullptr), dst_mem(nullptr), reorder_prim(nullptr) {}
  } context_;

  void Setup(const MklSliceParams& sliceParams) {
    // Actually, DummyData will not be used in computation,
    // because the real data will be filled before execution.
    context_.src_mem.reset(new MEMORY_CONSTRUCTOR_WITH_MEM_PD(
        sliceParams.from, cpu_engine_, DummyData));
    context_.dst_mem.reset(new MEMORY_CONSTRUCTOR_WITH_MEM_PD(
        sliceParams.to, cpu_engine_, DummyData));
    auto src_pd = context_.src_mem->GET_DESC;
    auto dst_pd = context_.dst_mem->GET_DESC;
#ifdef ENABLE_MKLDNN_V1
    // MKL-DNN 1.x removes struct view, alias of memory in 0.x version.
    // So the implementation is based on submemory.
    auto src_sub_desc = context_.src_mem->get_desc().submemory_desc(
        sliceParams.size_dims, sliceParams.begin_dims);
    context_.src_sub_mem.reset(new memory(src_sub_desc, cpu_engine_, nullptr));
    context_.reorder_pd = std::make_shared<reorder::primitive_desc>(
        reorder::primitive_desc(*context_.src_sub_mem, *context_.dst_mem));
    context_.reorder_prim =
        std::make_shared<mkldnn::reorder>(reorder(*context_.reorder_pd));

    context_.slice_primitives_args.push_back(
        {{MKLDNN_ARG_SRC, *context_.src_mem},
         { MKLDNN_ARG_DST,
           *context_.dst_mem }});
#else
    context_.view_pd =
        std::make_shared<view::primitive_desc>(view::primitive_desc(
            src_pd, sliceParams.size_dims, sliceParams.begin_dims));
    context_.reorder_pd =
        std::make_shared<reorder::primitive_desc>(reorder::primitive_desc(
            context_.view_pd->dst_primitive_desc(), dst_pd));
    context_.reorder_prim = std::make_shared<mkldnn::reorder>(
        reorder(*context_.reorder_pd, *context_.src_mem, *context_.dst_mem));
#endif
    context_.slice_primitives.push_back(*context_.reorder_prim);
  }
};

template <typename T>
class MklSlicePrimitiveFactory : public MklPrimitiveFactory<T> {
 public:
  static MklSlicePrimitive<T>* Get(const MklSliceParams& sliceParams) {
    auto reorderPrim = static_cast<MklSlicePrimitive<T>*>(
        MklSlicePrimitiveFactory<T>::GetInstance().GetReorder(sliceParams));
    if (reorderPrim == nullptr) {
      reorderPrim = new MklSlicePrimitive<T>(sliceParams);
      MklSlicePrimitiveFactory<T>::GetInstance().SetReorder(sliceParams,
                                                            reorderPrim);
    }
    return reorderPrim;
  }

  static MklSlicePrimitiveFactory& GetInstance() {
    static MklSlicePrimitiveFactory instance_;
    return instance_;
  }

 private:
  MklSlicePrimitiveFactory() {}
  ~MklSlicePrimitiveFactory() {}

  static string CreateKey(const MklSliceParams& sliceParams) {
    string prefix = "reorder";
    FactoryKeyCreator key_creator;
    auto const& from_desc = GET_MEMORY_DESC_FROM_MEM_PTR(sliceParams.from).data;
    auto const& to_desc = GET_MEMORY_DESC_FROM_MEM_PTR(sliceParams.to).data;
    const int kIdxFirstStride = 0;
    memory::dims from_dims(from_desc.dims, &from_desc.dims[from_desc.ndims]);
    memory::dims to_dims(to_desc.dims, &to_desc.dims[to_desc.ndims]);

    // MKL-DNN removes "struct view". Submemory has similar capability.
    auto from_strides = from_desc.MEMORY_FORMAT_DESC.blocking.strides;
    auto to_strides = to_desc.MEMORY_FORMAT_DESC.blocking.strides;
    memory::dims from_strides_outer_blocks(
        GET_BLOCK_STRIDES(from_strides, kIdxFirstStride),
        &GET_BLOCK_STRIDES(from_strides, kIdxFirstStride)[from_desc.ndims]);
    memory::dims to_strides_outer_blocks(
        GET_BLOCK_STRIDES(to_strides, kIdxFirstStride),
        &GET_BLOCK_STRIDES(to_strides, kIdxFirstStride)[to_desc.ndims]);

    key_creator.AddAsKey(prefix);
#ifndef ENABLE_MKLDNN_V1
    key_creator.AddAsKey(static_cast<int>(from_desc.format));
#endif
    key_creator.AddAsKey(static_cast<int>(from_desc.data_type));
    key_creator.AddAsKey(from_dims);
    key_creator.AddAsKey(from_strides_outer_blocks);
#ifndef ENABLE_MKLDNN_V1
    key_creator.AddAsKey(static_cast<int>(to_desc.format));
#endif
    key_creator.AddAsKey(static_cast<int>(to_desc.data_type));
    key_creator.AddAsKey(to_dims);
    key_creator.AddAsKey(to_strides_outer_blocks);
    key_creator.AddAsKey(sliceParams.begin_dims);
    key_creator.AddAsKey(sliceParams.size_dims);
    return key_creator.GetKey();
  }

  MklPrimitive* GetReorder(const MklSliceParams& sliceParams) {
    string key = CreateKey(sliceParams);
    return this->GetOp(key);
  }

  void SetReorder(const MklSliceParams& sliceParams, MklPrimitive* op) {
    string key = CreateKey(sliceParams);
    this->SetOp(key, op);
  }
};

// MKL-DNN implementation of Slice
template <typename Device, typename T>
class MklSliceOp : public OpKernel {
 public:
  explicit MklSliceOp(OpKernelConstruction* context) : OpKernel(context) {}

  ~MklSliceOp() {}

  void Compute(OpKernelContext* context) override {
    gtl::InlinedVector<int64, 4> begin;
    gtl::InlinedVector<int64, 4> size;
    bool done = false;

    CheckCommonCasesForMklInputs<T>(context, &begin, &size, &done);

    if (!context->status().ok() || done == true) return;

    // Though MKL-DNN supports more than 8 dimension and
    // less than 12 dimension tensor.
    // But we are mimicking functionality of Eigen Slice op for CPU.
    if (begin.size() >= 8) {
      OP_REQUIRES(
          context, false,
          errors::Unimplemented("MklSliceOp : Unhandled input dimensions"));
    }

    ComputeMklSlice(context, begin, size);
  }

 private:
  // Slice op implemented using MKL-DNN APIs.
  void ComputeMklSlice(OpKernelContext* context,
                       const gtl::InlinedVector<int64, 4>& begin,
                       const gtl::InlinedVector<int64, 4>& size) {
    try {
      // MKL-DNN API usage below is guided by description at:
      //  https://github.com/01org/mkl-dnn/issues/69
      //
      // Relevant part of the description is copied below:
      //
      // Let's say you want to copy a part of memory into another buffer (and
      // probably change the format). Then your steps are:
      //
      // 1. create memory primitive descriptor in_mem_pd and memory primitive
      //    in_mem_p for the entire source data. create view primitive
      //    descriptor in_submem_pd based on in_mem_pd, initial offsets,
      //    and sub-sizes
      // 2. create memory primitive descriptor out_mem_pd and memory primitive
      //    out_mem_p for the output (the logical sizes should match sub-sizes
      //    used in step 1, but the format might be arbitrary)
      // 3. create reorder primitive descriptor reorder_pd based on in_submem_pd
      //    and out_mem_pd. create reorder primitive itself based on reorder_pd,
      //    in_mem_p, and out_mem_p.
      //
      // Please notice that there is no view primitive. There is only view
      // primitive descriptor. And the reorder uses source memory as input but
      // traverses it according to a view in_submem_pd.

      auto cpu_engine = engine(ENGINE_CPU, 0);
      MklDnnData<T> src(&cpu_engine);
      MklDnnData<T> output(&cpu_engine);

      // Populate offsets and sizes in memory::dims format based on vector.
      memory::dims begin_dims = {};
      begin_dims.resize(begin.size());
      for (size_t i = 0; i < begin.size(); ++i) begin_dims[i] = begin[i];
      memory::dims size_dims = {};
      bool empty = false;
      size_dims.resize(size.size());
      for (size_t i = 0; i < size.size(); ++i) {
        size_dims[i] = size[i];
        if (size_dims[i] == 0) empty = true;
      }

      Tensor* output_tensor = nullptr;
      MklDnnShape output_mkl_shape;

      // If no dimension is selected in slice, the result should be empty.
      // Just return an empty output tensor, and a dummy Mkl-shape tensor.
      if (empty) {  // for empty dims
        auto shape_to = MklDnnDimsToTFShape(size_dims);
        AllocateOutputSetMklShape(context, 0, &output_tensor, shape_to,
                                  output_mkl_shape);
        return;
      }

      // Step 1 (as per above description) - Create memory for user data.
      // We use blocked format here to describe input tensor.
      const Tensor& input_tensor = MklGetInput(context, 0);
      memory::dims input_dims, input_strides;
      MklDnnShape input_mkl_shape;
      GetMklShape(context, 0, &input_mkl_shape);

      if (input_mkl_shape.IsMklTensor()) {
        auto input_mkl_format = input_mkl_shape.GetTfDataFormat();
        auto input_tf_format = MklDnnDataFormatToTFDataFormat(input_mkl_format);

        bool is_slice2d = (input_mkl_shape.GetDimension() == 4);
        begin_dims = is_slice2d
                         ? MklDnnDimsInNCHW(begin_dims, input_tf_format)
                         : MklDnnDimsInNCDHW(begin_dims, input_tf_format);
        size_dims = is_slice2d ? MklDnnDimsInNCHW(size_dims, input_tf_format)
                               : MklDnnDimsInNCDHW(size_dims, input_tf_format);
        auto input_md = input_mkl_shape.GetMklLayout();
        src.SetUsrMem(input_md, &input_tensor);

        // Handle data format safely, change them to block format.
        // Compute parameters of reorder primitive first.
        input_dims = input_mkl_shape.GetSizesAsMklDnnDims();
        input_strides = CalculateTFStrides(input_dims);
      } else {
        // Initialize input dimensions and strides to be used when input is not
        // in MklDnn layout.
        input_dims = TFShapeToMklDnnDims(input_tensor.shape());
        input_strides = CalculateTFStrides(input_dims);
        // Create input memory descriptor.
        auto input_md =
            MklDnnData<T>::CreateBlockedMemDesc(input_dims, input_strides);
        src.SetUsrMem(input_md, &input_tensor);
      }

      // If format not equal to block format, execute reorder.
      // Or else do nothing for it.
      auto op_md =
          MklDnnData<T>::CreateBlockedMemDesc(input_dims, input_strides);
#ifdef ENABLE_MKLDNN_V1
      src.CheckReorderToOpMem(op_md, cpu_engine, context);
#else
      auto op_pd = memory::primitive_desc(op_md, cpu_engine);
      src.CheckReorderToOpMem(op_pd);
#endif

      // Step 2 - Create memory for output.
      auto output_strides = CalculateTFStrides(size_dims);
      auto output_md =
          MklDnnData<T>::CreateBlockedMemDesc(size_dims, output_strides);
#ifdef ENABLE_MKLDNN_V1
      auto output_pd = output_md;
#else
      auto output_pd = memory::primitive_desc(output_md, cpu_engine);
#endif
      AllocateOutputTensor(context, input_mkl_shape, &output_pd, size_dims,
                           &output_tensor, &output_mkl_shape);
      DCHECK(output_tensor);
      DCHECK_EQ(input_mkl_shape.IsMklTensor(), output_mkl_shape.IsMklTensor());
      output.SetUsrMem(output_md, output_tensor);

      // Step 3 - create reorder primitive.
      MklSliceParams sliceParams(&src.GetOpMem(), output.GetUsrMem(),
                                 begin_dims, size_dims);
      MklSlicePrimitive<T>* reorder_prim =
          MklSlicePrimitiveFactory<T>::Get(sliceParams);
      // Execute slice reorder.
      std::shared_ptr<stream> slice_stream;
      slice_stream.reset(CreateStream(context, reorder_prim->GetEngine()));
      reorder_prim->Execute(sliceParams, slice_stream);
    } catch (mkldnn::error& e) {
      string error_msg = "Status: " + std::to_string(e.status) +
                         ", message: " + string(e.message) + ", in file " +
                         string(__FILE__) + ":" + std::to_string(__LINE__);
      OP_REQUIRES_OK(
          context,
          errors::Aborted("Operation received an exception:", error_msg));
    }
  }

 private:
  void AllocateOutputTensor(OpKernelContext* context,
                            const MklDnnShape& input_mkl_shape,
                            MEMORY_PRIMITIVE_DESC* output_pd,
                            const memory::dims& output_dims,
                            Tensor** output_tensor,
                            MklDnnShape* output_mkl_shape) {
    DCHECK(output_tensor);
    DCHECK(output_mkl_shape);

    TensorShape output_tf_shape;

    if (input_mkl_shape.IsMklTensor()) {
      // Since input tensor is in Mkl layout, output tensor will be in Mkl
      // layout.

      // Allocate shape of Mkl tensor.
      output_mkl_shape->SetMklTensor(true);
      output_mkl_shape->SetMklLayout(output_pd);
      output_mkl_shape->SetElemType(MklDnnType<T>());
      output_mkl_shape->SetTfLayout(input_mkl_shape.GetDimension(), output_dims,
                                    input_mkl_shape.GetTfDataFormat());

      output_tf_shape.AddDim(output_pd->get_size() / sizeof(T));
    } else {
      // If input is not in Mkl layout, then output won't be in Mkl layout.
      output_mkl_shape->SetMklTensor(false);
      output_tf_shape = MklDnnDimsToTFShape(output_dims);
    }

    AllocateOutputSetMklShape(context, 0, output_tensor, output_tf_shape,
                              *output_mkl_shape);
  }
};

// MKL-DNN Slice registration
#define REGISTER_MKL_SLICE(type)                               \
  REGISTER_KERNEL_BUILDER(                                     \
      Name("_MklSlice")                                        \
          .Device(DEVICE_CPU)                                  \
          .TypeConstraint<type>("T")                           \
          .HostMemory("begin")                                 \
          .HostMemory("size")                                  \
          .Label(mkl_op_registry::kMklLayoutDependentOpLabel), \
      MklSliceOp<CPUDevice, type>);

TF_CALL_float(REGISTER_MKL_SLICE);
TF_CALL_bfloat16(REGISTER_MKL_SLICE);
#undef REGISTER_MKL_SLICE

}  // namespace tensorflow

#endif  // INTEL_MKL
