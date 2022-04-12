/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#if GOOGLE_CUDA && GOOGLE_TENSORRT
#include <iterator>
#include <limits>
#include <memory>

#include "tensorflow/compiler/tf2tensorrt/common/utils.h"
#include "tensorflow/compiler/tf2tensorrt/convert/convert_nodes.h"
#include "tensorflow/compiler/tf2tensorrt/convert/op_converter.h"
#include "tensorflow/compiler/tf2tensorrt/convert/op_converter_registry.h"
#include "tensorflow/compiler/tf2tensorrt/convert/ops/layer_utils.h"
#include "tensorflow/compiler/tf2tensorrt/convert/utils.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/util/einsum_op_util.h"
#include "third_party/tensorrt/NvInfer.h"

#if IS_TRT_VERSION_GE(7, 1, 3, 0)

namespace tensorflow {
namespace tensorrt {
namespace convert {

namespace {
// Finds the indices of elements in [begin, end) in array
// [array_begin, array_end), and appends the indices to permute. This is used to
// construct the permutation sequence for the operand with input labels
// [array_begin, array_end) to the desired permuted labels [begin, end).
template <typename T>
Status FindIndicesoOfAllValuesInSrc(absl::Span<const T> values,
                                    absl::Span<const T> src,
                                    std::vector<int>* indices) {
  if (src.size() < values.size()) {
    return errors::Internal(
        "Span 'src' cannot contain all elements of 'values'");
  }
  for (auto i = 0; i < values.size(); i++) {
    auto iter = absl::c_find(src, values[i]);
    if (iter == src.end()) {
      return errors::Internal("Label ", values[i], " not found");
    }
    int idx = std::distance(src.begin(), iter);
    indices->push_back(idx);
  }
  return Status::OK();
}

// Layout of the einsum dimensions: Batch, Free and Contraction indices.
// Example: adbc,adce -> adbe. The first tensor has layout BFC, the second BCF.
enum class EinsumLayout { BFC, BCF, MIX };

using DimType = EinsumDimensionType;
constexpr auto kBatch = DimType::kBatch;
constexpr auto kFree = DimType::kFree;
constexpr auto kContract = DimType::kContract;

// Describes an operand: input shape, number of batch, free and contract
// dimensions, and the permutation that is needed to bring it to a matmul
// compatible form.
class EinsumDescriptor {
 private:
  // Checks whether input_labels[offset:offset+m] matches labels from other.
  static bool OrderMatches(const Labels& input_labels, int offset, int m,
                           EinsumDimensionType dim_type,
                           const std::unique_ptr<EinsumDescriptor>& other) {
    if (other == nullptr) {
      return true;
    }
    int offset_other = 0;
    if (dim_type == kFree) {
      offset = other->offset_f;
    } else if (dim_type == kContract) {
      offset = other->offset_c;
    }
    return std::equal(input_labels.begin() + offset,
                      input_labels.begin() + offset + m,
                      other->permuted_labels.begin() + offset_other);
  }

  using label_t_iterator = std::vector<EinsumDimensionType>::const_iterator;
  static int32_t CountLabels(label_t_iterator begin, label_t_iterator end,
                             EinsumDimensionType val) {
    return static_cast<int32_t>(std::count_if(
        begin, end, [val](EinsumDimensionType t) { return t == val; }));
  }

  // Appends indices to the "permute" vector where types maches value.
  void AppendMatchingIndicesToPermute(
      const std::vector<EinsumDimensionType>& types, EinsumDimensionType val) {
    for (int i = 0; i < types.size(); i++) {
      if (types[i] == val) {
        permute.push_back(i);
      }
    }
  }

  Status DetermineLayout(const Labels& input_labels,
                         const std::vector<EinsumDimensionType>& types,
                         const std::unique_ptr<EinsumDescriptor>& other) {
    // Check if the current layout is BFC or BCF. In that case we could avoid
    // transpose.
    layout = EinsumLayout::MIX;
    if (CountLabels(types.begin(), types.begin() + b, kBatch) == b &&
        OrderMatches(input_labels, 0, b, kBatch, other)) {
      // Batch dims are the leading dims. They have the same order as other.
      if (CountLabels(types.begin() + b, types.begin() + b + f, kFree) == f) {
        // All the free dims are placed consecutively after the batch dims.
        // Their order is arbitrary. The final transpose will ensure that the
        // output has correct order. We still have to check that the contract
        // indices have correct order.
        if (OrderMatches(input_labels, b + f, c, kContract, other)) {
          layout = EinsumLayout::BFC;
        }
      } else if (CountLabels(types.begin() + b, types.begin() + b + c,
                             kContract) == c) {
        // All the contract dims are placed consecutively after the batch
        // dims. Check whether the contract dims have the same order as the
        // contract dims in other.
        if (OrderMatches(input_labels, b, c, kContract, other)) {
          layout = EinsumLayout::BCF;
        }
      }
    }
    return Status::OK();
  }

  Status CalculateMixedLayoutPermutation(
      const EinsumLayout preferred_layout, const Labels& input_labels,
      const std::vector<EinsumDimensionType>& types,
      const std::unique_ptr<EinsumDescriptor>& other) {
    // Input label types are mixed. Calculate a permutation that maps them
    // to the preferred layout (BCF or BFC).
    layout = preferred_layout;
    if (other == nullptr) {
      AppendMatchingIndicesToPermute(types, kBatch);
    } else {
      TF_RETURN_IF_ERROR(
          FindIndicesoOfAllValuesInSrc(/*values=*/
                                       absl::MakeConstSpan(
                                           other->permuted_labels.begin(),
                                           other->b),
                                       /*src=*/
                                       absl::MakeConstSpan(input_labels.begin(),
                                                           input_labels.size()),
                                       /*indices=*/&permute));
    }
    if (layout == EinsumLayout::BFC) {
      AppendMatchingIndicesToPermute(types, kFree);
      if (other == nullptr) {
        AppendMatchingIndicesToPermute(types, kContract);
      } else {
        TF_RETURN_IF_ERROR(FindIndicesoOfAllValuesInSrc(
            /*values=*/absl::MakeConstSpan(
                other->permuted_labels.begin() + other->offset_c, other->c),
            /*src=*/
            absl::MakeConstSpan(input_labels.begin(), input_labels.size()),
            /*indices=*/&permute));
      }
      return Status::OK();
    }
    if (other == nullptr) {
      AppendMatchingIndicesToPermute(types, kContract);
    } else {
      TF_RETURN_IF_ERROR(FindIndicesoOfAllValuesInSrc(
          /*values=*/absl::MakeConstSpan(
              other->permuted_labels.begin() + other->offset_c, other->c),
          /*src=*/absl::MakeConstSpan(input_labels.begin(), input_labels.end()),
          /*indices=*/&permute));
    }
    AppendMatchingIndicesToPermute(types, kFree);
    return Status::OK();
  }

  Status Initialize(const TRT_TensorOrWeights& operand, Labels input_labels,
                    std::vector<EinsumDimensionType>& label_types,
                    EinsumLayout preferred_layout,
                    const std::unique_ptr<EinsumDescriptor>& other = nullptr) {
    if (preferred_layout == EinsumLayout::MIX) {
      return errors::Internal("Preferred einsum layout cannot be MIX");
    }
    // Map label indices to label types.
    std::vector<EinsumDimensionType> types;  // Input label types.
    std::transform(input_labels.begin(), input_labels.end(),
                   std::back_inserter(types),
                   [&label_types](int i) { return label_types.at(i); });

    b = CountLabels(types.begin(), types.end(), kBatch);
    f = CountLabels(types.begin(), types.end(), kFree);
    c = CountLabels(types.begin(), types.end(), kContract);

    if (c == 0 || f == 0) {
      VLOG(2) << "Einsum equation needs to have at least one free and one "
                 "contract dimension";
      return errors::Unimplemented("No conversion for einsum equation.");
    }

    TF_RETURN_IF_ERROR(DetermineLayout(input_labels, types, other));
    if (layout == EinsumLayout::MIX) {
      TF_RETURN_IF_ERROR(CalculateMixedLayoutPermutation(
          preferred_layout, input_labels, types, other));
    }

    if (layout == EinsumLayout::BFC) {
      offset_f = b;
      offset_c = f + b;
    } else {
      offset_f = b + c;
      offset_c = b;
    }

    dims = operand.GetTrtDims();
    for (int i = 0; i < b; i++) {
      // Set unknown batch dims to zero. These dims will be used in reshape op,
      // where zero is a special value for retaining the original dim size.
      if (dims.d[i] == -1) {
        dims.d[i] = 0;
      }
    }
    permuted_labels = input_labels;
    if (!permute.empty()) {
      // Apply the permutation on the dimension array.
      nvinfer1::Dims orig_dims = dims;
      for (int i = 0; i < permute.size(); i++) {
        dims.d[i] = orig_dims.d[permute[i]];
        permuted_labels[i] = input_labels[permute[i]];
      }
    }
    size_tensors.resize(dims.nbDims, nullptr);
    return Status::OK();
  }

 public:
  EinsumDescriptor() : b(0), f(0), c(0) {}

  // Deduces the number of batch, free, contract dimensions from the input
  // labels, decides what layout to use, and determines permutation indices for
  // that layout.
  static StatusOr<std::unique_ptr<EinsumDescriptor>> Create(
      const TRT_TensorOrWeights& operand, Labels input_labels,
      std::vector<EinsumDimensionType>& label_types,
      EinsumLayout preferred_layout,
      const std::unique_ptr<EinsumDescriptor>& other = nullptr) {
    auto desc = std::make_unique<EinsumDescriptor>();
    TF_RETURN_IF_ERROR(desc->Initialize(operand, input_labels, label_types,
                                        preferred_layout, other));
    VLOG(2) << desc->DebugString();
    return desc;
  }

  int NumBatchDims() const { return b; }
  int NumContractDims() const { return c; }
  int NumFreeDims() const { return f; }
  int ContractDimOffset() const { return offset_c; }
  const Labels& PermutedLabels() const { return permuted_labels; }

  std::string DebugString() const {
    return absl::StrCat("Descriptor with ",
                        (layout == EinsumLayout::BFC ? "BFC" : "BCF"),
                        " layout, b=", b, ", f=", f, ", c=", c);
  }

  // Returns whether the free and contract dimension have static shape.
  bool HasStaticShape() const {
    return !std::any_of(dims.d + b, dims.d + dims.nbDims,
                        [](int k) { return k == -1; });
  }

  nvinfer1::Permutation GetPermutation() const {
    nvinfer1::Permutation p;
    std::copy(permute.begin(), permute.end(), p.order);
    return p;
  }

  std::vector<int> PermuteVector() const { return permute; }

  // Sets the "size_tensors" vector to be filled with scalar constant tensors
  // representing the shape of the operand.
  Status SetDynamicSize(TRTNetworkBuilder* builder,
                        const TRT_TensorOrWeights& operand) {
    TRT_ENSURE(operand.GetTrtDims().nbDims == dims.nbDims);
    if (operand.is_weights()) {
      // Generate constants for each dimension of the constant weight tensor's
      // shape.
      for (int i = 0; i < operand.GetTrtDims().nbDims; i++) {
        StatusOr<nvinfer1::IConstantLayer*> size_tensor =
            builder->Constant<int32_t>(dims.d[i], 1);
        TRT_ENSURE_PTR_OK(size_tensor);
        size_tensors[i] = (*size_tensor)->getOutput(0);
      }
      return Status::OK();
    }

    // If the operand is a dynamic tensor, compute the shape value dynamically.
    StatusOr<nvinfer1::IShapeLayer*> shape_layer =
        builder->Shape(operand.tensor()->trt_tensor());
    TRT_ENSURE_PTR_OK(shape_layer);
    nvinfer1::ITensor* shape = (*shape_layer)->getOutput(0);
    for (int i = 0; i < operand.GetTrtDims().nbDims; i++) {
      int idx = permute.empty() ? i : permute.at(i);
      StatusOr<nvinfer1::ISliceLayer*> slice_layer =
          builder->Slice(shape, {1, {idx}}, {1, {1}}, {1, {1}});
      TRT_ENSURE_PTR_OK(slice_layer);
      size_tensors[i] = (*slice_layer)->getOutput(0);
    }
    return Status::OK();
  }

  EinsumLayout layout;
  int b;  // number of batch dims
  int f;  // number of free dims
  int c;  // number of conraction dims
  int offset_f;
  int offset_c;
  nvinfer1::Dims dims;
  std::vector<int> permute;
  std::vector<ITensorProxyPtr> size_tensors;
  Labels permuted_labels;
};

// Reshapes operand so that the free dimensions are combined into a single dim,
// and the contract dimensions are combined into another single dim.
Status GetEinsumNewDynamicShape(TRTNetworkBuilder* builder,
                                const EinsumDescriptor& desc,
                                ITensorProxyPtr* new_shape) {
  std::vector<nvinfer1::ITensor*> size;
  size.reserve(desc.b + 2);
  absl::c_transform(absl::MakeSpan(desc.size_tensors).subspan(0, desc.b + 2),
                    std::back_inserter(size),
                    [](const ITensorProxyPtr x) { return x->trt_tensor(); });

  int idx_f = desc.layout == EinsumLayout::BFC ? desc.b : desc.b + 1;
  int idx_c = desc.layout == EinsumLayout::BFC ? desc.b + 1 : desc.b;

  std::vector<nvinfer1::ITensor*> size_tensors;
  size_tensors.reserve(desc.size_tensors.size());
  absl::c_transform(desc.size_tensors, std::back_inserter(size_tensors),
                    [](const ITensorProxyPtr x) -> nvinfer1::ITensor* {
                      return x->trt_tensor();
                    });

  StatusOr<nvinfer1::ILayer*> shape_vol = builder->CumulativeProd(
      absl::MakeSpan(size_tensors).subspan(desc.offset_f, desc.f));
  TRT_ENSURE_PTR_OK(shape_vol);
  size[idx_f] = (*shape_vol)->getOutput(0);

  shape_vol = builder->CumulativeProd(
      absl::MakeSpan(size_tensors).subspan(desc.offset_c, desc.c));
  TRT_ENSURE_PTR_OK(shape_vol);
  size[idx_c] = (*shape_vol)->getOutput(0);
  StatusOr<nvinfer1::IConcatenationLayer*> layer =
      builder->Concat(size, /*axis=*/0);
  TRT_ENSURE_PTR_OK(layer);
  *new_shape = (*layer)->getOutput(0);
  return Status::OK();
}

// Reshapes operand so that the free dimensions are combined into a single dim,
// and the contract dimensions are combined into another single dim.
Status GetEinsumNewStaticShape(const EinsumDescriptor& desc,
                               nvinfer1::Dims* new_dims) {
  // Copy the batch dims and append two additional dimensions.
  DimsAdapter adap(
      absl::MakeSpan(static_cast<const int32_t*>(desc.dims.d), desc.b));
  adap.Append(1).Append(1);

  // Combine free dims and contract dims.
  int idx_f = desc.layout == EinsumLayout::BFC ? desc.b : desc.b + 1;
  int idx_c = desc.layout == EinsumLayout::BFC ? desc.b + 1 : desc.b;

  // Find the volume of the free dimensions.
  int64_t vol_f =
      DimsAdapter(
          absl::MakeSpan(
              static_cast<const int32_t*>(desc.dims.d) + desc.offset_f, desc.f))
          .Volume();

  // Find the volume of the contracted dimensions.
  int64_t vol_c =
      DimsAdapter(
          absl::MakeSpan(
              static_cast<const int32_t*>(desc.dims.d) + desc.offset_c, desc.c))
          .Volume();

  adap.dim(idx_f) = vol_f;
  adap.dim(idx_c) = vol_c;
  *new_dims = adap.AsTrtDims();
  return Status::OK();
}

StatusOr<TRT_TensorOrWeights> ConditionEinsumWeights(
    TRTNetworkBuilder* builder, const TRT_TensorOrWeights& operand,
    const EinsumDescriptor& desc, const bool need_transpose) {
  TRT_ENSURE(operand.is_weights());
  if (!need_transpose) {
    // If we don't need to transpose, then the operand remains as a weights
    // constant. In this case we also don't need a reshape.
    TRT_ShapedWeights weights(operand.weights());
    nvinfer1::Dims new_dims;
    TF_RETURN_IF_ERROR(GetEinsumNewStaticShape(desc, &new_dims));
    TF_RETURN_IF_ERROR(weights.SetShape(new_dims));
    return TRT_TensorOrWeights(weights);
  }

  // Let TensorRT handle constant folding where possible.
  StatusOr<nvinfer1::IConstantLayer*> tensor = builder->WeightsToConstant(
      operand.weights().GetTrtWeights(), operand.GetTrtDims());
  TRT_ENSURE_PTR_OK(tensor);
  return TRT_TensorOrWeights((*tensor)->getOutput(0));
}

// Builds a TRT shuffle operation for the given operand. Replaces operand with a
// pointer to the shuffle output.
Status ConditionEinsumTensor(TRTNetworkBuilder* builder,
                             std::unique_ptr<TRT_TensorOrWeights>* operand,
                             const EinsumDescriptor& desc,
                             const bool need_transpose,
                             const bool need_reshape) {
  StatusOr<ShuffleBuilder> shuffle =
      ShuffleBuilder::Create(builder, (*operand)->tensor()->trt_tensor());
  TRT_ENSURE_OK(shuffle);

  // Set new shape.
  if (need_reshape) {
    if (desc.HasStaticShape()) {
      nvinfer1::Dims new_dims;
      TF_RETURN_IF_ERROR(GetEinsumNewStaticShape(desc, &new_dims));
      shuffle->SetReshape(new_dims);
    } else {
      ITensorProxyPtr new_shape;
      TF_RETURN_IF_ERROR(GetEinsumNewDynamicShape(&*builder, desc, &new_shape));
      shuffle->SetReshape(new_shape->trt_tensor());
    }
  }

  if (need_transpose) {
    shuffle->SetFirstTranspose(desc.GetPermutation());
  }

  StatusOr<nvinfer1::ITensor*> shuffle_out = shuffle->Output();
  TRT_ENSURE_PTR_OK(shuffle_out);
  *operand = std::make_unique<TRT_TensorOrWeights>(*shuffle_out);
  return Status::OK();
}

// Handles einsum operand conditioning for both constant and non-constant
// inputs. This is supported using the ShuffleEinsumWeights and
// ShuffleEinsumTensor routines.
Status ConditionEinsumOperand(TRTNetworkBuilder* builder,
                              std::unique_ptr<TRT_TensorOrWeights>* operand,
                              const EinsumDescriptor& desc) {
  bool need_reshape = (desc.f != 1 || desc.c != 1);
  bool need_transpose = !desc.permute.empty();

  VLOG(2) << "Condition operand. Need reshape: " << need_reshape
          << ". Need transpose: " << need_transpose;

  if ((*operand)->is_weights()) {
    StatusOr<TRT_TensorOrWeights> result =
        ConditionEinsumWeights(builder, **operand, desc, need_transpose);
    TRT_ENSURE_OK(result);
    *operand =
        std::make_unique<TRT_TensorOrWeights>(result.ConsumeValueOrDie());
  }

  // If we didn't convert the operand to a tensor, we can return here.
  if ((*operand)->is_weights()) {
    return Status::OK();
  }

  TF_RETURN_IF_ERROR(ConditionEinsumTensor(builder, operand, desc,
                                           need_transpose, need_reshape));

  return Status::OK();
}

// Combines output dims/labels by copying batch and free dims/labels from input
// A, and concatenating free values from input B.
template <typename InputIterator, typename OutputIterator>
void AssembleOutput(InputIterator begin_a, InputIterator begin_b,
                    const EinsumDescriptor& desc_a,
                    const EinsumDescriptor& desc_b, OutputIterator out) {
  std::copy(begin_a, begin_a + desc_a.b, out);
  begin_a += desc_a.offset_f;
  std::copy(begin_a, begin_a + desc_a.f, out + desc_a.b);
  begin_b += desc_b.offset_f;
  std::copy(begin_b, begin_b + desc_b.f, out + desc_a.b + desc_a.f);
}

// Restores free dimensions and sets final index order. Consider C = A * B,
// batched MatMul op, where A.shape = [B, x, k] and B.shape = [B, k, y]. Then
// C.shape = [B, x, y]. Here B can denote multiple batch indices while x, y, k
// are single indices. The original inputs to Einsum can have multiple free
// indices. These were combined into a singe free dimension x and y, for example
// x = f_a1 * f_a2 * f_a3, y = f_b1 * f_b2. This routine creates a shuffle layer
// to expand x into and y the original free dims, e.g. C is reshaped to
// [B, f_a1, f_a2, f_a3, f_b1, f_b2]. Finally, a permutation is applied to
// transform the shape to the shape of the original Einsum output.
Status ShuffleEinsumOutput(OpConverterParams* params, EinsumDescriptor desc_a,
                           EinsumDescriptor desc_b,
                           const std::vector<int>& permutation,
                           ITensorProxyPtr* output) {
  if (permutation.empty() && (desc_a.f == 1 && desc_b.f == 1)) {
    return Status::OK();
  }

  nvinfer1::IShuffleLayer* layer =
      params->converter->network()->addShuffle(*(*output)->trt_tensor());
  TFTRT_RETURN_ERROR_IF_NULLPTR(layer, params->node_def.name());
  params->converter->SetLayerName(layer, params->node_def, "shuffle",
                                  /*sub_op_instance=*/2);

  int output_rank = desc_a.b + desc_a.f + desc_b.f;
  if (desc_a.f != 1 || desc_b.f != 1) {
    if (desc_a.HasStaticShape() && desc_b.HasStaticShape()) {
      nvinfer1::Dims dims_out = {output_rank, {}};
      AssembleOutput(desc_a.dims.d, desc_b.dims.d, desc_a, desc_b, dims_out.d);
      layer->setReshapeDimensions(dims_out);
    } else {
      std::vector<ITensorProxyPtr> size_tensors(output_rank);
      AssembleOutput(desc_a.size_tensors.begin(), desc_b.size_tensors.begin(),
                     desc_a, desc_b, size_tensors.begin());
      ITensorProxyPtr new_shape;
      auto builder = TRTNetworkBuilder::Create(params->converter->network(),
                                               params->weight_store);
      TRT_ENSURE_OK(builder);
      std::vector<nvinfer1::ITensor*> size_itensors;
      absl::c_transform(size_tensors, std::back_inserter(size_itensors),
                        [](auto x) { return x->trt_tensor(); });
      StatusOr<nvinfer1::IConcatenationLayer*> concat =
          builder->Concat(size_itensors, /*axis=*/0);
      TRT_ENSURE_PTR_OK(concat);
      new_shape = (*concat)->getOutput(0);
      layer->setInput(1, *new_shape->trt_tensor());
    }
  }

  if (!permutation.empty()) {
    nvinfer1::Permutation p;
    std::copy(permutation.begin(), permutation.end(), p.order);
    layer->setSecondTranspose(p);
  }
  *output = layer->getOutput(0);
  return Status::OK();
}

// Updates "final_transpose" according to the given descriptors and output
// labels.
StatusOr<std::vector<int>> GetOutputTranspose(
    const EinsumDescriptor& descriptor_a, const EinsumDescriptor& descriptor_b,
    Labels output_labels) {
  // Get final transpose.
  std::vector<int> final_transpose;
  final_transpose.reserve(descriptor_a.b + descriptor_a.f + descriptor_b.f);
  Labels matmul_output_labels(descriptor_a.b + descriptor_a.f + descriptor_b.f);
  AssembleOutput(descriptor_a.permuted_labels.begin(),
                 descriptor_b.permuted_labels.begin(), descriptor_a,
                 descriptor_b, matmul_output_labels.begin());
  TF_RETURN_IF_ERROR(
      FindIndicesoOfAllValuesInSrc(/*values=*/
                                   absl::MakeConstSpan(output_labels.begin(),
                                                       output_labels.end()),
                                   /*src=*/
                                   absl::MakeConstSpan(
                                       matmul_output_labels.begin(),
                                       matmul_output_labels.end()),
                                   /*indices=*/&final_transpose));
  // Clear identity transpose.
  bool identity_transpose = true;
  for (int i = 0; i < final_transpose.size() && identity_transpose; i++) {
    identity_transpose &= final_transpose.at(i) == i;
  }
  if (identity_transpose) {
    final_transpose.clear();
  }
  return final_transpose;
}

// Prepares EinsumDescriptors after parsing the equation and determines the
// final transpose.
Status ParseEquation(const std::string& equation,
                     std::unique_ptr<TRT_TensorOrWeights>* input_a,
                     std::unique_ptr<TRT_TensorOrWeights>* input_b,
                     std::unique_ptr<EinsumDescriptor>* descriptor_a,
                     std::unique_ptr<EinsumDescriptor>* descriptor_b,
                     std::vector<int>* final_transpose) {
  VLOG(2) << "Einsum equation " << equation;
  OperandLabels input_labels;
  Labels output_labels;
  std::vector<EinsumDimensionType> label_types;
  OperandLabelCounts input_label_counts;
  LabelCounts output_label_counts;
  absl::InlinedVector<bool, 2> input_has_ellipsis;
  bool output_has_ellipsis;
  TF_RETURN_IF_ERROR(
      ParseEinsumEquation(equation, &input_labels, &output_labels, &label_types,
                          &input_label_counts, &output_label_counts,
                          &input_has_ellipsis, &output_has_ellipsis));

  if (input_has_ellipsis[0] || input_has_ellipsis[1] || output_has_ellipsis) {
    // TODO(tfeher): Handle ellipsis like EinsumHelper::ProcessDimensions.
    // Note: ProcessDimensions would introduce kBroadcasting labels, which we
    // need to replace with kBatch before we call InitDescriptor.
    VLOG(2) << "Ellipsis not yet supported";
    return errors::Unimplemented("No conversion for einsum equation.");
  }

  if (absl::c_any_of(label_types, [](auto l) {
        return l == EinsumDimensionType::kReduce ||
               l == EinsumDimensionType::kBroadcasting;
      })) {
    VLOG(2) << "Einsum reductions not implemented";
    return errors::Unimplemented("No conversion for einsum equation.");
  }

  auto no_duplicated_labels = [](const LabelCounts& label_counts) {
    return absl::c_any_of(label_counts, [](int i) { return i > 1; });
  };
  if (no_duplicated_labels(input_label_counts[0]) ||
      no_duplicated_labels(input_label_counts[1]) ||
      no_duplicated_labels(output_label_counts)) {
    VLOG(2) << "Einsum invalid label count";
    return errors::Unimplemented("No conversion for einsum equation.");
  }

  if ((*input_a)->is_weights() && (*input_b)->is_tensor()) {
    // We prefer to use FC layer, needs A as tensor and B as weight.
    std::swap(*input_a, *input_b);
    std::swap(input_labels[0], input_labels[1]);
    std::swap(input_label_counts[0], input_label_counts[1]);
  }

  auto desc = EinsumDescriptor::Create(**input_a, input_labels[0], label_types,
                                       EinsumLayout::BFC);
  TF_RETURN_IF_ERROR(desc.status());
  *descriptor_a = desc.ConsumeValueOrDie();

  desc = EinsumDescriptor::Create(**input_b, input_labels[1], label_types,
                                  EinsumLayout::BCF, *descriptor_a);
  TF_RETURN_IF_ERROR(desc.status());
  *descriptor_b = desc.ConsumeValueOrDie();

  auto out_transpose =
      GetOutputTranspose(**descriptor_a, **descriptor_b, output_labels);

  TRT_ENSURE_OK(out_transpose)
  *final_transpose = out_transpose.ConsumeValueOrDie();
  return Status::OK();
}

class ConvertEinsum : public OpConverterBase<ConvertEinsum> {
 public:
  explicit ConvertEinsum(OpConverterParams* params)
      : OpConverterBase<ConvertEinsum>(params) {}

  static constexpr std::array<DataType, 3> AllowedDataTypes() {
    return {DataType::DT_FLOAT, DataType::DT_HALF};
  }

  static constexpr std::array<InputArgSpec, 2> InputSpec() {
    return {InputArgSpec::Create("input_a", TrtInputArg::kBoth),
            InputArgSpec::Create("input_b", TrtInputArg::kBoth)};
  }

  Status Validate() {
    const auto& inputs = params_->inputs;
    if (params_->use_implicit_batch) {
      return errors::Unimplemented(
          "Einsum converter requires dynamic shape mode");
    }

    input_a = std::make_unique<TRT_TensorOrWeights>(inputs.at(0));
    input_b = std::make_unique<TRT_TensorOrWeights>(inputs.at(1));

    StatusOr<std::string> eq = GetAttrValue<std::string>("equation");
    TRT_ENSURE_OK(eq);
    TF_RETURN_IF_ERROR(ParseEquation(*eq, &input_a, &input_b, &descriptor_a,
                                     &descriptor_b, &final_transpose));

    return Status::OK();
  }

  Status Convert() {
    auto builder = TRTNetworkBuilder::Create(params_->converter->network(),
                                             params_->weight_store);
    TRT_ENSURE_OK(builder);
    TRT_ENSURE(input_a && input_b);
    TRT_ENSURE(descriptor_a && descriptor_b);

    // Populate the size_tensor vector in the descriptor.
    TF_RETURN_IF_ERROR(descriptor_a->SetDynamicSize(&*builder, *input_a));
    TF_RETURN_IF_ERROR(descriptor_b->SetDynamicSize(&*builder, *input_b));

    // Condition the operands for lowering to matmul.
    TF_RETURN_IF_ERROR(
        ConditionEinsumOperand(&*builder, &input_a, *descriptor_a));
    TF_RETURN_IF_ERROR(
        ConditionEinsumOperand(&*builder, &input_b, *descriptor_b));

    // Build the matmul implementation.
    StatusOr<ITensorProxyPtr> result = ConvertMatMulImpl(
        params_, *input_a, *input_b, descriptor_a->layout == EinsumLayout::BCF,
        descriptor_b->layout == EinsumLayout::BFC);
    TF_RETURN_IF_ERROR(result.status());
    ITensorProxyPtr output = result.ValueOrDie();

    // Reshape and permute the output.
    TF_RETURN_IF_ERROR(ShuffleEinsumOutput(
        params_, *descriptor_a, *descriptor_b, final_transpose, &output));
    this->AddOutput(output);
    return Status::OK();
  }

 private:
  std::unique_ptr<TRT_TensorOrWeights> input_a{nullptr};
  std::unique_ptr<TRT_TensorOrWeights> input_b{nullptr};
  std::vector<int> final_transpose;
  std::unique_ptr<EinsumDescriptor> descriptor_a{nullptr};
  std::unique_ptr<EinsumDescriptor> descriptor_b{nullptr};
};

}  // namespace

REGISTER_DEFAULT_TRT_OP_CONVERTER(MakeConverterFunction<ConvertEinsum>(),
                                  "Einsum");
#endif

}  // namespace convert
}  // namespace tensorrt
}  // namespace tensorflow

#endif  // GOOGLE_CUDA && GOOGLE_TENSORRT
