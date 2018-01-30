#include "tensorflow/contrib/lite/kernels/ssd_ops.h"

#include <chrono>
#include <iostream>
#include <tuple>
#include <vector>

#include "tensorflow/contrib/lite/context.h"
#include "tensorflow/contrib/lite/kernels/internal/tensor.h"
#include "tensorflow/contrib/lite/kernels/internal/types.h"
#include "tensorflow/contrib/lite/kernels/kernel_util.h"
#include "tensorflow/contrib/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/contrib/lite/kernels/internal/reference/reference_ops.h"

namespace tflite {
namespace ops {
namespace custom {

// NEON takes precedence, then EIGEN. If both commented,
// then reference is used.
//#define SSD_USE_NEON
#define SSD_USE_EIGEN

inline void Exp(const float* input_data, const Dims<4>& input_dims,
                float* output_data, const Dims<4>& output_dims) {

#ifdef SSD_USE_EIGEN

  using tflite::optimized_ops::MapAsVector;

  // Implementation using EIGEN
  auto input_map = MapAsVector(input_data, input_dims);
  auto output_map = MapAsVector(output_data, output_dims);
  output_map.array() = input_map.array().exp();

#else
  // Get sizes
  const int batches = MatchingArraySize(input_dims, 3, output_dims, 3);
  const int height = MatchingArraySize(input_dims, 2, output_dims, 2);
  const int width = MatchingArraySize(input_dims, 1, output_dims, 1);
  const int depth = MatchingArraySize(input_dims, 0, output_dims, 0);

  // Loop
  for (int b = 0; b < batches; ++b) {
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        for (int c = 0; c < depth; ++c) {
          // Get offsets
          int input_offset = Offset(input_dims, c, x, y, b);
          int output_offset = Offset(output_dims, c, x, y, b);

          // Compute exp
          output_data[output_offset] = std::exp(input_data[input_offset]);
        }
      }
    }
  }
#endif // SSD_USE_EIGEN
}

inline void Scale(const float* input_data, const Dims<4>& input_dims,
                  float scale,
                  float* output_data, const Dims<4>& output_dims) {
#ifdef SSD_USE_EIGEN

  using tflite::optimized_ops::MapAsVector;

  // Implementation using EIGEN
  auto input_map = MapAsVector(input_data, input_dims);
  auto output_map = MapAsVector(output_data, output_dims);
  output_map.array() = input_map.array() * scale;

#else
   // Get sizes
  const int batches = MatchingArraySize(input_dims, 3, output_dims, 3);
  const int height = MatchingArraySize(input_dims, 2, output_dims, 2);
  const int width = MatchingArraySize(input_dims, 1, output_dims, 1);
  const int depth = MatchingArraySize(input_dims, 0, output_dims, 0);

  // Loop
  for (int b = 0; b < batches; ++b) {
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        for (int c = 0; c < depth; ++c) {
          // Get offsets
          int input_offset = Offset(input_dims, c, x, y, b);
          int output_offset = Offset(output_dims, c, x, y, b);

          // scale values
          output_data[output_offset] = input_data[input_offset] * scale;
        }
      }
    }
  }
#endif // SSD_USE_EIGEN
}

inline void ClipMinMax(const float* input_data, const Dims<4>& input_dims,
                       float min_val, float max_val,
                       float* output_data, const Dims<4>& output_dims) {
#ifdef SSD_USE_EIGEN

  using tflite::optimized_ops::MapAsVector;

  // Implementation using EIGEN
  auto input_map = MapAsVector(input_data, input_dims);
  auto output_map = MapAsVector(output_data, output_dims);
  output_map = input_map.cwiseMax(min_val).cwiseMin(max_val);

#else
   // Get sizes
  const int batches = MatchingArraySize(input_dims, 3, output_dims, 3);
  const int height = MatchingArraySize(input_dims, 2, output_dims, 2);
  const int width = MatchingArraySize(input_dims, 1, output_dims, 1);
  const int depth = MatchingArraySize(input_dims, 0, output_dims, 0);

  // Loop
  for (int b = 0; b < batches; ++b) {
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        for (int c = 0; c < depth; ++c) {
          // Get offsets
          int input_offset = Offset(input_dims, c, x, y, b);
          int output_offset = Offset(output_dims, c, x, y, b);

          // scale values
          output_data[output_offset] = std::min(std::max(input_data[input_offset],
                                                         min_val),
                                                max_val);
        }
      }
    }
  }
#endif // SSD_USE_EIGEN
}


inline void Mul(const float* input1_data, const Dims<4>& input1_dims,
                const float* input2_data, const Dims<4>& input2_dims,
                float* output_data, const Dims<4>& output_dims) {

#ifdef SSD_USE_NEON

  // Activation min/max
  // TODO(maly): Make more generic
  float ac_min = std::numeric_limits<float>::min();
  float ac_max = std::numeric_limits<float>::max();

  tflite::reference_ops::Mul(input1_data, input1_dims,
          input2_data, input2_dims,
          ac_min, ac_max,
          output_data, output_dims);

#elif defined(SSD_USE_EIGEN)

  using tflite::optimized_ops::MapAsVector;

  // Implementation using EIGEN
  auto input1_map = MapAsVector(input1_data, input1_dims);
  auto input2_map = MapAsVector(input2_data, input2_dims);

  auto output_map = MapAsVector(output_data, output_dims);

  output_map = input1_map.cwiseProduct(input2_map);

#else

  const int batches =
      MatchingArraySize(input1_dims, 3, input2_dims, 3, output_dims, 3);
  const int height =
      MatchingArraySize(input1_dims, 2, input2_dims, 2, output_dims, 2);
  const int width =
      MatchingArraySize(input1_dims, 1, input2_dims, 1, output_dims, 1);
  const int depth =
      MatchingArraySize(input1_dims, 0, input2_dims, 0, output_dims, 0);

  for (int b = 0; b < batches; ++b) {
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        for (int c = 0; c < depth; ++c) {
          // Get offsets
          int input_offset = Offset(input1_dims, c, x, y, b);
          int output_offset = Offset(output_dims, c, x, y, b);

          // scale values
          output_data[output_offset] = input1_data[input_offset] *
                                       input2_data[input_offset];
        }
      }
    }
  }
#endif
}

inline void Add(const float* input1_data, const Dims<4>& input1_dims,
                const float* input2_data, const Dims<4>& input2_dims,
                float* output_data, const Dims<4>& output_dims) {

#ifdef SSD_USE_NEON

  // Activation min/max
  // TODO(maly): Make more generic
  float ac_min = std::numeric_limits<float>::min();
  float ac_max = std::numeric_limits<float>::max();

  tflite::reference_ops::Add(input1_data, input1_dims,
          input2_data, input2_dims,
          ac_min, ac_max,
          output_data, output_dims);

#elif defined(SSD_USE_EIGEN)

  using tflite::optimized_ops::MapAsVector;

  // Implementation using EIGEN
  auto input1_map = MapAsVector(input1_data, input1_dims);
  auto input2_map = MapAsVector(input2_data, input2_dims);

  auto output_map = MapAsVector(output_data, output_dims);

  output_map = input1_map + input2_map;

#else

  const int batches =
      MatchingArraySize(input1_dims, 3, input2_dims, 3, output_dims, 3);
  const int height =
      MatchingArraySize(input1_dims, 2, input2_dims, 2, output_dims, 2);
  const int width =
      MatchingArraySize(input1_dims, 1, input2_dims, 1, output_dims, 1);
  const int depth =
      MatchingArraySize(input1_dims, 0, input2_dims, 0, output_dims, 0);

  for (int b = 0; b < batches; ++b) {
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        for (int c = 0; c < depth; ++c) {
          // Get offsets
          int input_offset = Offset(input1_dims, c, x, y, b);
          int output_offset = Offset(output_dims, c, x, y, b);

          // scale values
          output_data[output_offset] = input1_data[input_offset] +
                                       input2_data[input_offset];
        }
      }
    }
  }
#endif
}

inline void Sub(const float* input1_data, const Dims<4>& input1_dims,
                const float* input2_data, const Dims<4>& input2_dims,
                float* output_data, const Dims<4>& output_dims) {
#ifdef SSD_USE_NEON

  // Activation min/max
  // TODO(maly): Make more generic
  float ac_min = std::numeric_limits<float>::min();
  float ac_max = std::numeric_limits<float>::max();

  tflite::reference_ops::Sub(input1_data, input1_dims,
          input2_data, input2_dims,
          ac_min, ac_max,
          output_data, output_dims);

#elif defined(SSD_USE_EIGEN)

  using tflite::optimized_ops::MapAsVector;

  // Implementation using EIGEN
  auto input1_map = MapAsVector(input1_data, input1_dims);
  auto input2_map = MapAsVector(input2_data, input2_dims);

  auto output_map = MapAsVector(output_data, output_dims);

  output_map = input1_map - input2_map;

#else

  const int batches =
      MatchingArraySize(input1_dims, 3, input2_dims, 3, output_dims, 3);
  const int height =
      MatchingArraySize(input1_dims, 2, input2_dims, 2, output_dims, 2);
  const int width =
      MatchingArraySize(input1_dims, 1, input2_dims, 1, output_dims, 1);
  const int depth =
      MatchingArraySize(input1_dims, 0, input2_dims, 0, output_dims, 0);

  for (int b = 0; b < batches; ++b) {
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        for (int c = 0; c < depth; ++c) {
          // Get offsets
          int input_offset = Offset(input1_dims, c, x, y, b);
          int output_offset = Offset(output_dims, c, x, y, b);

          // scale values
          output_data[output_offset] = input1_data[input_offset] -
                                       input2_data[input_offset];
        }
      }
    }
  }
#endif
}

// Copied from reference_ops.h
inline void Logistic(const float* input_data, const Dims<4>& input_dims,
                     float* output_data, const Dims<4>& output_dims) {

#ifdef SSD_USE_NEON

  tflite::optimized_ops::Logistic(input_data, input_dims,
                                  output_data, output_dims);

#elif defined(SSD_USE_EIGEN)

  using tflite::optimized_ops::MapAsVector;

  // Implementation using EIGEN
  auto input_map = MapAsVector(input_data, input_dims);
  auto output_map = MapAsVector(output_data, output_dims);

  output_map.array() = input_map.array().unaryExpr(
          Eigen::internal::scalar_sigmoid_op<float>());

  //output_map.array() = 1.f / (1.f + (-input_map.array()).exp());

#else

  const int batches = MatchingArraySize(input_dims, 3, output_dims, 3);
  const int height = MatchingArraySize(input_dims, 2, output_dims, 2);
  const int width = MatchingArraySize(input_dims, 1, output_dims, 1);
  const int depth = MatchingArraySize(input_dims, 0, output_dims, 0);
  for (int b = 0; b < batches; ++b) {
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        for (int c = 0; c < depth; ++c) {
          float val = input_data[Offset(input_dims, c, x, y, b)];
          float result = 1.f / (1.f + std::exp(-val));
          output_data[Offset(output_dims, c, x, y, b)] = result;
        }
      }
    }
  }
#endif
}

// Copied from: tensorflow/core/kernels/non_max_suppression_op.cc
inline void DecreasingArgSort(const float* values, int size,
                              std::vector<int>* indices) {
  // resize just in case
  indices->resize(size);
  // fill in indices
  for (int i = 0; i < size; ++i) (*indices)[i] = i;
  // sort
  std::sort(
      indices->begin(), indices->end(),
      [&values](const int i, const int j) { return values[i] > values[j]; });
}


// Returns sorted indices
inline void DecreasingArgSort(const std::vector<float>& values,
                              std::vector<int>* indices) {
  DecreasingArgSort(values.data(), values.size(), indices);
}

// Return true if intersection-over-union overlap between boxes i and j
// is greater than iou_threshold.
// Copied from: tensorflow/core/kernels/non_max_suppression_op.cc
inline bool IOUGreaterThanThreshold(
    float* ymin, float* xmin, float* ymax, float* xmax,
    int i, int j, float iou_threshold) {
  const float ymin_i = std::min<float>(ymin[i], ymax[i]);
  const float xmin_i = std::min<float>(xmin[i], xmax[i]);
  const float ymax_i = std::max<float>(ymin[i], ymax[i]);
  const float xmax_i = std::max<float>(xmin[i], xmax[i]);

  const float ymin_j = std::min<float>(ymin[j], ymax[j]);
  const float xmin_j = std::min<float>(xmin[j], xmax[j]);
  const float ymax_j = std::max<float>(ymin[j], ymax[j]);
  const float xmax_j = std::max<float>(xmin[j], xmax[j]);

  const float area_i = (ymax_i - ymin_i) * (xmax_i - xmin_i);
  const float area_j = (ymax_j - ymin_j) * (xmax_j - xmin_j);
  if (area_i <= 0 || area_j <= 0) return false;

  const float intersection_ymin = std::max<float>(ymin_i, ymin_j);
  const float intersection_xmin = std::max<float>(xmin_i, xmin_j);
  const float intersection_ymax = std::min<float>(ymax_i, ymax_j);
  const float intersection_xmax = std::min<float>(xmax_i, xmax_j);

  const float intersection_area =
      std::max<float>(intersection_ymax - intersection_ymin, 0.0) *
      std::max<float>(intersection_xmax - intersection_xmin, 0.0);

  const float iou = intersection_area / (area_i + area_j - intersection_area);

  return iou > iou_threshold;
}


namespace ssd {

// Get pointers for y, x, h, w of input tensor in specific batch
// Tensor shape is assumed b x D x A
// b defaults to the first batch i.e. b = 0
// The function returns 4 points to each of the D tensors of length A
// i.e. a tensor of shape 1x1xA
std::tuple<float*, float*, float*, float*, Dims<4>>
    get_single_pointers(TfLiteTensor* tensor, int b = 0) {

  auto tensor_data = GetTensorData<float>(tensor);
  auto tensor_dims = GetTensorDims(tensor);

  float* y = tensor_data + tensor_dims.strides[1] * 0 +
                           tensor_dims.strides[2] * b;
  float* x = tensor_data + tensor_dims.strides[1] * 1 +
                           tensor_dims.strides[2] * b;
  float* h = tensor_data + tensor_dims.strides[1] * 2 +
                           tensor_dims.strides[2] * b;
  float* w = tensor_data + tensor_dims.strides[1] * 3 +
                           tensor_dims.strides[2] * b;
  Dims<4> dims = GetTensorDims({1, 1, tensor->dims->data[2]});

  return std::make_tuple(y, x, h, w, dims);
}

// Prints out the dimensions of a tensor t
#define PRINT_DIMS(t)
/*  std::cerr << #t << ": "; \*/
  //for (int i = 0; i < NumDimensions(t); ++i) \
      //std::cerr << t->dims->data[i] << " "; \
/*  s*/td::cerr << std::endl

#define PRINT_DIM4(t)
/*  std::cerr << #t << ": "; \*/
  //for (int i = 0; i < 4; ++i) \
      //std::cerr << t.sizes[i] << "&" << t.strides[i] << ", "; \
/*  s*/td::cerr << std::endl

#define PRINT_VEC(t, l)
/*  std::cerr << #t << ": "; \*/
  //for (int i = 0; i < l; ++i) { \
      //std::cerr << (t)[i] << " "; \
  //} \
  //std::cerr << std::endl

#define EXTRACT_SINGLE(tensor, b) \
  float * tensor##_y_data, * tensor##_x_data,  \
        * tensor##_h_data, * tensor##_w_data; \
  Dims<4> tensor##_single_dims; \
  std::tie(tensor##_y_data, tensor##_x_data, \
           tensor##_h_data, tensor##_w_data, \
           tensor##_single_dims) = \
    get_single_pointers(tensor , (b))

#define GET_TIME(t1, t2) \
  std::chrono::duration_cast<std::chrono::microseconds>( \
                (t2) - (t1)).count() / 1000.

#define PRINT_TIME2(str, t1, t2) \
    std::cerr << (str) << " took: " << GET_TIME((t1), (t2)) \
        << " msec.\n"

#define PRINT_TIME1(str, t) \
    std::cerr << (str) << " took: " << (t) << " msecs.\n"

#define START_TIMER(val) \
    auto val##_start = std::chrono::steady_clock::now()
#define STOP_TIMER(val) \
    auto val##_stop = std::chrono::steady_clock::now()
#define PRINT_TIMER(val) \
    std::cerr << #val << " took: " << \
    GET_TIME((val##_start), (val##_stop)) << " msec.\n"


// Copied over from tensorflow/contrib/lite/kernels/svdf.cc
void* Init(TfLiteContext* context,
           const char* buffer, size_t length) {
  auto* scratch_tensor_index = new int;
  context->AddTensors(context, 1, scratch_tensor_index);
  return scratch_tensor_index;
}

void Free(TfLiteContext* context, void* buffer) {
  delete reinterpret_cast<int*>(buffer);
}

// Scales are for: y, x, h, w
// TODO(maly): Add to graph and read into attr
const float kBoxScales[] = {0.1f, 0.1f, 0.2f, 0.2f};

// Bounding Box Postprocessing
TfLiteStatus PostprocessBoxesPrepare(TfLiteContext* context, TfLiteNode* node) {
  std::cerr << std::endl << "Inside PostprocessBoxesPrepare" << std::endl;

  // Takes 2 inputs:
  // 0 -> concat (which contains the output of the box encoders)
  //    with shape ? x 4 x A
  // 1 -> anchors with shape (1 x 4 x A) where
  //    A: number of anchors and the second dimension has
  //    height, width, y_center, x_center
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 2);
  // 1 output: the bounding boxes in format [ymin, xmin, ymax, xmas]
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  // Get inputs and outputs
  TfLiteTensor* box_in = GetInput(context, node, 0);
  TfLiteTensor* anchors = GetInput(context, node, 1);
  TfLiteTensor* box_out = GetOutput(context, node, 0);

  // Checks dimensions of anchors
  TF_LITE_ENSURE_EQ(context, box_in->dims->size, 3);
  TF_LITE_ENSURE_EQ(context, anchors->dims->size, 3);
  TF_LITE_ENSURE_EQ(context, box_in->dims->data[1],
                             anchors->dims->data[1]);
  TF_LITE_ENSURE_EQ(context, box_in->dims->data[2],
                             anchors->dims->data[2]);
  // anchors have dimension 1x4xA
  TF_LITE_ENSURE_EQ(context, anchors->dims->data[0], 1);
  TF_LITE_ENSURE_EQ(context, anchors->dims->data[1], 4);

  // Resize output to ? x 4 x A (same as box_in)
  int num_dims = NumDimensions(box_in);
  TfLiteIntArray* output_size = TfLiteIntArrayCreate(num_dims);
  for (int i=0; i<num_dims; ++i) {
    output_size->data[i] = box_in->dims->data[i];
  }

  // Resize scratch: to be 1 x 4 x A to hold temporary results
  // (same for each batch) same size as the anchors
  int* scratch_tensor_index = reinterpret_cast<int*>(node->user_data);
  TfLiteIntArrayFree(node->temporaries);
  node->temporaries = TfLiteIntArrayCreate(1);
  node->temporaries->data[0] = *scratch_tensor_index;

  TfLiteIntArray* scratch_size_array = TfLiteIntArrayCreate(3);
  scratch_size_array->data[0] = anchors->dims->data[0];
  scratch_size_array->data[1] = anchors->dims->data[1];
  scratch_size_array->data[2] = anchors->dims->data[2];

  TfLiteTensor* scratch_tensor = &context->tensors[node->temporaries->data[0]];
  scratch_tensor->type = anchors->type;
  scratch_tensor->allocation_type = kTfLiteArenaRw;
  TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, scratch_tensor,
                                                   scratch_size_array));

  PRINT_DIMS(box_in);
  PRINT_DIMS(anchors);

  float* ad = anchors->data.f;
  for (int i = 0; i < 5; ++i) std::cerr << ad[i + 0 * anchors->dims->data[2]] << " ";
  std::cerr << std::endl;

  auto dims = GetTensorDims(anchors);
  std::cerr << "dims: ";
  for (int i = 0; i < 4; ++i)
      std::cerr << dims.sizes[i] << " & " << dims.strides[i] << ", ";
  std::cerr << std::endl;

  for (int i = 0; i < 5; ++i)
    // get first 4 heights i.e. anchors[0, 1, i], but to do that, we need
    // to pass in reverse order [i, 1, 0]
    std::cerr << ad[Offset(dims, i, 1, 0, 0)] << " ";
  std::cerr << std::endl;

  // Allocate another tensor the same size as box_in for temp

  return context->ResizeTensor(context, box_out, output_size);
}

TfLiteStatus PostprocessBoxesEval(TfLiteContext* context, TfLiteNode* node) {
  std::cerr << std::endl << "Inside PostprocessBoxesEval new!!" << std::endl;
  START_TIMER(PostprocessBoxes);

  // Get inputs and outputs
  //
  // shape B x 4 x A
  // dimensions are: y, x, h, w
  TfLiteTensor* box_in = GetInput(context, node, 0);
  // shape 1 x 4 x A
  // dimensions are: y_center, x_center, height, width
  TfLiteTensor* anchors = GetInput(context, node, 1);
  // shape B x 4 x A
  // dimensions are: y_min, x_min, y_max, x_max
  TfLiteTensor* box_out = GetOutput(context, node, 0);
  // Scratch with sahpe 1 x 4 x A
  TfLiteTensor* scratch = &context->tensors[node->temporaries->data[0]];

  // Get data
  const float* box_in_data = GetTensorData<float>(box_in);
  const float* anchors_data = GetTensorData<float>(anchors);
  float* box_out_data = GetTensorData<float>(box_out);
  float* scratch_data = GetTensorData<float>(scratch);

  // Get dimensions
  // Recall: dims are in reverse order
  auto box_dims = GetTensorDims(box_in);
  auto anchor_dims = GetTensorDims(anchors);
  auto scratch_dims = GetTensorDims(scratch);
  // Number of boxes
  int A = box_in->dims->data[2];
  // Number of images (batch)
  int B = box_in->dims->data[0];
  // 4 dimensions per box
  const int D = 4;

  // Scales are for: y, x, h, w
  // TODO(maly): read from params
  std::vector<float> scales({kBoxScales[0], kBoxScales[1],
                             kBoxScales[2], kBoxScales[3]});

  // Extract scratch
  EXTRACT_SINGLE(scratch, 0);
  PRINT_DIMS(scratch);
  PRINT_DIM4(scratch_single_dims);

  // Extract anchors
  EXTRACT_SINGLE(anchors, 0);
  PRINT_DIM4(anchors_single_dims);
  PRINT_VEC(anchors_y_data, 5);
  PRINT_VEC(anchors_x_data, 5);
  PRINT_VEC(anchors_h_data, 5);
  PRINT_VEC(anchors_w_data, 5);

 // Loop on batch
  for (int b = 0; b < B; ++b) {

    // Extract box_in & box_out
    EXTRACT_SINGLE(box_in, b);
    EXTRACT_SINGLE(box_out, b);
    PRINT_DIM4(box_in_single_dims);
    PRINT_DIM4(box_out_single_dims);

    PRINT_VEC(box_in_y_data, 5);
    PRINT_VEC(box_in_x_data, 5);
    PRINT_VEC(box_in_h_data, 5);
    PRINT_VEC(box_in_w_data, 5);

    // Multiply by scales
    //      ty /= self._scale_factors[0]
    //      tx /= self._scale_factors[1]
    //      th /= self._scale_factors[2]
    //      tw /= self._scale_factors[3]
    /*
    for (int d = 0; d < D; ++d) {
      // get scale factor for this dimension
      const float scale = scales[d];

      for (int a = 0; a < A; ++a) {
        // get offsets (in reverse)
        int box_offset = Offset(box_dims, a, d, b, 0);
        int scratch_offset = Offset(scratch_dims, a, d, 0, 0);

        box_out_data[scratch_offset] = box_in_data[box_offset] * scale;
      }
    }
    */
    Scale(box_in_y_data, box_in_single_dims, scales[0],
          scratch_y_data, scratch_single_dims);
    PRINT_VEC(box_in_y_data, 5);
    PRINT_VEC(scratch_y_data, 5);
    Scale(box_in_x_data, box_in_single_dims, scales[1],
          scratch_x_data, scratch_single_dims);
    PRINT_VEC(box_in_x_data, 5);
    PRINT_VEC(scratch_x_data, 5);
    Scale(box_in_h_data, box_in_single_dims, scales[2],
          scratch_h_data, scratch_single_dims);
    PRINT_VEC(box_in_h_data, 5);
    PRINT_VEC(scratch_h_data, 5);
    Scale(box_in_w_data, box_in_single_dims, scales[3],
          scratch_w_data, scratch_single_dims);

    // Compute output width & height: in scratch dim (2,3)
    //      w = tf.exp(tw) * wa
    //      h = tf.exp(th) * ha
    //
    /*
    for (int d = 2; d < 4; ++d) {
      for (int a = 0; a < A; ++a) {
        // get offset (in reverse)
        int offset = Offset(scratch_offset, a, d, 0, 0);
        // For width&height (in dims 2&3) perform
        // w = exp(w) * anchor_width
        box_out_data[offset] = std::exp(scratch_data[offset]) *
                               anchors_data[offset];
      }
    }
    */
    // Compute exp(tw & th) in scratch (dim 2 & 3)
    std::cerr << "Exp\n";
    PRINT_VEC(scratch_h_data, 5);
    Exp(scratch_h_data, scratch_single_dims, scratch_h_data, scratch_single_dims);
    PRINT_VEC(scratch_h_data, 5);
    PRINT_VEC(scratch_w_data, 5);
    Exp(scratch_w_data, scratch_single_dims, scratch_w_data, scratch_single_dims);
    PRINT_VEC(scratch_w_data, 5);
    // Mul with anchor w & h
    std::cerr << "Mul\n";
    PRINT_VEC(scratch_h_data, 5);
    PRINT_VEC(anchors_h_data, 5);
    Mul(scratch_h_data, scratch_single_dims, anchors_h_data, anchors_single_dims,
        scratch_h_data, scratch_single_dims);
    PRINT_VEC(scratch_h_data, 5);
    Mul(scratch_w_data, scratch_single_dims, anchors_w_data, anchors_single_dims,
        scratch_w_data, scratch_single_dims);

    // Compute x_center & y_center in scratch dim (0, 1)
    //      ycenter = ty * ha + ycenter_a
    //      xcenter = tx * wa + xcenter_a
    /*
    for (int d = 0; d < 2; ++d) {
      for (int a = 0; a < A; ++a) {
        // get offset (in reverse)
        int anchor_offset = Offset(anchor_dims, a, d, 0, 0);
        // For width&height (in dims 2&3) perform
        // w = exp(w) * anchor_width
        box_out_data[offset] = std::exp(scratch_data[offset]) *
                               anchors_data[offset];
      }
    }
    */
    // Mul scratch_y * anchors_h then add anchors_y
    std::cerr << "Mul & Add\n";
    PRINT_VEC(scratch_y_data, 5);
    PRINT_VEC(anchors_h_data, 5);
    Mul(scratch_y_data, scratch_single_dims, anchors_h_data, anchors_single_dims,
        scratch_y_data, scratch_single_dims);
    PRINT_VEC(scratch_y_data, 5);
    PRINT_VEC(anchors_y_data, 5);
    Add(scratch_y_data, scratch_single_dims, anchors_y_data, anchors_single_dims,
        scratch_y_data, scratch_single_dims);
    PRINT_VEC(scratch_y_data, 5);
    // Same for scratch_x
    PRINT_VEC(scratch_x_data, 5);
    PRINT_VEC(anchors_w_data, 5);
    Mul(scratch_x_data, scratch_single_dims, anchors_w_data, anchors_single_dims,
        scratch_x_data, scratch_single_dims);
    PRINT_VEC(scratch_x_data, 5);
    PRINT_VEC(anchors_x_data, 5);
    Add(scratch_x_data, scratch_single_dims, anchors_x_data, anchors_single_dims,
        scratch_x_data, scratch_single_dims);
    PRINT_VEC(scratch_x_data, 5);

    // Compute y_min/max and x_min/max in box_out from scratch y, x, h, w
    //
    // Scale scratch_h & w by 0.5
    Scale(scratch_h_data, scratch_single_dims, 0.5f,
          scratch_h_data, scratch_single_dims);
    Scale(scratch_w_data, scratch_single_dims, 0.5f,
          scratch_w_data, scratch_single_dims);

    // y_min = scratch_y - scratch_h / 2 (dim 0 i.e. y)
    Sub(scratch_y_data, scratch_single_dims, scratch_h_data, scratch_single_dims,
        box_out_y_data, box_out_single_dims);
    // x_min = scratch_x - scratch_h / 2 (dim 1 i.e. x)
    Sub(scratch_x_data, scratch_single_dims, scratch_w_data, scratch_single_dims,
        box_out_x_data, box_out_single_dims);
    // y_max = scratch_y + scratch_h / 2 (dim 2 i.e. h)
    Add(scratch_y_data, scratch_single_dims, scratch_h_data, scratch_single_dims,
        box_out_h_data, box_out_single_dims);
    // x_max = scratch_x + scratch_h / 2 (dim 3 i.e. w)
    Add(scratch_x_data, scratch_single_dims, scratch_w_data, scratch_single_dims,
        box_out_w_data, box_out_single_dims);

    // Debug
    PRINT_VEC(box_out_y_data, 5);
    PRINT_VEC(box_out_x_data, 5);
    PRINT_VEC(box_out_h_data, 5);
    PRINT_VEC(box_out_w_data, 5);

  }  // for batch

  STOP_TIMER(PostprocessBoxes);
  PRINT_TIMER(PostprocessBoxes);

  return kTfLiteOk;
}

// Probability Postprocessing
TfLiteStatus PostprocessProbsPrepare(TfLiteContext* context, TfLiteNode* node) {
  std::cerr << std::endl << "Inside PostprocessProbPrepare" << std::endl;

  // Takes 1 inputs:
  // 0 -> concat_1 (which contains the output probabilities)
  //    with shape ? x (C+1) x A
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  // 1 output: the sigmoid of prob (after removing the background class which is
  // the first "row" in the batch)
  // with shape ? x C x A
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  // Get inputs and outputs
  TfLiteTensor* probs_in = GetInput(context, node, 0);
  TfLiteTensor* probs_out = GetOutput(context, node, 0);

  // Checks dimensions
  TF_LITE_ENSURE_EQ(context, probs_in->dims->size, 3);

  // Resize output to ? x C x A
  int num_dims = NumDimensions(probs_in);
  TfLiteIntArray* output_size = TfLiteIntArrayCreate(num_dims);
  for (int i=0; i<num_dims; ++i) {
    output_size->data[i] = probs_in->dims->data[i];
  }
  // subtract 1 from dim 1 (C + 1) to make it C
  output_size->data[1] -= 1;

  TF_LITE_ENSURE_OK(context,
          context->ResizeTensor(context, probs_out, output_size));

  PRINT_DIMS(probs_in);
  PRINT_DIMS(probs_out);

  // Allocate another tensor the same size as box_in for temp
  return kTfLiteOk;
}

TfLiteStatus PostprocessProbsEval(TfLiteContext* context, TfLiteNode* node) {
  std::cerr << std::endl << "Inside PostprocessProbsEval" << std::endl;
  START_TIMER(PostprocessProbs);

  // Get inputs and outputs
  //
  // shape B x C+1 x A
  // background class is on index 0 (of dim 1)
  TfLiteTensor* probs_in = GetInput(context, node, 0);

  // shape B x C x A
  TfLiteTensor* probs_out = GetOutput(context, node, 0);

  // Get data
  const float* probs_in_data = GetTensorData<float>(probs_in);
  float* probs_out_data = GetTensorData<float>(probs_out);

  // Get dimensions
  // Recall: dims are in reverse order
  auto probs_in_dims = GetTensorDims(probs_in);
  auto probs_out_dims = GetTensorDims(probs_out);
  PRINT_DIMS(probs_in);
  PRINT_DIMS(probs_out);


  // Number of boxes
  int A = probs_in->dims->data[2];
  // Number of images (batch)
  int B = probs_in->dims->data[0];
  // Number of classes
  int C = probs_out->dims->data[1];
  TF_LITE_ENSURE_EQ(context, C + 1, probs_in->dims->data[1]);

  // Loop on batch
  for (int b = 0; b < B; ++b) {

    // Copy all classes except background
    //
    // offset in probs_in (reverse order):
    //   a = 0, c = 1, b = b, 0
    int in_offset = Offset(probs_in_dims, 0, 1, b, 0);
    // offset in probs_out (reverse)
    //   a = 0, c = 0, b = b, 0
    int out_offset = Offset(probs_out_dims, 0, 0, b, 0);
    // how many to copy?
    int num = C * A;
    // copy
    std::copy(probs_in_data + in_offset, probs_in_data + in_offset + num,
              probs_out_data + out_offset);
  }

  // Compute sigmoid on probs_out
  // TODO(maly): Don't compute Logistic here, and instead compute it
  // at the end of NMS after all is done, and we could just compare
  // the score to the logit (inverse of logistic/sigmoid)
/*  Logistic(probs_out_data, probs_out_dims,*/
           /*probs_out_data, probs_out_dims);*/

  // print first 5 boxes in first 5 classes
  for (int c = 0; c < 5; ++c) {
    PRINT_VEC(probs_in_data + A + c * A, 5);
    PRINT_VEC(probs_out_data + c * A, 5);
  }

  STOP_TIMER(PostprocessProbs);
  PRINT_TIMER(PostprocessProbs);

  return kTfLiteOk;
}

// Non Max Suppression
//
//

// TODO(maly): Add as attributes/parameters
//
// Number of boxes to return from each image
const int kNMSMaxBoxes = 100;
// IOU overlap threshold
const float kNMSIOUThreshold = 0.6f;
// Clip window for bounding boxes in format ymin, xmin, ymax, xmax
const float kNMSClipWindow[] = {0.f, 0.f, 1.f, 1.f};
// Minimum score of boxes to return i.e. discard any box below this threshold
const float kNMSScoreThreshold = 0.20f;

TfLiteStatus NonMaxSuppressionPrepare(TfLiteContext* context,
                                      TfLiteNode* node) {
  std::cerr << std::endl << "Inside NonMaxSuppressionPrepare"
      << std::endl;

  // Takes 2 inputs:
  // 0 -> boxes with shape B x D x A (with D = 4)
  // 1 -> probs with shape B x 90 x A (with C = number of classes)
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 2);

  // 4 outputs
  // 0 -> number of detections per batch, with shape B
  // 1 -> bounding boxes with size B x D x MB (max boxes)
  // 2 -> scores for each box with shape B x 1 x MB
  // 3 -> classes for each boxe with shape B x 1 x MB
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 4);

  // Get inputs and outputs
  TfLiteTensor* boxes_in = GetInput(context, node, 0);
  TfLiteTensor* probs_in = GetInput(context, node, 1);

  TfLiteTensor* num_detections_out = GetOutput(context, node, 0);
  TfLiteTensor* boxes_out = GetOutput(context, node, 1);
  TfLiteTensor* scores_out = GetOutput(context, node, 2);
  TfLiteTensor* classes_out = GetOutput(context, node, 3);

  // Checks dimensions
  TF_LITE_ENSURE_EQ(context, boxes_in->dims->size, 3);
  TF_LITE_ENSURE_EQ(context, probs_in->dims->size, 3);
  TF_LITE_ENSURE_EQ(context, probs_in->dims->data[0],
                             boxes_in->dims->data[0]);
  TF_LITE_ENSURE_EQ(context, probs_in->dims->data[2],
                             boxes_in->dims->data[2]);

  // Get dimensions
  const int B = boxes_in->dims->data[0];
  const int D = boxes_in->dims->data[1];
  const int A = boxes_in->dims->data[2];
  const int C = probs_in->dims->data[1];
  // Number of output boxes
  // TODO(maly): read this from params/attr
  const int Ao = kNMSMaxBoxes;

  // Resize num_detections_out to shape B
  TfLiteIntArray* output_size = TfLiteIntArrayCreate(1);
  output_size->data[0] = B;
  TF_LITE_ENSURE_OK(context,
          context->ResizeTensor(context, num_detections_out, output_size));

  // Resize boxes_out
  output_size = TfLiteIntArrayCreate(3);
  output_size->data[0] = B;
  output_size->data[1] = D;
  output_size->data[2] = Ao;
  TF_LITE_ENSURE_OK(context,
          context->ResizeTensor(context, boxes_out, output_size));

  // Resize scores_out
  output_size = TfLiteIntArrayCreate(2);
  output_size->data[0] = B;
  output_size->data[1] = Ao;
  TF_LITE_ENSURE_OK(context,
          context->ResizeTensor(context, scores_out, output_size));

  // resize classes_out
  output_size = TfLiteIntArrayCreate(2);
  output_size->data[0] = B;
  output_size->data[1] = Ao;
  TF_LITE_ENSURE_OK(context,
          context->ResizeTensor(context, classes_out, output_size));

  PRINT_DIMS(probs_in);
  PRINT_DIMS(boxes_in);

  PRINT_DIMS(boxes_out);
  PRINT_DIMS(num_detections_out);
  PRINT_DIMS(scores_out);
  PRINT_DIMS(classes_out);

  return kTfLiteOk;
}

TfLiteStatus NonMaxSuppressionEval(TfLiteContext* context, TfLiteNode* node) {
  std::cerr << std::endl << "Inside NonMaxSuppressionEval"
      << std::endl;

  START_TIMER(NMS);

  // Get inputs and outputs
  TfLiteTensor* boxes_in = GetInput(context, node, 0);
  TfLiteTensor* probs_in = GetInput(context, node, 1);

  TfLiteTensor* num_detections_out = GetOutput(context, node, 0);
  TfLiteTensor* boxes_out = GetOutput(context, node, 1);
  TfLiteTensor* scores_out = GetOutput(context, node, 2);
  TfLiteTensor* classes_out = GetOutput(context, node, 3);

  // Get dimensions
  const int B = boxes_in->dims->data[0];
  const int D = boxes_in->dims->data[1];
  const int A = boxes_in->dims->data[2];
  const int C = probs_in->dims->data[1];
  // Number of output boxes
  const int Ao = boxes_out->dims->data[2];

  // Get data
  float* boxes_in_data = GetTensorData<float>(boxes_in);
  float* probs_in_data = GetTensorData<float>(probs_in);

  float* boxes_out_data = GetTensorData<float>(boxes_out);
  float* classes_out_data = GetTensorData<float>(classes_out);
  float* scores_out_data = GetTensorData<float>(scores_out);
  float* num_detections_out_data = GetTensorData<float>(num_detections_out);


  // Get dimensions
  // Recall: dims are in reverse order
  auto probs_in_dims = GetTensorDims(probs_in);
  auto boxes_in_dims = GetTensorDims(boxes_in);

  auto boxes_out_dims = GetTensorDims(boxes_out);
  auto classes_out_dims = GetTensorDims(classes_out);
  auto scores_out_dims = GetTensorDims(scores_out);
  auto num_detections_out_dims = GetTensorDims(num_detections_out);

  // For computing times
  auto t1 = std::chrono::steady_clock::now();
  auto t2 = t1;

  // Stores flags for empty boxes
  std::vector<bool> empty_boxes(A, false);

  // Score theshold to compare to: inverse of kNMSScoreThreshold
  // This is used when the probs_in are not passed through the
  // Logistic/Sigmoid function to save time.
  const float kNMSLogitThreshold = std::log(kNMSScoreThreshold /
                                            (1.f - kNMSScoreThreshold));

  // Loop on batch
  for (int b = 0; b < B; ++b) {
    // Extract single dimensions
    EXTRACT_SINGLE(boxes_in, b);
    EXTRACT_SINGLE(boxes_out, b);

    // Clip boxes
    // TODO(maly): Modify this to use scratch space for boxes_in
    // instead of overwriting the input tensors
    // TODO(maly): Look into moving these clip operations inside
    // PostprocessBoxes instead.
    //
    std::cerr << "ClipMinMax\n";
    // ymin
    PRINT_VEC(boxes_in_y_data, 5);
    ClipMinMax(boxes_in_y_data, boxes_in_single_dims,
               kNMSClipWindow[0], kNMSClipWindow[2],
               boxes_in_y_data, boxes_in_single_dims);
    PRINT_VEC(boxes_in_y_data, 5);

    // ymax
    PRINT_VEC(boxes_in_h_data, 5);
    ClipMinMax(boxes_in_h_data, boxes_in_single_dims,
               kNMSClipWindow[0], kNMSClipWindow[2],
               boxes_in_h_data, boxes_in_single_dims);
    PRINT_VEC(boxes_in_h_data, 5);
    // xmin
    PRINT_VEC(boxes_in_x_data, 5);
    ClipMinMax(boxes_in_x_data, boxes_in_single_dims,
               kNMSClipWindow[1], kNMSClipWindow[3],
               boxes_in_x_data, boxes_in_single_dims);
    PRINT_VEC(boxes_in_x_data, 5);
    // xmax
    PRINT_VEC(boxes_in_w_data, 5);
    ClipMinMax(boxes_in_w_data, boxes_in_single_dims,
               kNMSClipWindow[1], kNMSClipWindow[3],
               boxes_in_w_data, boxes_in_single_dims);
    PRINT_VEC(boxes_in_w_data, 5);

    // Mark boxes with no overlap (i.e. area = 0)
    // TODO(maly): optimize this to only process boxes being considered for
    // addition
    //
    // offset for start of batch
    int probs_offset = Offset(probs_in_dims, 0, 0, b, 0);
    t1 = std::chrono::steady_clock::now();

    // Init to false
    std::fill(empty_boxes.begin(), empty_boxes.end(), false);
    for (int a = 0; a < A; ++a) {
      // compute area
      float area = (boxes_in_h_data[a] - boxes_in_y_data[a]) *
                   (boxes_in_w_data[a] - boxes_in_x_data[a]);

      // Zero out its scores
      if (std::fabs(area) < 1e-5) {
          empty_boxes[a] = true;
      }
    }
    t2 = std::chrono::steady_clock::now();
    PRINT_TIME2("Area", t1, t2);

    // These should be filled in incrementally by the loop below
    // TODO(maly): optimize this and later appending
    // Stores scores for boxes for each class
    std::vector<float> scores;
    // Classes
    std::vector<int> classes;
    // indices of selected boxes
    std::vector<int> indices;

    double sort_time = 0.;

    // Loop over the classes and do NMS for each class independently
    // Code mostly copied from tensorflow/core/kernels/non_max_supression_op.cc
    for (int c = 0; c < C; ++c) {
      // pointer to current class probabilities
      const float* probs_in_c_data = probs_in_data + probs_offset + c * A;

      // Get scores of boxes that are only above score threshold and
      // that have overlap with clip window, to reduce the sort time
      // TODO(maly): remove outside of the loop
      std::vector<float> presort_probs(A);
      std::vector<int> presort_indices(A);
      int presort_num = 0;
      // loop on boxes
      for (int a = 0; a < A; ++a) {
        // If not empty and has good score
        // if (!empty_boxes[a] && probs_in_c_data[a] > kNMSScoreThreshold) {
        if (!empty_boxes[a] && probs_in_c_data[a] > kNMSLogitThreshold) {
          // save its prob and index
          presort_probs[presort_num] = probs_in_c_data[a];
          presort_indices[presort_num] = a;
          // increment counter
          presort_num++;
        }
      }
      // Now we have only non-empty boxes with score above threshold

      // Copy scores for this class
      // std::copy_n(probs_in_data + probs_offset, A, scores.begin());

      // Sort and get indices in descending order of score (prob)
      // TODO(maly): Remove out of the loop and preallocate
      std::vector<int> sorted_indices(A);
      // Pass in probs of current class
      t1 = std::chrono::steady_clock::now();
      //DecreasingArgSort(probs_in_c_data, A, &sorted_indices);
      DecreasingArgSort(presort_probs.data(), presort_num, &sorted_indices);
      t2 = std::chrono::steady_clock::now();
      sort_time += GET_TIME(t1, t2);

      // Add Ao boxes greedily by discarding those overlapping with boxes
      // that have higher scores (that were added previously)
      std::vector<int> selected;
      //std::vector<int> selected_indices(Ao, 0);
      //int num_selected = 0;
      //for (int i = 0; i < A; ++i) {
      for (int i = 0; i < sorted_indices.size(); ++i) {
        // Stop processing?
        if (selected.size() >= Ao) break;

        // index of box to add
        //int index = sorted_indices[i];
        int sorted_index = sorted_indices[i];
        int index = presort_indices[sorted_index];

        // Stop processing if score below threshold
        // if (probs_in_c_data[index] < kNMSScoreThreshold) break;

        // Empty box? skip
        // if (empty_boxes[index]) continue;

        bool should_select = true;
        // Overlapping boxes are likely to have similar scores,
        // therefore we iterate through the selected boxes backwards.
        //for (int j = num_selected - 1; j >= 0; --j) {
        for (int j = selected.size() - 1; j >= 0; --j) {
          if (IOUGreaterThanThreshold(
                      boxes_in_y_data, boxes_in_x_data,
                      boxes_in_h_data, boxes_in_w_data,
                      index,
                      //sorted_indices[selected_indices[j]],
                      //presort_indices[sorted_indices[selected_indices[j]]],
                      selected[j],
                      kNMSIOUThreshold)) {
            should_select = false;
            break;
          }
        }

        if (should_select) {
          selected.push_back(index);
          //selected_indices[num_selected++] = i;
        }
      }

      // Now selected has the set of indices of boxes to add
      // TODO(maly): optimize this
      // Add indices
      indices.insert(indices.end(), selected.begin(), selected.end());
      // Add the scores & classes
      for (auto index: selected) {
        // add score (prob)
        scores.push_back(probs_in_c_data[index]);
        // add class
        classes.push_back(c + 1);
      }
    }  // for c

    // Now we have a list of boxes with scores, we need to sort them again
    // and clip the top kNMSMaxBoxes

    // Sort all boxes again based on scores
    std::vector<int> sorted_indices;
    DecreasingArgSort(scores, &sorted_indices);

    // Number of boxes to return for this batch
    int num_detections = std::min(Ao, static_cast<int>(sorted_indices.size()));

    // Put num_detections (it's a vector, one entry for each batch)
    num_detections_out_data[b] = num_detections;

    // Get offsets for current batch: skip Ao boxes
    int scores_offset = b * Ao;
    int classes_offset = b * Ao;

    // Loop on boxes and copy output
    for (int a = 0; a < num_detections; ++a) {
      // sorted index
      const int sorted_index = sorted_indices[a];

      // index of the current box
      const int box_index = indices[sorted_index];

      // Copy box
      boxes_out_y_data[a] = boxes_in_y_data[box_index];
      boxes_out_x_data[a] = boxes_in_x_data[box_index];
      boxes_out_h_data[a] = boxes_in_h_data[box_index];
      boxes_out_w_data[a] = boxes_in_w_data[box_index];

      // Copy score and class
      classes_out_data[classes_offset + a] = classes[sorted_index];
      scores_out_data[scores_offset + a] = scores[sorted_index];
    }

    PRINT_TIME1("Sorting", sort_time);
    //std::cerr << "Sorting took: " << sort_time << " msec.\n";

    PRINT_VEC(boxes_out_y_data, 5);
    PRINT_VEC(boxes_out_x_data, 5);
    PRINT_VEC(boxes_out_h_data, 5);
    PRINT_VEC(boxes_out_w_data, 5);

    PRINT_VEC(num_detections_out_data, 1);
    PRINT_VEC(classes_out_data + classes_offset, 5);
    PRINT_VEC(scores_out_data + scores_offset, 5);
  } // for b

  // Compute Lopgistic on scores_out
  Logistic(scores_out_data, scores_out_dims,
           scores_out_data, scores_out_dims);
  PRINT_VEC(scores_out_data, 5);

  STOP_TIMER(NMS);
  PRINT_TIMER(NMS);

  return kTfLiteOk;
}

} // namespace ssd


TfLiteRegistration* Register_SSDPostprocessBoxes() {
  static TfLiteRegistration r = {ssd::Init, ssd::Free,
                                 ssd::PostprocessBoxesPrepare,
                                 ssd::PostprocessBoxesEval};
  return &r;
}

TfLiteRegistration* Register_SSDPostprocessProbs() {
  static TfLiteRegistration r = {nullptr, nullptr,
                                 ssd::PostprocessProbsPrepare,
                                 ssd::PostprocessProbsEval};
  return &r;
}

TfLiteRegistration* Register_SSDNonMaxSuppression() {
  static TfLiteRegistration r = {nullptr, nullptr,
                                 ssd::NonMaxSuppressionPrepare,
                                 ssd::NonMaxSuppressionEval};
  return &r;
}

} // namespace custom
} // namespace ops
} // namespace tflite

