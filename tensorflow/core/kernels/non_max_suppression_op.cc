/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

// See docs in ../ops/image_ops.cc

#define EIGEN_USE_THREADS

#include "tensorflow/core/kernels/non_max_suppression_op.h"

#include <functional>
#include <queue>
#include <vector>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/stl_util.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
namespace {

typedef Eigen::ThreadPoolDevice CPUDevice;

static inline void CheckScoreSizes(OpKernelContext* context, int num_boxes,
                                   const Tensor& scores) {
  // The shape of 'scores' is [num_boxes]
  OP_REQUIRES(context, scores.dims() == 1,
              errors::InvalidArgument("scores must be 1-D",
                                      scores.shape().DebugString()));
  OP_REQUIRES(context, scores.dim_size(0) == num_boxes,
              errors::InvalidArgument("scores has incompatible shape"));
}

static inline void ParseAndCheckOverlapSizes(OpKernelContext* context,
                                             const Tensor& overlaps,
                                             int* num_boxes) {
  // the shape of 'overlaps' is [num_boxes, num_boxes]
  OP_REQUIRES(context, overlaps.dims() == 2,
              errors::InvalidArgument("overlaps must be 2-D",
                                      overlaps.shape().DebugString()));

  *num_boxes = overlaps.dim_size(0);
  OP_REQUIRES(context, overlaps.dim_size(1) == *num_boxes,
              errors::InvalidArgument("overlaps must be square",
                                      overlaps.shape().DebugString()));
}

static inline void ParseAndCheckBoxSizes(OpKernelContext* context,
                                         const Tensor& boxes, int* num_boxes) {
  // The shape of 'boxes' is [num_boxes, 4]
  OP_REQUIRES(context, boxes.dims() == 2,
              errors::InvalidArgument("boxes must be 2-D",
                                      boxes.shape().DebugString()));
  *num_boxes = boxes.dim_size(0);
  OP_REQUIRES(context, boxes.dim_size(1) == 4,
              errors::InvalidArgument("boxes must have 4 columns"));
}

static inline void CheckNMSLiteScoreSizes(OpKernelContext* context, 
                                          int num_boxes,
                                          const Tensor& scores) {
  // The shape of 'scores' is [batch_size, num_boxes, num_classes]
  OP_REQUIRES(context, scores.dims() == 3,
              errors::InvalidArgument("scores must be 3-D",
                                      scores.shape().DebugString()));
  OP_REQUIRES(context, scores.dim_size(1) == num_boxes,
              errors::InvalidArgument("scores has incompatible shape"));
}

static inline void ParseAndCheckNMSLiteBoxSizes(OpKernelContext* context,
                                         const Tensor& boxes, int* num_boxes, 
                                         const int num_classes) {
  // The shape of 'boxes' is [batch_size, num_boxes, q, 4]
  OP_REQUIRES(context, boxes.dims() == 4,
              errors::InvalidArgument("boxes must be 4-D",
                                      boxes.shape().DebugString()));

  bool box_check = boxes.dim_size(2) == 1 || boxes.dim_size(2) == num_classes;
  OP_REQUIRES(context, box_check,
              errors::InvalidArgument(
                  "third dimension of boxes must be either 1 or num classes"));
  // Num_boxes * q where q is either 1 or num_classes
  // TODO check if this should take num_classes into account
  *num_boxes = boxes.dim_size(1); //*boxes.dim_size(2);
  OP_REQUIRES(context, boxes.dim_size(3) == 4,
              errors::InvalidArgument("boxes must have 4 columns"));
}
// Return intersection-over-union overlap between boxes i and j
template <typename T>
static inline bool IOUGreaterThanThreshold(
    typename TTypes<T, 2>::ConstTensor boxes, int i, int j, T iou_threshold) {
  const T ymin_i = std::min<T>(boxes(i, 0), boxes(i, 2));
  const T xmin_i = std::min<T>(boxes(i, 1), boxes(i, 3));
  const T ymax_i = std::max<T>(boxes(i, 0), boxes(i, 2));
  const T xmax_i = std::max<T>(boxes(i, 1), boxes(i, 3));
  const T ymin_j = std::min<T>(boxes(j, 0), boxes(j, 2));
  const T xmin_j = std::min<T>(boxes(j, 1), boxes(j, 3));
  const T ymax_j = std::max<T>(boxes(j, 0), boxes(j, 2));
  const T xmax_j = std::max<T>(boxes(j, 1), boxes(j, 3));
  const T area_i = (ymax_i - ymin_i) * (xmax_i - xmin_i);
  const T area_j = (ymax_j - ymin_j) * (xmax_j - xmin_j);
  if (area_i <= static_cast<T>(0) || area_j <= static_cast<T>(0)) return 0;
  const T intersection_ymin = std::max<T>(ymin_i, ymin_j);
  const T intersection_xmin = std::max<T>(xmin_i, xmin_j);
  const T intersection_ymax = std::min<T>(ymax_i, ymax_j);
  const T intersection_xmax = std::min<T>(xmax_i, xmax_j);
  const T intersection_area =
      std::max<T>(intersection_ymax - intersection_ymin, static_cast<T>(0.0)) *
      std::max<T>(intersection_xmax - intersection_xmin, static_cast<T>(0.0));
  const T iou = intersection_area / (area_i + area_j - intersection_area);
  return iou > iou_threshold;
}

static inline bool OverlapsGreaterThanThreshold(
    typename TTypes<float, 2>::ConstTensor overlaps, int i, int j,
    float overlap_threshold) {
  return overlaps(i, j) > overlap_threshold;
}

template <typename T>
static inline std::function<bool(int, int)> CreateIOUSuppressCheckFn(
    const Tensor& boxes, float threshold) {
  typename TTypes<T, 2>::ConstTensor boxes_data = boxes.tensor<T, 2>();
  return std::bind(&IOUGreaterThanThreshold<T>, boxes_data,
                   std::placeholders::_1, std::placeholders::_2,
                   static_cast<T>(threshold));
}

static inline std::function<bool(int, int)> CreateOverlapsSuppressCheckFn(
    const Tensor& overlaps, float threshold) {
  typename TTypes<float, 2>::ConstTensor overlaps_data =
      overlaps.tensor<float, 2>();
  return std::bind(&OverlapsGreaterThanThreshold, overlaps_data,
                   std::placeholders::_1, std::placeholders::_2, threshold);
}

template <typename T>
void DoNonMaxSuppressionOp(
    OpKernelContext* context, const Tensor& scores, int num_boxes,
    const Tensor& max_output_size, const float score_threshold,
    const std::function<bool(int, int)>& suppress_check_fn,
    bool pad_to_max_output_size = false, int* ptr_num_valid_outputs = nullptr) {
  const int output_size = max_output_size.scalar<int>()();

  std::vector<T> scores_data(num_boxes);
  std::copy_n(scores.flat<T>().data(), num_boxes, scores_data.begin());

  // Data structure for selection candidate in NMS.
  struct Candidate {
    int box_index;
    T score;
  };

  auto cmp = [](const Candidate bs_i, const Candidate bs_j) {
    return bs_i.score < bs_j.score;
  };
  std::priority_queue<Candidate, std::deque<Candidate>, decltype(cmp)>
      candidate_priority_queue(cmp);
  for (int i = 0; i < scores_data.size(); ++i) {
    if (static_cast<float>(scores_data[i]) > score_threshold) {
      candidate_priority_queue.emplace(Candidate({i, scores_data[i]}));
    }
  }

  std::vector<int> selected;
  std::vector<T> selected_scores;
  Candidate next_candidate;

  while (selected.size() < output_size && !candidate_priority_queue.empty()) {
    next_candidate = candidate_priority_queue.top();
    candidate_priority_queue.pop();

    // Overlapping boxes are likely to have similar scores,
    // therefore we iterate through the previously selected boxes backwards
    // in order to see if `next_candidate` should be suppressed.
    bool should_select = true;

    for (int j = static_cast<int>(selected.size()) - 1; j >= 0; --j) {
      if (suppress_check_fn(next_candidate.box_index, selected[j])) {
        should_select = false;
        break;
      }
    }

    if (should_select) {
      selected.push_back(next_candidate.box_index);
      selected_scores.push_back(next_candidate.score);
    }
  }

  int num_valid_outputs = selected.size();
  if (pad_to_max_output_size) {
    selected.resize(output_size, 0);
    selected_scores.resize(output_size, static_cast<T>(0));
  }
  if (ptr_num_valid_outputs) {
    *ptr_num_valid_outputs = num_valid_outputs;
  }

  // Allocate output tensors
  Tensor* output_indices = nullptr;
  TensorShape output_shape({static_cast<int>(selected.size())});
  OP_REQUIRES_OK(context,
                 context->allocate_output(0, output_shape, &output_indices));
  TTypes<int, 1>::Tensor output_indices_data = output_indices->tensor<int, 1>();
  std::copy_n(selected.begin(), selected.size(), output_indices_data.data());
}

void BatchedNonMaxSuppressionOp(OpKernelContext* context, 
                           const Tensor& inp_boxes, const Tensor& inp_scores, 
                           int num_boxes, 
                           const Tensor& max_size_per_class, 
                           const Tensor& max_total_size, 
                           const float score_threshold, 
                           const float iou_threshold) {

  int q = inp_boxes.dim_size(2);
  int num_classes = inp_scores.dim_size(2);
  //unpack along batch dimension
  const int num_batches = inp_boxes.dim_size(0);

  // [num_batches, max_detections, 4]
  std::vector <std::vector<float>> nmsed_boxes(num_batches);
  // [num_batches, max_detections]
  std::vector <std::vector<float>> nmsed_scores(num_batches); 
  // [num_batches, max_detections]
  std::vector <std::vector<float>> nmsed_classes(num_batches); 
  // [num_batches]
  std::vector <int> final_valid_detections;
  // [num_batches, max_detections]
  std::vector <std::vector<int>> selected_indices(num_batches);

  int max_detections = 0;
  //perform non_max_suppression operation for each batch independently
  for (int batch = 0; batch < num_batches; ++batch) {
    std::cout << " iteration " << batch <<std::endl;
    // dims of per_batch_boxes [num_boxes, q, 4]
    //Tensor per_batch_boxes = inp_boxes.SubSlice(batch);
    Tensor per_batch_boxes = inp_boxes.Slice(batch, batch+1);
    // dims of per_batch_scores [num_boxes, num_classes]
    Tensor per_batch_scores = inp_scores.Slice(batch, batch+1);

    struct ResultCandidate {
          int box_index;
          float score;
          int class_idx;
          float box_coord[4];
    };

    auto rc_cmp = [](const ResultCandidate rc_i, const ResultCandidate rc_j) {
        return rc_i.score < rc_j.score;
    };
    std::priority_queue<ResultCandidate, std::vector<ResultCandidate>, 
      decltype(rc_cmp)> result_candidate_pq(rc_cmp);

    float * scoresData = per_batch_scores.unaligned_flat<float>().data();
    float * boxesData = per_batch_boxes.unaligned_flat<float>().data();

    //Iterate through all classes
    for (int class_idx = 0; class_idx < num_classes; ++class_idx) {
      std::vector<float> scores_data;
      std::vector<float> boxes_data_vec;

      for (int box = 0; box < num_boxes; ++box) {
        //Get the scores per class 
        //scores_data dim is [num_boxes].
        scores_data.push_back(scoresData[box * num_classes + class_idx]);
        for(int cid = 0; cid < 4; ++cid){
          if(q > 1){
            //Get the boxes per class. boxes_data_vec dims is [num_boxes, 4]
            boxes_data_vec.push_back(boxesData[box * q * 4 + 
                  class_idx * 4 + cid]);
            //std::cout << " Pushing box_data " << boxes_data_vec[boxes_data_vec.size()-1] << std::endl;
          }
          else 
              boxes_data_vec.push_back(boxesData[box * 4 + cid]);
        }
      }
        
      //Copy boxes_data_vec to a tensor
      TensorShape boxesShape({num_boxes, 4});
      Tensor boxes(per_batch_boxes.dtype(), boxesShape);
      std::copy_n(boxes_data_vec.begin(), boxes_data_vec.size(), 
          boxes.unaligned_flat<float>().data());

      const int output_size = std::min(max_size_per_class.scalar<int>()(), 
                                   num_boxes);
      // Do NMS, get the candidate indices of form vector<int>
      // Data structure for selection candidate in NMS.
      struct Candidate {
        int box_index;
        float score;
      };
      auto cmp = [](const Candidate bs_i, const Candidate bs_j) {
        return bs_i.score < bs_j.score;
      };
      std::priority_queue<Candidate, std::deque<Candidate>, decltype(cmp)>
        candidate_priority_queue(cmp);
      for (int i = 0; i < scores_data.size(); ++i) {
          //std::cout << "scores_data "<< scores_data[i] << std::endl;
        if (scores_data[i] > score_threshold) {
          candidate_priority_queue.emplace(Candidate({i, scores_data[i]}));
        }
      }

      std::vector<int> selected;
      std::vector<float> selected_boxes;
      Candidate next_candidate;

      const Tensor const_boxes = boxes;
      typename TTypes<float, 2>::ConstTensor boxes_data = 
        const_boxes.tensor<float, 2>();
      LOG(ERROR) << " boxes data " << boxes.DebugString();
      while (selected.size() < output_size && !candidate_priority_queue.empty())
      {
        next_candidate = candidate_priority_queue.top();
        //std::cout << " next_candidate " << next_candidate.box_index << std::endl;
        candidate_priority_queue.pop();

        // Overlapping boxes are likely to have similar scores,
        // therefore we iterate through the previously selected boxes backwards
        // in order to see if `next_candidate` should be suppressed.
        bool should_select = true;
        for (int j = selected.size() - 1; j >= 0; --j) {
          if (IOUGreaterThanThreshold(boxes_data, next_candidate.box_index, 
              selected[j], iou_threshold)) {
              //std::cout << " Dont select. box id is " << selected[j] << " idx " << j <<std::endl;  
            should_select = false;
            break;
          }
        }

        if (should_select) {
          selected.push_back(next_candidate.box_index);
          //std::cout << " select box_index " << next_candidate.box_index << std::endl;
          //Add the selected box to the result candidate. Sorted by score
          int id = next_candidate.box_index;
          ResultCandidate rc = {next_candidate.box_index, next_candidate.score,
            class_idx, {boxes_data(id, 0), boxes_data(id, 1), boxes_data(id, 2),
            boxes_data(id, 3)}}; 
          result_candidate_pq.push(rc);
        }
      }

    }
    int total_size_per_batch = max_total_size.scalar<int>()();
    std::cout << " pq size " << result_candidate_pq.size() << std::endl;

    if (total_size_per_batch > 0)
      max_detections = std::min((int) result_candidate_pq.size(), 
        total_size_per_batch);
    else max_detections = (int) result_candidate_pq.size();

    std::cout << " max_detections " << max_detections << std::endl;
    final_valid_detections.push_back(max_detections);
    // Pick the top max_total_size values 
    while(total_size_per_batch > 0 && !result_candidate_pq.empty())
    {
      ResultCandidate next_candidate = result_candidate_pq.top();
      result_candidate_pq.pop();
      // Add to final output vectors
      nmsed_boxes[batch].push_back(next_candidate.box_coord[0]);
      nmsed_boxes[batch].push_back(next_candidate.box_coord[1]);
      nmsed_boxes[batch].push_back(next_candidate.box_coord[2]);
      nmsed_boxes[batch].push_back(next_candidate.box_coord[3]);
      nmsed_scores[batch].push_back(next_candidate.score);
      nmsed_classes[batch].push_back(next_candidate.class_idx);
      selected_indices[batch].push_back(next_candidate.box_index);
      total_size_per_batch--;
    }
            
  }

  Tensor* nmsed_boxes_t = nullptr;
  TensorShape boxes_shape({num_batches, max_detections, 4});
  OP_REQUIRES_OK(context,
                 context->allocate_output(0, boxes_shape, &nmsed_boxes_t));
  auto nmsed_boxes_flat = nmsed_boxes_t->template flat<float>();

  Tensor* nmsed_scores_t = nullptr;
  TensorShape scores_shape({num_batches, max_detections});
  OP_REQUIRES_OK(context,
                 context->allocate_output(1, scores_shape, &nmsed_scores_t));
  auto nmsed_scores_flat = nmsed_scores_t->template flat<float>();

  Tensor* nmsed_classes_t = nullptr;
  OP_REQUIRES_OK(context,
                 context->allocate_output(2, scores_shape, &nmsed_classes_t));
  auto nmsed_classes_flat = nmsed_classes_t->template flat<float>();

  Tensor* valid_detections_t = nullptr;
  TensorShape valid_detections_shape({num_batches});
  OP_REQUIRES_OK(context, context->allocate_output(3, valid_detections_shape, 
                 &valid_detections_t));
  auto valid_detections_flat = valid_detections_t->template flat<int>();

  Tensor* selected_indices_t = nullptr;
  OP_REQUIRES_OK(context,
                 context->allocate_output(4, scores_shape, &selected_indices_t));
  auto selected_indices_flat = selected_indices_t->template flat<int>();
  for (int i = 0; i < num_batches; ++i) {
    valid_detections_flat(i) = final_valid_detections[i];
    for (int j = 0; j < max_detections; ++j) {
      nmsed_scores_flat(i * max_detections + j) = nmsed_scores[i][j];
      nmsed_classes_flat(i * max_detections + j) = nmsed_classes[i][j];
      selected_indices_flat(i * max_detections + j) = selected_indices[i][j];
      for (int k = 0; k < 4; ++k) {
        nmsed_boxes_flat(i * max_detections * 4 + j * 4 + k) = 
          nmsed_boxes[i][j * 4 + k]; 
      }
    }
  }
  LOG(ERROR) << " boxes " << nmsed_boxes_t->DebugString();
  LOG(ERROR) << " Scores " << nmsed_scores_t->DebugString();
  LOG(ERROR) << " classes " << nmsed_classes_t->DebugString();
}

}  // namespace

template <typename Device>
class NonMaxSuppressionOp : public OpKernel {
 public:
  explicit NonMaxSuppressionOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("iou_threshold", &iou_threshold_));
  }

  void Compute(OpKernelContext* context) override {
    // boxes: [num_boxes, 4]
    const Tensor& boxes = context->input(0);
    // scores: [num_boxes]
    const Tensor& scores = context->input(1);
    // max_output_size: scalar
    const Tensor& max_output_size = context->input(2);
    OP_REQUIRES(
        context, TensorShapeUtils::IsScalar(max_output_size.shape()),
        errors::InvalidArgument("max_output_size must be 0-D, got shape ",
                                max_output_size.shape().DebugString()));

    OP_REQUIRES(context, iou_threshold_ >= 0 && iou_threshold_ <= 1,
                errors::InvalidArgument("iou_threshold must be in [0, 1]"));
    int num_boxes = 0;
    ParseAndCheckBoxSizes(context, boxes, &num_boxes);
    CheckScoreSizes(context, num_boxes, scores);
    if (!context->status().ok()) {
      return;
    }
    auto suppress_check_fn =
        CreateIOUSuppressCheckFn<float>(boxes, iou_threshold_);

    const float score_threshold_val = std::numeric_limits<float>::lowest();
    DoNonMaxSuppressionOp<float>(context, scores, num_boxes, max_output_size,
                                 score_threshold_val, suppress_check_fn);
  }

 private:
  float iou_threshold_;
};

template <typename Device, typename T>
class NonMaxSuppressionV2Op : public OpKernel {
 public:
  explicit NonMaxSuppressionV2Op(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // boxes: [num_boxes, 4]
    const Tensor& boxes = context->input(0);
    // scores: [num_boxes]
    const Tensor& scores = context->input(1);
    // max_output_size: scalar
    const Tensor& max_output_size = context->input(2);
    OP_REQUIRES(
        context, TensorShapeUtils::IsScalar(max_output_size.shape()),
        errors::InvalidArgument("max_output_size must be 0-D, got shape ",
                                max_output_size.shape().DebugString()));
    // iou_threshold: scalar
    const Tensor& iou_threshold = context->input(3);
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(iou_threshold.shape()),
                errors::InvalidArgument("iou_threshold must be 0-D, got shape ",
                                        iou_threshold.shape().DebugString()));
    const float iou_threshold_val = iou_threshold.scalar<float>()();

    OP_REQUIRES(context, iou_threshold_val >= 0 && iou_threshold_val <= 1,
                errors::InvalidArgument("iou_threshold must be in [0, 1]"));
    int num_boxes = 0;
    ParseAndCheckBoxSizes(context, boxes, &num_boxes);
    CheckScoreSizes(context, num_boxes, scores);
    if (!context->status().ok()) {
      return;
    }
    auto suppress_check_fn =
        CreateIOUSuppressCheckFn<T>(boxes, iou_threshold_val);

    const float score_threshold_val = std::numeric_limits<float>::lowest();
    DoNonMaxSuppressionOp<T>(context, scores, num_boxes, max_output_size,
                             score_threshold_val, suppress_check_fn);
  }
};

template <typename Device, typename T>
class NonMaxSuppressionV3Op : public OpKernel {
 public:
  explicit NonMaxSuppressionV3Op(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // boxes: [num_boxes, 4]
    const Tensor& boxes = context->input(0);
    // scores: [num_boxes]
    const Tensor& scores = context->input(1);
    // max_output_size: scalar
    const Tensor& max_output_size = context->input(2);
    OP_REQUIRES(
        context, TensorShapeUtils::IsScalar(max_output_size.shape()),
        errors::InvalidArgument("max_output_size must be 0-D, got shape ",
                                max_output_size.shape().DebugString()));
    // iou_threshold: scalar
    const Tensor& iou_threshold = context->input(3);
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(iou_threshold.shape()),
                errors::InvalidArgument("iou_threshold must be 0-D, got shape ",
                                        iou_threshold.shape().DebugString()));
    const float iou_threshold_val = iou_threshold.scalar<float>()();
    OP_REQUIRES(context, iou_threshold_val >= 0 && iou_threshold_val <= 1,
                errors::InvalidArgument("iou_threshold must be in [0, 1]"));
    // score_threshold: scalar
    const Tensor& score_threshold = context->input(4);
    OP_REQUIRES(
        context, TensorShapeUtils::IsScalar(score_threshold.shape()),
        errors::InvalidArgument("score_threshold must be 0-D, got shape ",
                                score_threshold.shape().DebugString()));
    const float score_threshold_val = score_threshold.scalar<float>()();

    int num_boxes = 0;
    ParseAndCheckBoxSizes(context, boxes, &num_boxes);
    CheckScoreSizes(context, num_boxes, scores);
    if (!context->status().ok()) {
      return;
    }

    auto suppress_check_fn =
        CreateIOUSuppressCheckFn<T>(boxes, iou_threshold_val);

    DoNonMaxSuppressionOp<T>(context, scores, num_boxes, max_output_size,
                             score_threshold_val, suppress_check_fn);
  }
};

template <typename Device, typename T>
class NonMaxSuppressionV4Op : public OpKernel {
 public:
  explicit NonMaxSuppressionV4Op(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("pad_to_max_output_size",
                                             &pad_to_max_output_size_));
  }

  void Compute(OpKernelContext* context) override {
    // boxes: [num_boxes, 4]
    const Tensor& boxes = context->input(0);
    // scores: [num_boxes]
    const Tensor& scores = context->input(1);
    // max_output_size: scalar
    const Tensor& max_output_size = context->input(2);
    OP_REQUIRES(
        context, TensorShapeUtils::IsScalar(max_output_size.shape()),
        errors::InvalidArgument("max_output_size must be 0-D, got shape ",
                                max_output_size.shape().DebugString()));
    // iou_threshold: scalar
    const Tensor& iou_threshold = context->input(3);
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(iou_threshold.shape()),
                errors::InvalidArgument("iou_threshold must be 0-D, got shape ",
                                        iou_threshold.shape().DebugString()));
    const float iou_threshold_val = iou_threshold.scalar<float>()();
    OP_REQUIRES(context, iou_threshold_val >= 0 && iou_threshold_val <= 1,
                errors::InvalidArgument("iou_threshold must be in [0, 1]"));
    // score_threshold: scalar
    const Tensor& score_threshold = context->input(4);
    OP_REQUIRES(
        context, TensorShapeUtils::IsScalar(score_threshold.shape()),
        errors::InvalidArgument("score_threshold must be 0-D, got shape ",
                                score_threshold.shape().DebugString()));
    const float score_threshold_val = score_threshold.scalar<float>()();

    int num_boxes = 0;
    ParseAndCheckBoxSizes(context, boxes, &num_boxes);
    CheckScoreSizes(context, num_boxes, scores);
    if (!context->status().ok()) {
      return;
    }

    auto suppress_check_fn =
        CreateIOUSuppressCheckFn<T>(boxes, iou_threshold_val);
    int num_valid_outputs;

    DoNonMaxSuppressionOp<T>(context, scores, num_boxes, max_output_size,
                             score_threshold_val, suppress_check_fn,
                             pad_to_max_output_size_, &num_valid_outputs);

    // Allocate scalar output tensor for number of indices computed.
    Tensor* num_outputs_t = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(
                                1, tensorflow::TensorShape{}, &num_outputs_t));
    num_outputs_t->scalar<int32>().setConstant(num_valid_outputs);
  }

 private:
  bool pad_to_max_output_size_;
};

template <typename Device>
class NonMaxSuppressionWithOverlapsOp : public OpKernel {
 public:
  explicit NonMaxSuppressionWithOverlapsOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // overlaps: [num_boxes, num_boxes]
    const Tensor& overlaps = context->input(0);
    // scores: [num_boxes]
    const Tensor& scores = context->input(1);
    // max_output_size: scalar
    const Tensor& max_output_size = context->input(2);
    OP_REQUIRES(
        context, TensorShapeUtils::IsScalar(max_output_size.shape()),
        errors::InvalidArgument("max_output_size must be 0-D, got shape ",
                                max_output_size.shape().DebugString()));
    // overlap_threshold: scalar
    const Tensor& overlap_threshold = context->input(3);
    OP_REQUIRES(
        context, TensorShapeUtils::IsScalar(overlap_threshold.shape()),
        errors::InvalidArgument("overlap_threshold must be 0-D, got shape ",
                                overlap_threshold.shape().DebugString()));
    const float overlap_threshold_val = overlap_threshold.scalar<float>()();

    // score_threshold: scalar
    const Tensor& score_threshold = context->input(4);
    OP_REQUIRES(
        context, TensorShapeUtils::IsScalar(score_threshold.shape()),
        errors::InvalidArgument("score_threshold must be 0-D, got shape ",
                                score_threshold.shape().DebugString()));
    const float score_threshold_val = score_threshold.scalar<float>()();

    int num_boxes = 0;
    ParseAndCheckOverlapSizes(context, overlaps, &num_boxes);
    CheckScoreSizes(context, num_boxes, scores);
    if (!context->status().ok()) {
      return;
    }
    auto suppress_check_fn =
        CreateOverlapsSuppressCheckFn(overlaps, overlap_threshold_val);

    DoNonMaxSuppressionOp<float>(context, scores, num_boxes, max_output_size,
                                 score_threshold_val, suppress_check_fn);
  }
};

template <typename Device>
class NonMaxSuppressionLiteOp : public OpKernel {
 public:
  explicit NonMaxSuppressionLiteOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // boxes: [batch_size, num_anchors, q, 4]
    const Tensor& boxes = context->input(0);
    LOG(ERROR) << " inp_boxes " << boxes.DebugString();
    // scores: [batch_size, num_anchors, num_classes]
    const Tensor& scores = context->input(1);
    OP_REQUIRES(context, (boxes.dim_size(0) == scores.dim_size(0)),
              errors::InvalidArgument(
                "boxes and scores must have same batch size"));

    // max_output_size: scalar
    const Tensor& max_output_size = context->input(2);
    OP_REQUIRES(
        context, TensorShapeUtils::IsScalar(max_output_size.shape()),
        errors::InvalidArgument("max_size_per_class must be 0-D, got shape ",
                                max_output_size.shape().DebugString()));
    // max_total_size: scalar
    const Tensor& max_total_size = context->input(3);
    OP_REQUIRES(
        context, TensorShapeUtils::IsScalar(max_total_size.shape()),
        errors::InvalidArgument("max_total_size must be 0-D, got shape ",
                                max_total_size.shape().DebugString()));
    // iou_threshold: scalar
    const Tensor& iou_threshold = context->input(4);
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(iou_threshold.shape()),
                errors::InvalidArgument("iou_threshold must be 0-D, got shape ",
                                        iou_threshold.shape().DebugString()));
    const float iou_threshold_val = iou_threshold.scalar<float>()();

    // score_threshold: scalar
    const Tensor& score_threshold = context->input(5);
    OP_REQUIRES(
        context, TensorShapeUtils::IsScalar(score_threshold.shape()),
        errors::InvalidArgument("score_threshold must be 0-D, got shape ",
                                score_threshold.shape().DebugString()));
    const float score_threshold_val = score_threshold.scalar<float>()();

    OP_REQUIRES(context, iou_threshold_val >= 0 && iou_threshold_val <= 1,
                errors::InvalidArgument("iou_threshold must be in [0, 1]"));
    int num_boxes = 0;
    const int num_classes = scores.dim_size(2);
    ParseAndCheckNMSLiteBoxSizes(context, boxes, &num_boxes, num_classes);
    CheckNMSLiteScoreSizes(context, num_boxes, scores);
    if (!context->status().ok()) {
      return;
    }
    BatchedNonMaxSuppressionOp(context, boxes, scores, num_boxes, 
                               max_output_size, max_total_size, 
                               score_threshold_val, iou_threshold_val);
  }
};


REGISTER_KERNEL_BUILDER(Name("NonMaxSuppression").Device(DEVICE_CPU),
                        NonMaxSuppressionOp<CPUDevice>);

REGISTER_KERNEL_BUILDER(
    Name("NonMaxSuppressionV2").TypeConstraint<float>("T").Device(DEVICE_CPU),
    NonMaxSuppressionV2Op<CPUDevice, float>);
REGISTER_KERNEL_BUILDER(Name("NonMaxSuppressionV2")
                            .TypeConstraint<Eigen::half>("T")
                            .Device(DEVICE_CPU),
                        NonMaxSuppressionV2Op<CPUDevice, Eigen::half>);

REGISTER_KERNEL_BUILDER(
    Name("NonMaxSuppressionV3").TypeConstraint<float>("T").Device(DEVICE_CPU),
    NonMaxSuppressionV3Op<CPUDevice, float>);
REGISTER_KERNEL_BUILDER(Name("NonMaxSuppressionV3")
                            .TypeConstraint<Eigen::half>("T")
                            .Device(DEVICE_CPU),
                        NonMaxSuppressionV3Op<CPUDevice, Eigen::half>);

REGISTER_KERNEL_BUILDER(
    Name("NonMaxSuppressionV4").TypeConstraint<float>("T").Device(DEVICE_CPU),
    NonMaxSuppressionV4Op<CPUDevice, float>);
REGISTER_KERNEL_BUILDER(Name("NonMaxSuppressionV4")
                            .TypeConstraint<Eigen::half>("T")
                            .Device(DEVICE_CPU),
                        NonMaxSuppressionV4Op<CPUDevice, Eigen::half>);

REGISTER_KERNEL_BUILDER(
    Name("NonMaxSuppressionWithOverlaps").Device(DEVICE_CPU),
    NonMaxSuppressionWithOverlapsOp<CPUDevice>);

REGISTER_KERNEL_BUILDER(Name("NonMaxSuppressionLite").Device(DEVICE_CPU),
        NonMaxSuppressionLiteOp<CPUDevice>);

}  // namespace tensorflow
