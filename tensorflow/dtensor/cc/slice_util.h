/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_DTENSOR_CC_SLICE_UTIL_H_
#define TENSORFLOW_DTENSOR_CC_SLICE_UTIL_H_

#include <optional>
#include <string>
#include <vector>

#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/dtensor/cc/tensor_layout.h"

namespace tensorflow {
namespace dtensor {
namespace slice_util {

// Defines a token of the strided slicing mini-language.
// Refer to the definition of StridedSlice Op for the informal definition of
// the language. During slicing, axes of the input tensor are processed one
// by one according to the tokens of the slicing spec vector.
struct Token {
  enum TokenType {
    REGULAR,   // Slice the current axis by begin/end/begin_mask/end_mask and
               // stride.
    NEW_AXIS,  // Add a new axis at the current location to the output.
    ELLIPSIS,  // Copy over following axes to the output till the ellipsis ends.
    SHRINK_AXIS  // Like a regular axis, but sequeeze this axis from output
                 // after slicing.
  } token_type;

  int64_t begin = 0;          // Begin of the slice.
  int64_t end = 0;            // End of the slice.
  int64_t stride = 0;         // Stride of the slice.
  bool dynamic_mask = false;  // If begin, end, or stride is a dynamic value.
  bool begin_mask = false;    // True if the begin is maximal.
  bool end_mask = false;      // True if the end is maximal.

  Token() = default;
  Token(TokenType token_type, int64_t begin, int64_t end, int64_t stride,
        bool dynamic_mask = false, bool begin_mask = false,
        bool end_mask = false)
      : token_type(token_type),
        begin(begin),
        end(end),
        stride(stride),
        dynamic_mask(dynamic_mask),
        begin_mask(begin_mask),
        end_mask(end_mask) {}

  // Normalizes the token such that (end - begin) is evenly divided by stride,
  // and the result equals the total elements after the slicing.
  Token normalize(int64_t dim_size) const;
  std::optional<Token> GetLocalToken(int64_t dim_size,
                                     int64_t num_shards) const;
};

// TODO(feyu): is there a C++ way to do vari args and templates move this out
// of this class?
template <typename T, typename... Types>
StatusOr<T> CreateAndRun(const std::vector<Token>& tokens, Types... args) {
  T visitor(args...);
  TF_RETURN_IF_ERROR(visitor.Run(tokens));
  return visitor;
}

class TokenProcessor {
 public:
  explicit TokenProcessor(int64_t input_rank) : input_rank_(input_rank) {}
  virtual ~TokenProcessor() = default;

  Status Run(const std::vector<Token>& tokens);

 protected:
  // Loop for an ellipsis or the unconsumed axes in the end.
  bool VisitLoop(int64_t input_rank, int64_t output_rank, int64_t ellipsis_size,
                 int64_t* input_index, int64_t* output_index);

  virtual void VisitImplicitAxis(int64_t input_index, int64_t output_index) = 0;

  virtual void VisitEllipsisAxis(const Token& token) = 0;

  virtual void VisitShrinkAxis(const Token& token, int64_t input_index,
                               int64_t output_index) = 0;

  virtual void VisitNewAxis(const Token& token, int64_t input_index,
                            int64_t output_index) = 0;

  virtual void VisitRegularAxis(const Token& token, int64_t input_index,
                                int64_t output_index) = 0;

  virtual void PrepareResults(int64_t spec_rank, int64_t input_rank,
                              int64_t output_rank) = 0;

  virtual Status FinalizeResults(int64_t input_rank, int64_t output_rank) = 0;

 private:
  const int64_t input_rank_;
};

// Forward layout inference of from a StridedSlice token vector.
//
// For value_layout = StridedSlice(input_layout, tokens)
//
// The inference consumes input_layout, and produces:
//  - a planned expander_input_layout that is suitable for SPMD expansion.
//  - a planned expander_value_layout that is suitable for SPMD expansion.
//  - a local_tokens vector for the arguments of the post-SPMD StridedSliceOp.
//  expander_input_layout and expander_value_layout are consistent with
//  local_tokens.
class ForwardLayoutInference : public TokenProcessor {
 public:
  ForwardLayoutInference(const Layout& input_layout,
                         const llvm::ArrayRef<int64_t> input_shape)
      : TokenProcessor(input_shape.size()),
        input_layout_(input_layout),
        input_shape_(input_shape),
        input_sharding_(input_layout.sharding_spec_strs()) {}

  const Layout& expander_value_layout() const { return expander_value_layout_; }

  const Layout& expander_input_layout() const { return expander_input_layout_; }

  const std::vector<Token>& local_tokens() const { return local_tokens_; }

 protected:
  void VisitEllipsisAxis(const Token& token) override {
    local_tokens_.push_back(token);
  }

  void VisitImplicitAxis(int64_t input_index, int64_t output_index) override {
    expander_input_sharding_.push_back(input_sharding_[output_index]);
    expander_value_sharding_.push_back(input_sharding_[output_index]);
  }

  void VisitShrinkAxis(const Token& token, int64_t input_index,
                       int64_t output_index) override {
    local_tokens_.push_back(token);
    expander_input_sharding_.push_back(Layout::kUnshardedDim);
    // Skips this axis from values, since it will be removed from the inputs.
  }

  void VisitNewAxis(const Token& token, int64_t input_index,
                    int64_t output_index) override {
    local_tokens_.push_back(token);
    expander_value_sharding_.push_back(Layout::kUnshardedDim);
  }

  void VisitRegularAxis(const Token& token, int64_t input_index,
                        int64_t output_index) override {
    auto local_token = token.GetLocalToken(
        /*dim_size=*/input_shape_[input_index],
        /*num_shards*/ input_layout_.num_shards_for_dim(input_index));
    std::string sharding = input_sharding_[input_index];
    if (local_token.has_value()) {
      local_tokens_.push_back(*local_token);
    } else {
      sharding = Layout::kUnshardedDim;
      local_tokens_.push_back(token);
    }
    expander_value_sharding_.push_back(sharding);
    expander_input_sharding_.push_back(sharding);
  }

  void PrepareResults(int64_t spec_rank, int64_t input_rank,
                      int64_t output_rank) override {
    local_tokens_.reserve(spec_rank);
    expander_input_sharding_.reserve(input_rank);
    expander_value_sharding_.reserve(output_rank);
  }

  Status FinalizeResults(int64_t input_rank, int64_t output_rank) override {
    DCHECK_EQ(expander_input_sharding_.size(), input_rank);
    DCHECK_EQ(expander_value_sharding_.size(), output_rank);
    TF_ASSIGN_OR_RETURN(
        expander_input_layout_,
        Layout::GetLayout(expander_input_sharding_, input_layout_.mesh()));
    TF_ASSIGN_OR_RETURN(
        expander_value_layout_,
        Layout::GetLayout(expander_value_sharding_, input_layout_.mesh()));
    return absl::OkStatus();
  }

 private:
  const Layout& input_layout_;
  const llvm::ArrayRef<int64_t> input_shape_;
  std::vector<std::string> input_sharding_;
  std::vector<std::string> expander_value_sharding_;
  std::vector<std::string> expander_input_sharding_;
  // Outputs
  Layout expander_value_layout_;
  Layout expander_input_layout_;
  std::vector<Token> local_tokens_;
};

// Backward layout inference for a StridedSlice token vector.
//
// For value_layout = StridedSlice(input_layout, tokens)
//
// The inference consumes value_layout, and produces:
//  - a planned expander_input_layout that is suitable for SPMD expansion.
//  - a planned expander_value_layout that is suitable for SPMD expansion.
//  - a local_tokens vector for the arguments of the post-SPMD StridedSliceOp.
//  expander_input_layout and expander_value_layout are consistent with
//  local_tokens.
class BackwardLayoutInference : public TokenProcessor {
 public:
  BackwardLayoutInference(const Layout& value_layout,
                          const llvm::ArrayRef<int64_t> input_shape)
      : TokenProcessor(input_shape.size()),
        value_layout_(value_layout),
        input_shape_(input_shape),
        value_sharding_(value_layout.sharding_spec_strs()) {}

  const Layout& expander_input_layout() const { return expander_input_layout_; }

  const Layout& expander_value_layout() const { return expander_value_layout_; }

  const std::vector<Token>& local_tokens() const { return local_tokens_; }

 protected:
  void VisitEllipsisAxis(const Token& token) override {
    local_tokens_.push_back(token);
  }

  void VisitImplicitAxis(int64_t input_index, int64_t output_index) override {
    expander_input_sharding_.push_back(value_sharding_[output_index]);
    expander_value_sharding_.push_back(value_sharding_[output_index]);
  }

  void VisitShrinkAxis(const Token& token, int64_t input_index,
                       int64_t output_index) override {
    local_tokens_.push_back(token);
    // There is no constraint on the input sharding, but we prefer to keep it
    // unsharded to avoid inserting relayout toward the internal input layout.
    expander_input_sharding_.push_back(Layout::kUnshardedDim);
  }

  void VisitNewAxis(const Token& token, int64_t input_index,
                    int64_t output_index) override {
    local_tokens_.push_back(token);
    // No corresponding input axis.
    expander_value_sharding_.push_back(Layout::kUnshardedDim);
  }

  void VisitRegularAxis(const Token& token, int64_t input_index,
                        int64_t output_index) override {
    auto local_token = token.GetLocalToken(
        /*dim_size=*/input_shape_[input_index],
        /*num_shards*/ value_layout_.num_shards_for_dim(output_index));
    if (local_token.has_value()) {
      std::string sharding = value_sharding_[output_index];
      local_tokens_.push_back(*local_token);
      expander_input_sharding_.push_back(sharding);
      expander_value_sharding_.push_back(sharding);
    } else {
      local_tokens_.push_back(token);
      // There is no constraint on the input sharding, but we prefer to keep it
      // unsharded to avoid inserting relayout toward the internal input layout.
      expander_input_sharding_.push_back(Layout::kUnshardedDim);
      expander_value_sharding_.push_back(Layout::kUnshardedDim);
    }
  }

  void PrepareResults(int64_t spec_rank, int64_t input_rank,
                      int64_t output_rank) override {
    local_tokens_.reserve(spec_rank);
    expander_input_sharding_.reserve(input_rank);
    expander_value_sharding_.reserve(output_rank);
  }

  Status FinalizeResults(int64_t input_rank, int64_t output_rank) override {
    DCHECK_EQ(expander_input_sharding_.size(), input_rank);
    DCHECK_EQ(expander_value_sharding_.size(), output_rank);
    TF_ASSIGN_OR_RETURN(
        expander_input_layout_,
        Layout::GetLayout(expander_input_sharding_, value_layout_.mesh()));
    TF_ASSIGN_OR_RETURN(
        expander_value_layout_,
        Layout::GetLayout(expander_value_sharding_, value_layout_.mesh()));
    return absl::OkStatus();
  }

 private:
  const Layout& value_layout_;
  const llvm::ArrayRef<int64_t> input_shape_;
  std::vector<std::string> value_sharding_;
  std::vector<std::string> expander_input_sharding_;
  std::vector<std::string> expander_value_sharding_;
  // Outputs
  Layout expander_input_layout_;
  Layout expander_value_layout_;
  std::vector<Token> local_tokens_;
};

}  // namespace slice_util
}  // namespace dtensor
}  // namespace tensorflow
#endif  // TENSORFLOW_DTENSOR_CC_SLICE_UTIL_H_
