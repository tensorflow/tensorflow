// Copyright 2025 TF.Text Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace text {

absl::Status SentenceFragmentShapeFn(
    ::tensorflow::shape_inference::InferenceContext* c) {
  for (int i = 0; i < c->num_outputs(); ++i) {
    c->set_output(i, c->UnknownShapeOfRank(1));
  }

  return absl::OkStatus();
}

REGISTER_OP("SentenceFragments")
    .Attr("input_encoding: string")
    .Attr("errors: {'strict', 'replace', 'ignore'} = 'replace'")
    .Attr("replacement_char: int = 65533")  // 0xFFFD unicode replacement char
    .Attr("replace_control_characters: bool = false")
    .Input("row_lengths: int64")
    .Input("token_start: int64")
    .Input("token_end: int64")
    .Input("token_word: string")
    .Input("token_properties: int64")
    .Output("fragment_start: int64")
    .Output("fragment_end: int64")
    .Output("fragment_properties: int64")
    .Output("terminal_punc_token: int64")
    .Output("output_row_lengths: int64")
    .SetShapeFn(SentenceFragmentShapeFn);

}  // namespace text
}  // namespace tensorflow
