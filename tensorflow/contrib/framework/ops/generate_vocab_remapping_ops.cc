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

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {
REGISTER_OP("GenerateVocabRemapping")
    .Input("new_vocab_file: string")
    .Input("old_vocab_file: string")
    .Attr("new_vocab_offset: int >= 0")
    .Attr("num_new_vocab: int >= 0")
    .Output("remapping: int64")
    .Output("num_present: int32")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));

      int64 new_vocab_offset;
      TF_RETURN_IF_ERROR(c->GetAttr("new_vocab_offset", &new_vocab_offset));
      int64 num_new_vocab;
      TF_RETURN_IF_ERROR(c->GetAttr("num_new_vocab", &num_new_vocab));

      c->set_output(0, c->Vector(num_new_vocab));
      c->set_output(1, c->Scalar());
      return Status::OK();
    })
    .Doc(R"doc(
Given a path to new and old vocabulary files, returns a remapping Tensor of
length `num_new_vocab`, where `remapping[i]` contains the row number in the old
vocabulary that corresponds to row `i` in the new vocabulary (starting at line
`new_vocab_offset` and up to `num_new_vocab` entities), or `-1` if entry `i`
in the new vocabulary is not in the old vocabulary.  `num_vocab_offset` enables
use in the partitioned variable case, and should generally be set through
examining partitioning info.  The format of the files should be a text file,
with each line containing a single entity within the vocabulary.

For example, with `new_vocab_file` a text file containing each of the following
elements on a single line: `[f0, f1, f2, f3]`, old_vocab_file = [f1, f0, f3],
`num_new_vocab = 3, new_vocab_offset = 1`, the returned remapping would be
`[0, -1, 2]`.

The op also returns a count of how many entries in the new vocabulary
were present in the old vocabulary, which is used to calculate the number of
values to initialize in a weight matrix remapping

This functionality can be used to remap both row vocabularies (typically,
features) and column vocabularies (typically, classes) from TensorFlow
checkpoints.  Note that the partitioning logic relies on contiguous vocabularies
corresponding to div-partitioned variables.  Moreover, the underlying remapping
uses an IndexTable (as opposed to an inexact CuckooTable), so client code should
use the corresponding index_table_from_file() as the FeatureColumn framework
does (as opposed to tf.feature_to_id(), which uses a CuckooTable).

new_vocab_file: Path to the new vocab file.
old_vocab_file: Path to the old vocab file.
new_vocab_offset: How many entries into the new vocab file to start reading.
num_new_vocab: Number of entries in the new vocab file to remap.
remapping: A Tensor of length num_new_vocab where the element at index i
  is equal to the old ID that maps to the new ID i.  This element is -1 for any
  new ID that is not found in the old vocabulary.
num_present: Number of new vocab entries found in old vocab.
)doc");
}  // namespace tensorflow
