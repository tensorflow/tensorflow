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

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {
namespace text {

REGISTER_OP("MaxSpanningTree")
    .Attr("T: {int32, float, double}")
    .Attr("forest: bool = false")
    .Input("num_nodes: int32")
    .Input("scores: T")
    .Output("max_scores: T")
    .Output("argmax_sources: int32")
    .SetShapeFn([](tensorflow::shape_inference::InferenceContext *context) {
      tensorflow::shape_inference::ShapeHandle num_nodes;
      tensorflow::shape_inference::ShapeHandle scores;
      TF_RETURN_IF_ERROR(context->WithRank(context->input(0), 1, &num_nodes));
      TF_RETURN_IF_ERROR(context->WithRank(context->input(1), 3, &scores));

      // Extract dimensions while asserting that they match.
      tensorflow::shape_inference::DimensionHandle batch_size;  // aka "B"
      TF_RETURN_IF_ERROR(context->Merge(context->Dim(num_nodes, 0),
                                        context->Dim(scores, 0), &batch_size));
      tensorflow::shape_inference::DimensionHandle max_nodes;  // aka "M"
      TF_RETURN_IF_ERROR(context->Merge(context->Dim(scores, 1),
                                        context->Dim(scores, 2), &max_nodes));

      context->set_output(0, context->Vector(batch_size));
      context->set_output(1, context->Matrix(batch_size, max_nodes));
      return absl::OkStatus();
    })
    .Doc(R"doc(
Finds the maximum directed spanning tree of a digraph.

Given a batch of directed graphs with scored arcs and root selections, solves
for the maximum spanning tree of each digraph, where the score of a tree is
defined as the sum of the scores of the arcs and roots making up the tree.

Returns the score of the maximum spanning tree of each digraph, as well as the
arcs and roots in that tree.  Each digraph in a batch may contain a different
number of nodes, so the sizes of the digraphs must be provided as an input.

Note that this operation is only differentiable w.r.t. its |scores| input and
its |max_scores| output.

The code here is intended for NLP applications, but attempts to remain
agnostic to particular NLP tasks (such as dependency parsing).

forest: If true, solves for a maximum spanning forest instead of a maximum
        spanning tree, where a spanning forest is a set of disjoint trees that
        span the nodes of the digraph.
num_nodes: [B] vector where entry b is number of nodes in the b'th digraph.
scores: [B,M,M] tensor where entry b,t,s is the score of the arc from node s to
        node t in the b'th directed graph if s!=t, or the score of selecting
        node t as a root in the b'th digraph if s==t. This uniform tenosor
        requires that M is >= num_nodes[b] for all b (ie. all graphs in the
        batch), and ignores entries b,s,t where s or t is >= num_nodes[b].
        Arcs or root selections with non-finite score are treated as
        nonexistent.
max_scores: [B] vector where entry b is the score of the maximum spanning tree
            of the b'th digraph.
argmax_sources: [B,M] matrix where entry b,t is the source of the arc inbound to
                t in the maximum spanning tree of the b'th digraph, or t if t is
                a root. Entries b,t where t is >= num_nodes[b] are set to -1.
                Quickly finding the roots can be done as:
                tf.equal(tf.map_fn(lambda x: tf.range(tf.size(x)),
                argmax_sources), argmax_sources)
)doc");

}  // namespace text
}  // namespace tensorflow
