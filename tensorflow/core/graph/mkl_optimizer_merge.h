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

// An optimization pass that performs node merging and rewrite on graph nodes

#ifndef TENSORFLOW_CORE_GRAPH_MKL_OPTIMIZER_MERGE_H_
#define TENSORFLOW_CORE_GRAPH_MKL_OPTIMIZER_MERGE_H_

#ifdef INTEL_MKL

#include <sys/types.h>
#include <vector>
#include <string>
#include <memory>
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/common_runtime/optimization_registry.h"

namespace tensorflow {

// How many hops do we search for matching node in the backward dataflow graph?
// We use maxhop of 10 based on empirical observations. Also, these are
// maxhops in backward data-flow graph. Since input of forward nodes (Conv2D)
// directly goes to backward nodes, we do not expect the hop-distance
// would be more than few nodes.
#define NODEMERGE_CONTEXT_MAXDEPTH 10

// This optimization pass performs two tasks: merge
// nodes in the forward pass, and rewrite the gradient ops
// corresponding to merged forward ops.
//
// Merging nodes in the graph: Currently, it merges Conv2D+AddBias
// and MatMul+AddBias nodes together.
//
// Rewriting nodes in the graph:
// This is neded in order to optimize gradient ops of
// Conv2D+AddBias and MatMul+AddBias. Gradient op of
// both the Conv2D and MatMul is BiasAddGrad, and we
// need to rewrite BiasAddGrad into Conv2D-specific BiasAddGrad,
// and MatMul-specific BiasAddGrad. This is context-specific
// optimization, where the context is the forward operator
// that the BiasAddGrad corresponds to.
//
class NodeMergeRewritePass : public GraphOptimizationPass {
 public:
  NodeMergeRewritePass() {
    csinfo_.conv2d                     = "Conv2D";
    csinfo_.conv2dwithbias             = "Conv2DWithBias";
    csinfo_.conv2dwithbiasbackpropbias = "Conv2DWithBiasBackpropBias";
    csinfo_.biasadd                    = "BiasAdd";
    csinfo_.matmul                     = "MatMul";
    csinfo_.matmulmkl                  = "MatMulMkl";
    csinfo_.add                        = "Add";
    csinfo_.biasaddgrad                = "BiasAddGrad";

    minfo_.push_back({csinfo_.conv2d, csinfo_.biasadd, 0,
                      csinfo_.conv2dwithbias});
    // NOTE: We no longer use this optimization now.
    // minfo_.push_back({csinfo_.matmul, csinfo_.add, 0,
    //                csinfo_.matmulmkl});

    // We use maxhop of 10 based on emperical observations. Also, these are
    // maxhops in backward data-flow graph. Since input of forward nodes
    // (Conv2D) directly goes to backward nodes, we do not expect the
    // hop-distance would be more than few nodes.
    rinfo_.push_back({csinfo_.biasaddgrad, csinfo_.conv2dwithbiasbackpropbias,
                  {csinfo_.conv2dwithbias, NODEMERGE_CONTEXT_MAXDEPTH}});
    rinfo_.push_back({csinfo_.biasaddgrad, csinfo_.conv2dwithbiasbackpropbias,
                  {csinfo_.conv2d, NODEMERGE_CONTEXT_MAXDEPTH}});
    // For now, we are rewriting BiasAddGrad to BiasAddGrad for MatMul. This is
    // because we do not have a separate Op for MatMulwithBias.
    rinfo_.push_back({csinfo_.biasaddgrad, csinfo_.biasaddgrad,
                      {csinfo_.matmul, NODEMERGE_CONTEXT_MAXDEPTH}});
    rinfo_.push_back({csinfo_.biasaddgrad, csinfo_.biasaddgrad,
                      {csinfo_.matmulmkl, NODEMERGE_CONTEXT_MAXDEPTH}});
  }

  Status Run(const GraphOptimizationPassOptions& options);

  /*
   * Helper function which does most of heavy lifting for node merge
   *
   * Extracts common functionality between Run public interface and
   * test interface.
   *
   * @return true, if and only if graph is mutated; false otherwise.
   */
  bool DoNodeMerge(std::unique_ptr<Graph>* g);

 private:
  /// Structure to specify information used in node merge
  typedef struct {
    string pred;  // Predecessor node string
    string succ;  // Successor node string
    int    op;    // What operand no the predecessor node corresponds
                  // to successor node?
    string newnode;  // Name of the node after merge
  } MergeInfo;

  /// Structure to specify information used in node rewrite
  typedef struct {
    string node;  // Name of the node to be rewritten
    string rewrite;  // New name of the node after rewrite
    typedef struct {
        string fwd;  // Node name in forward pass that this node
                       // corresponds to
        int maxhop;  // Maximum number of hops the mfwd_ is located
                         // from this node. If mfwd_ is farther than mmaxhop_
                         // then we do not rewrite the node.
    } ContextInfo;
    ContextInfo cinfo;  // Context for rewrite
  } RewriteInfo;

  /// Structure to store all constant strings
  typedef struct {
    string conv2d;
    string conv2dwithbias;
    string conv2dwithbiasbackpropbias;
    string biasadd;

    string matmul;
    string matmulmkl;
    string add;

    string biasaddgrad;
  } ConstStringInfo;

  ConstStringInfo csinfo_;
  std::vector<MergeInfo> minfo_;
  std::vector<RewriteInfo> rinfo_;

 private:
  /*
   *  Return a node that can be merged with input node
   *
   *  @return pointer to the node if we can find such a
   *  node. Otherwise, it returns NULL.
   */
  Node* FindNodeForMerge(const Node* a) const;

  /*
   *  Merge predecessor node with its successor.
   *  Currently, we support merging Conv2D with AddBias, and
   *  MatMul with Add node.
   *
   *  Input nodes succ and pred may be deleted if the call to
   *  this function is successful. Attempt to use the pointers
   *  after the call to function may result is undefined behaviors.
   *
   *  @input g - input graph, succ - successor node, pred - predecessor node
   *  @return true, if merging is successful and supported.
   *           Returns false otherwise.
   */
  bool MergeNode(std::unique_ptr<Graph>* g, Node *succ, Node *pred);

  /*
   *  Can the input node (n) be rewritten via rewrite node method?
   *
   *  @return true, if it can be rewritten; false, otherwise.
   *          In case of true, returns context ID of the matching context.
   */
  bool IsApplicableRewriteNode(const Node *n) const;

  /*
   * Rewrite input node to its corresponding node specified in rewrite info.
   *
   * Input node may be deleted in case of rewrite. Attempt to use the node
   * after the call can result in undefined behaviors.
   *
   * @input  g - input graph, n - Node to be rewritten
   * @return true, if the input node can be rewritten; false, otherwise.
   *         Graph is updated in case the input node can be rewritten.
   *         Otherwise, it is not updated.
   */
  bool RewriteNode(std::unique_ptr<Graph>* g, Node *n);

  /*
   * Helper function that searches the matching rewriteinfo for the node.
   * Implements depth-first search in the data dependence graph for the
   * gradient op in backward direction.
   *
   * @input n - Node (gradient op) whose rewriteinfo is to be searched,
   *        fwdn - pointer to node from the forward pass that this node
   *        belongs to
   * @return Matching rewriteinfo in case a match is found; null otherwise.
   */
  const RewriteInfo* FindMatchingRewriteInfo(const Node *n,
                                             const Node **fwdn) const;
};

/*
 * Interface to invoke the pass for unit test
 */
extern bool OptimizeNodeMerge(std::unique_ptr<Graph>* g);

/* We register merge optimizer for phase 1 and MKLToTF insertion
 * for phase 2.
 */
REGISTER_OPTIMIZATION(OptimizationPassRegistry::PRE_PLACEMENT, 1,
                      NodeMergeRewritePass);
}  // namespace tensorflow

#endif

#endif  // TENSORFLOW_CORE_GRAPH_MKL_OPTIMIZER_MERGE_H_
