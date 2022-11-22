// License TODO ....

// For each subgraph in a computational graph that can be split, eXLA will split
// the subgraph and create a while loop for it to replace the subgraph. In many
// cases, two subgraphs may have a large portion of common paths, but these two
// subgraphs are split and replaced with two different while loops, resulting in
// the common paths being computed multiple times. In order to merge those
// while-loops that can be merged, aiming at speeding up memory-optimised
// algorithms while maintaining the memory benefits of eXLA-v1.

#include "tensorflow/compiler/xla/service/tensor_splitter.h"

#include <stdlib.h>

#include <queue>

#include "absl/algorithm/container.h"
#include "tensorflow/compiler/xla/debug_options_flags.h"
#include "tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h"
#include "tensorflow/compiler/xla/service/hlo_creation_utils.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/compiler/xla/status_macros.h"
namespace xla {

#define PRIME_SIZE 512
const int64_t prime_numbers[PRIME_SIZE] = {
    2,    3,    5,    7,    11,   13,   17,   19,   23,   29,   31,   37,
    41,   43,   47,   53,   59,   61,   67,   71,   73,   79,   83,   89,
    97,   101,  103,  107,  109,  113,  127,  131,  137,  139,  149,  151,
    157,  163,  167,  173,  179,  181,  191,  193,  197,  199,  211,  223,
    227,  229,  233,  239,  241,  251,  257,  263,  269,  271,  277,  281,
    283,  293,  307,  311,  313,  317,  331,  337,  347,  349,  353,  359,
    367,  373,  379,  383,  389,  397,  401,  409,  419,  421,  431,  433,
    439,  443,  449,  457,  461,  463,  467,  479,  487,  491,  499,  503,
    509,  521,  523,  541,  547,  557,  563,  569,  571,  577,  587,  593,
    599,  601,  607,  613,  617,  619,  631,  641,  643,  647,  653,  659,
    661,  673,  677,  683,  691,  701,  709,  719,  727,  733,  739,  743,
    751,  757,  761,  769,  773,  787,  797,  809,  811,  821,  823,  827,
    829,  839,  853,  857,  859,  863,  877,  881,  883,  887,  907,  911,
    919,  929,  937,  941,  947,  953,  967,  971,  977,  983,  991,  997,
    1009, 1013, 1019, 1021, 1031, 1033, 1039, 1049, 1051, 1061, 1063, 1069,
    1087, 1091, 1093, 1097, 1103, 1109, 1117, 1123, 1129, 1151, 1153, 1163,
    1171, 1181, 1187, 1193, 1201, 1213, 1217, 1223, 1229, 1231, 1237, 1249,
    1259, 1277, 1279, 1283, 1289, 1291, 1297, 1301, 1303, 1307, 1319, 1321,
    1327, 1361, 1367, 1373, 1381, 1399, 1409, 1423, 1427, 1429, 1433, 1439,
    1447, 1451, 1453, 1459, 1471, 1481, 1483, 1487, 1489, 1493, 1499, 1511,
    1523, 1531, 1543, 1549, 1553, 1559, 1567, 1571, 1579, 1583, 1597, 1601,
    1607, 1609, 1613, 1619, 1621, 1627, 1637, 1657, 1663, 1667, 1669, 1693,
    1697, 1699, 1709, 1721, 1723, 1733, 1741, 1747, 1753, 1759, 1777, 1783,
    1787, 1789, 1801, 1811, 1823, 1831, 1847, 1861, 1867, 1871, 1873, 1877,
    1879, 1889, 1901, 1907, 1913, 1931, 1933, 1949, 1951, 1973, 1979, 1987,
    1993, 1997, 1999, 2003, 2011, 2017, 2027, 2029, 2039, 2053, 2063, 2069,
    2081, 2083, 2087, 2089, 2099, 2111, 2113, 2129, 2131, 2137, 2141, 2143,
    2153, 2161, 2179, 2203, 2207, 2213, 2221, 2237, 2239, 2243, 2251, 2267,
    2269, 2273, 2281, 2287, 2293, 2297, 2309, 2311, 2333, 2339, 2341, 2347,
    2351, 2357, 2371, 2377, 2381, 2383, 2389, 2393, 2399, 2411, 2417, 2423,
    2437, 2441, 2447, 2459, 2467, 2473, 2477, 2503, 2521, 2531, 2539, 2543,
    2549, 2551, 2557, 2579, 2591, 2593, 2609, 2617, 2621, 2633, 2647, 2657,
    2659, 2663, 2671, 2677, 2683, 2687, 2689, 2693, 2699, 2707, 2711, 2713,
    2719, 2729, 2731, 2741, 2749, 2753, 2767, 2777, 2789, 2791, 2797, 2801,
    2803, 2819, 2833, 2837, 2843, 2851, 2857, 2861, 2879, 2887, 2897, 2903,
    2909, 2917, 2927, 2939, 2953, 2957, 2963, 2969, 2971, 2999, 3001, 3011,
    3019, 3023, 3037, 3041, 3049, 3061, 3067, 3079, 3083, 3089, 3109, 3119,
    3121, 3137, 3163, 3167, 3169, 3181, 3187, 3191, 3203, 3209, 3217, 3221,
    3229, 3251, 3253, 3257, 3259, 3271, 3299, 3301, 3307, 3313, 3319, 3323,
    3329, 3331, 3343, 3347, 3359, 3361, 3371, 3373, 3389, 3391, 3407, 3413,
    3433, 3449, 3457, 3461, 3463, 3467, 3469, 3491, 3499, 3511, 3517, 3527,
    3529, 3533, 3539, 3541, 3547, 3557, 3559, 3571, 3581, 3583, 3593, 3607,
    3613, 3617, 3623, 3631, 3637, 3643, 3659, 3671};

namespace {
namespace m = match;
// TensorSplitProperties is a tuple with the format of (split_size, split_count,
// split_rest)
using TensorSplitProperties = std::tuple<int64_t, int64_t, int64_t>;
#define CREATE_CONSTANT_INT32(number) \
  HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32_t>(number))

// SplitNodeKey = inst_id, a key is the inst_id of the final result node of the
// split path
using SplitNodeKey = int64_t;
// SplitNodeVal = <inst_ptr,split_dim,split_size,parent_inst_ptr>
using SplitNodeVal =
    std::tuple<HloInstruction*, int64_t, int64_t, HloInstruction*>;

// A helper function to generate a SplitNodeKey instance
//  Input:
//    inst: the target instruction
//  Output:
//    the SplitNodeKey of the given inst
SplitNodeKey MakeSplitNodeKey(const HloInstruction* inst) {
  return inst->unique_id();
}

// A helper function to check if two SplitNodeVal are equal
//  Input:
//    first_val: first SplitNodeVal
//    second_val: second SplitNodeVal
//  Output:
//    a bool value indicates if the two values are equal
bool SplitNodeValEqual(const SplitNodeVal& first_val,
                       const SplitNodeVal& second_val) {
  if (std::get<1>(first_val) == -1 || std::get<1>(second_val) == -1) {
    return false;
  }
  // two node_val are equal if and onlf if they have the same inst_id, split_dim
  // and split_size
  return std::get<0>(first_val)->unique_id() ==
             std::get<0>(second_val)->unique_id() &&
         std::get<1>(first_val) == std::get<1>(second_val) &&
         std::get<2>(first_val) == std::get<2>(second_val);
}

//  Input:
//    inst: an instruction pointer
//    split_dim: the split dimension of the instruction
//    split_size: the split size of the instruction
//  Output:
//    a SplitNodeVal made from the given input
SplitNodeVal MakeSplitNodeVal(HloInstruction* inst, int64_t split_dim,
                              int64_t split_size,
                              HloInstruction* parent_inst = nullptr) {
  return std::make_tuple(inst, split_dim, split_size, parent_inst);
}

// A base visitor class which decides splitting information like whether to
// split an instruction or not, best split dimension and split size
class SplitDeterminer : public DfsHloRewriteVisitor {
 public:
  // Constructor of SplitDeterminer
  //  Input:
  //    max_size_threshold: a thereshold decides whether an instruction is big
  //      enought to split
  //    target_split_size: the target size of an instruction after splitting
  //    split_size: the split size of the instruction
  //  Output:
  //    a SplitDeterminer instance
  explicit SplitDeterminer(int64_t max_size_threshold,
                           int64_t target_split_size)
      : max_size_threshold(max_size_threshold),
        target_split_size(target_split_size) {}

  //  Input:
  //    inst: an instruction pointer
  //  Output:
  //    a bool value indicates if an operand is large enough such that we
  //    are interested in splitting it.
  bool OperandShouldBeSplit(HloInstruction* inst);

  // Determine if an operand can be split by traversing it's
  // inputs until a splittable node is found. This will also
  // directly modifies the list of leafs, the list of exclude_dimensions which
  // can not be split (if an internal op is only partially point- wise).
  //  Input:
  //    inst: an instruction pointer
  //    split_leafs: a vector stores all found leaf nodes during the path
  //    traversal original_dimensions: a vector mapping the current instruction
  //    dimensions to the dimensions
  //      root node of the split_path.
  //    exclude_dimensions: a vector stores all non-splittable dimensions of the
  //      root node of the spit_path
  //  Output:
  //    a bool value indicates whether an operation can be split
  bool OperandCanBeSplit(HloInstruction* inst,
                         std::vector<HloInstruction*>* split_leafs = nullptr,
                         std::vector<int64_t>* original_dimensions = nullptr,
                         std::vector<int64_t>* exclude_dimensions = nullptr);

  // Matches any pointwise unary operator which has no side effects. If so,
  // assign the pointer of its operand to the input operand pointer
  //  Input:
  //    inst: an instruction pointer
  //    operand: an pointer used to record the operand of the PointwiseUnary
  //      instruction
  //  Output:
  //    a bool value indicates whether an operation is a PointwiseUnary and
  //    doesn't have any side effections
  static bool MatchPointwiseUnary(HloInstruction* inst,
                                  HloInstruction** operand = nullptr);

  // Matches any pointwise n-ary operator.
  //  Input:
  //    inst: an instruction pointer
  //    operands: an vector used to record all the operands of the PointwiseNary
  //      instruction
  //  Output:
  //    a bool value indicates whether an operation is a PointwiseNary and
  //    doesn't have any side effections
  static bool MatchPointwiseNary(
      HloInstruction* inst, std::vector<HloInstruction*>* operands = nullptr);

  // Matches a reduce operation where all operands have the same shape
  // and all initilizers are scalars.
  //  Input:
  //    inst: an instruction pointer
  //  Output:
  //    a bool value indicates whether an operation is a reduce operation where
  //    all operands have the same shape and all initilizers are scalars.
  static bool MatchSupportedReduce(HloInstruction* inst);

  static bool MatchSupportedNestedReduce(HloInstruction* inst);

  // Determine all the best dimesions to split on, excluding a given one.
  //  Input:
  //    inst: an instruction pointer
  //    excluded: a vector all non-splittable dimensions of the given inst
  //  Output:
  //    a vector contains all best split dimensions of the inst
  std::vector<int64_t> BestSplitDim(HloInstruction* inst,
                                    absl::Span<const int64_t> excluded);

  int64_t BestEvenSplitSizeFold(int64_t (&factors)[PRIME_SIZE], int offset,
                                int64_t current, int64_t best, int64_t size,
                                int64_t max_size);

  // Given a split dimension, determine the best possible split
  // size with equally shaped pieces. If no split size is possible, returns -1.
  //  Input:
  //    inst: an instruction pointer
  //    split_dim: the split dimension of the given inst
  //  Output:
  //    a number represents the best even split size on split_dim of the inst
  int64_t BestEvenSplitSize(HloInstruction* inst, int64_t split_dim);

  // Given a split dimension, determine the best possible split
  // size, allowing for un uneven split. Split_count denotes the
  // number of pieces of split_size size; split_rest is the size of
  // the last piece.
  //  Input:
  //    inst: an instruction pointer
  //    split_dim: the split dimension of the given inst
  //  Output:
  //    a TensorSplitProperties class contains the best split information on
  //    split_dim of the given inst
  TensorSplitProperties DetermineSplitSize(HloInstruction* inst,
                                           int64_t split_dim);

 protected:
  // thereshold determines if we are interested to split an instruction
  int64_t max_size_threshold;
  // the target instruction size after splitting
  int64_t target_split_size;
};

// In eXLA-v1, the splitting is completed during a traversal of a computational
// graph, but in order to merge the subgraphs that can be merged after the
// splitting, we need to divide the original splitting process into three
// stages: record the paths that need to be split, assign while-loops, perform
// splitting and replacing. The general recording process is similar to what
// eXLA-v1 does. Every time we meet a node that need to be split during
// traversing, we then calculate all the dimensions in which this node can be
// split, i.e. split_dim and its corresponding split_size, and then for each
// split_dim, record its split_path on that dimension. For each node in a
// split_path, we need to record its original node and its split_size and
// split_dim in the path. Also for each split_dim, we need to mark whether it is
// a contracting/reduce dimension,, which will be eliminated in the operation
// and not retained in the resulting dimension. As this affects whether the two
// paths can be merged, which will be discussed in detail in subsequent
// sections. A split_path terminates in a number of split_leaves. Four types of
// nodes may be used as a split_leaf in eXLA-v1: dot, broadcast, iota, and
// parameter. The first three types of nodes can be considered split_leaves if
// they accept a small tensor as input, but output a very large tensor. eXLA-v2
// continues to use these as split_leaves.

// a visitor class which records all splittable paths and decides how to merge
// paths
class SplittablePathRecorder : public SplitDeterminer {
 public:
  // Constructor of SplittablePathRecorder
  //  Input:
  //    max_size_threshold: a thereshold decides whether an instruction is big
  //      enought to split
  //    target_split_size: the target size of an instruction after splitting
  //    split_size: the split size of the instruction
  //    input_parent_module: the pointer of the parent module
  //  Output:
  //    a SplittablePathRecorder instance
  explicit SplittablePathRecorder(int64_t max_size_threshold,
                                  int64_t target_split_size,
                                  HloModule* input_parent_module)
      : SplitDeterminer(max_size_threshold, target_split_size),
        parent_module(input_parent_module),
        while_loop_num_counter(0),
        lcs_merge_threshold(0.1) {
    // TODO: FIND A MORE SUITABLE THRESHOLD
  }
  Status DefaultAction(HloInstruction* hlo) override { return OkStatus(); }
  Status FinishVisit(HloInstruction* hlo) override { return OkStatus(); }

  // Create a new empty split path using the give key and inst, store the path
  // information into corresponding data structure
  //  Input:
  //    key: the key of a split path
  //    inst: the root instruction of a split path
  //  Output:
  //    a status indicates if the path is created successfully
  Status CreateNewEmptyPath(SplitNodeKey key, HloInstruction* inst);

  // Add a node into a split path
  //  Input:
  //    path_key: the key of a split path
  //    path_index: the path index of the path of the given path_key(a key may
  //      has several paths because a root node can be split in multiple ways)
  //    inst: the node instruction
  //    split_dim: the split dimension of the inst
  //    split_size: the split size of the inst
  //    parent_inst: the parent instruction of the inst
  //  Output:
  //    a status indicates if the node is added into the path successfully
  Status AppendToPath(SplitNodeKey path_key, int64_t path_index,
                      HloInstruction* inst, int64_t split_dim,
                      int64_t split_size, HloInstruction* parent_inst);

  // Recursively record the path with the given key
  //  Input:
  //    path_key: the key of a split path
  //    path_index: the path index of the path of the given path_key(a key may
  //      has several paths because a root node can be split in multiple ways)
  //    inst: the node instruction
  //    split_dim: the split dimension of the inst
  //    split_size: the split size of the inst
  //    split_leafs: all leaf nodes of the split path
  //    parent_inst: the parent instruction of the inst
  //  Output:
  //    a status indicates if the path is recorded successfully
  Status RecordPath(SplitNodeKey path_key, int64_t path_index,
                    HloInstruction* inst, int64_t split_dim, int64_t split_size,
                    std::vector<HloInstruction*>& split_leafs,
                    HloInstruction* parent_inst);

  // Recording handle function when visiting a dot instruction
  //  Input:
  //    dot: currently visited dot instruction
  //  Output:
  //    a status indicates if the recording process is successful
  Status HandleDot(HloInstruction* dot) override;

  // Recording handle function when visiting a reduce instruction
  //  Input:
  //    reduce: currently visited reduce instruction
  //  Output:
  //    a status indicates if the recording process is successful
  Status HandleReduce(HloInstruction* reduce) override;

  // Recording handle function when visiting a sort instruction
  //  Input:
  //    reduce: currently visited sort instruction
  //  Output:
  //    a status indicates if the recording process is successful
  Status HandleSort(HloInstruction* sort) override;

  // Decide how to merge recorded paths and allocate while_loop num to each path
  //  Output:
  //    a status indicates if all paths are allocted with a while loop
  //    successfully
  Status AllocateWhileLoops();

  // Helper function to access start_node_to_splittable_paths
  //  Output:
  //    a reference to start_node_to_splittable_paths
  absl::flat_hash_map<SplitNodeKey, std::vector<std::vector<SplitNodeVal>>>&
  GetStartNodeToSplittablePaths() {
    return start_node_to_splittable_paths;
  }

  // Helper function to access start_node_to_while_loop_num
  //  Output:
  //    a reference to start_node_to_while_loop_num
  absl::flat_hash_map<SplitNodeKey, int64_t>& GetStartNodeToWhileLoopNum() {
    return start_node_to_while_loop_num;
  }

  // Helper function to access while_loop_num_to_start_node
  //  Output:
  //    a reference to while_loop_num_to_start_node
  absl::flat_hash_map<int64_t, std::vector<SplitNodeKey>>&
  GetWhileLoopNumToStartNode() {
    return while_loop_num_to_start_node;
  }

  // Helper function to access while_loop_num_to_instructions
  //  Output:
  //    a reference to while_loop_num_to_instructions
  absl::flat_hash_map<int64_t, absl::flat_hash_set<HloInstruction*>>&
  GetWhileLoopNumToInstructions() {
    return while_loop_num_to_instructions;
  }

  // Helper function to decide whether two SplitNodeVal contain the same
  // instruction but different split dimensions
  //  Output:
  //    a bool indicates if two SplitNodeVal contain the same instruction but
  //    different split dimensions
  bool IsSplitNodeValSameInstDifferentDim(const SplitNodeVal& first_val,
                                          const SplitNodeVal& second_val) {
    // two node_val are equal if and onlf if they have the same inst_id,
    // split_dim and split_size

    return std::get<0>(first_val)->unique_id() ==
               std::get<0>(second_val)->unique_id() &&
           std::get<1>(first_val) != std::get<1>(second_val);
  }

 private:
  // Helper function to generate increasingly unique while loop number
  //  Output:
  //    next availabe while loop number
  int64_t GenerateNewWhileLoopNum() { return while_loop_num_counter++; }

  // Init all starting_node with a unique while_loop_num
  //  Output:
  //    a status indicates if all paths are allocted with an initial while loop
  //    number
  Status InitWhileLoops();

  // Record all dependence relationships
  //  Output:
  //    a status indicates if the searching process is successful
  Status RecordAllDescdantStartNodes();

  // Find all decedant starting_nodes for start_node_key
  //  Input:
  //    start_node_key: search all descdants of the give key
  //  Output:
  //    a status indicates if the searching process is successful
  Status FindAllDescdantStartNodes(SplitNodeKey start_node_key);

  // Check whether the two keys' while_loops can be merged
  //  Input:
  //    first_key: the fisrt key
  //    second_key: the second key
  //  Output:
  //    a status indicates if the two nodes have dependency
  bool HasDesedantRelationship(SplitNodeKey first_key, SplitNodeKey second_key);

  // Try to merge two while_loops using the given available paths, paths which
  // the same index in path arrays are corresponding pair which can be used to
  // merge the two while-loops
  //  Input:
  //    first_key: the key of the root node of the first while-loop
  //    second_key: the key of the root node of the second while-loop
  //    first_path_indices: the availabe paths of the first while-loop
  //    second_path_indices: the availabe paths of the second while-loop
  // Output:
  //    a status indicates if the two nodes have dependency
  Status TryMergableRelativeWhileLoop(
      SplitNodeKey first_key, SplitNodeKey second_key,
      const std::vector<size_t>& first_path_indices,
      const std::vector<size_t>& second_path_indices);

  // If the leaf_dot's operand is another while-loop, record the
  // start node the another while-loop used for furthur merging
  // this is used for the cashe that the root node of a split path is a
  // parameter of another split path. If the two path are split on the same
  // non-contracting/non-reduce dimension the part result of the first split
  // root instruction can be used to compute the part result of the second split
  // path, which means the two paths can be merged into a single while-loop to
  // speed up the computation
  //  Input:
  //    path_key: the key of the split path
  //    path_index: the index of the split path
  //    dot: the pointer of the leaf dot instruction
  //    split_dim: the split dimension of the split path
  //    split_size: the split size of the split path
  // Output:
  //    a status indicates if the recording is successfully
  Status FinishRecordLeafDot(SplitNodeKey path_key, int64_t path_index,
                             HloInstruction* dot, int64_t split_dim,
                             int64_t split_size);

  // The function is the same as the above funciton but used for leaf broadcast
  //  Input:
  //    path_key: the key of the split path
  //    path_index: the index of the split path
  //    broadcast: the pointer of the leaf broadcast instruction
  //    split_dim: the split dimension of the split path
  //    split_size: the split size of the split path
  //  Output:
  //    a status indicates if the recording is successfully
  Status FinishRecordLeafBroadcast(SplitNodeKey path_key, int64_t path_index,
                                   HloInstruction* broadcast, int64_t split_dim,
                                   int64_t split_size);

  // Infer the split dim of a dot instruction given its operand and the split
  // dim of the operand when the splitting is performed on a non-contracting
  // dimension i.e. the split dimension will exist in the dot instruction
  //  Input:
  //    dot: the dot instruction
  //    operand_split_diim: the split dimension of its operand
  //    split_is_lhs: a bool value indicate whetehr the lhs of the dot will be
  //    split
  //  Output:
  //    the split dimension of the dot instruction
  int64_t InferDotResultSplitDim(HloInstruction* dot,
                                 int64_t operand_split_diim, bool split_is_lhs);

  // Infer the split dim of a reduce instruction given its operand and the split
  // dim of the operand when the splitting is performed on a non-reduce
  // dimension i.e. the split dimension will exist in the reduce instruction
  //  Input:
  //    reduce: the reduce instruction
  //    operand_split_diim: the split dimension of its operand
  //  Output:
  //    the split dimension of the reduce instruction
  int64_t InferReduceResultSplitDim(HloInstruction* reduce,
                                    int64_t operand_split_diim);

  // Mark two while loops as unmergable
  //  Input:
  //    first_num: the while-loop number of the first while-loop
  //    second_num: the while-loop number of the second while-loop
  //  Output:
  //    a status indicates whether the recording is successful
  Status RecordUnmergableWhileLoops(int64_t first_num, int64_t second_num);

  // Check whether the two while loops of the two given split paths will not
  // exceed the memory threshold after merging
  //  Input:
  //    first_num: the root key of the first split path
  //    second_num: the root key of the second split path
  // Output:
  //    a bool value indicates whether the two while loop can be merged
  bool SizeCanMerge(SplitNodeKey first_key, SplitNodeKey second_key);

  // Check whether the two given split paths have already been merged into a
  // single while_loop
  //  Input:
  //    first_num: the root key of the first split path
  //    second_num: the root key of the second split path
  //  Output:
  //    a bool value indicates whether the two paths have already been merged
  bool Merged(SplitNodeKey first_key, SplitNodeKey second_key);

  // Calculate the longest common subsequence(the common subpath) of two given
  // paths
  //  Input:
  //    first_path: the first path
  //    second_path: the second path
  //  Output:
  //    the length of the longest common sub-sequence (the length of the common
  //    path)
  int64_t LCS(const std::vector<SplitNodeVal>& first_path,
              const std::vector<SplitNodeVal>& second_path);

  // Determin whether two paths should be merged give their path lengths and the
  // common sub-path length
  //  Input:
  //    lcs_len: the length of their common sub-path
  //    first_path_len: the length of the first path
  //    second_path_len: the length of the second path
  //  Output:
  //   a bool value indicates whether the two paths should be merged
  bool ShouldMerge(int64_t lcs_len, int64_t first_path_len,
                   int64_t second_path_len);

  // Update available split paths for the while loop of the given split path key
  //  Input:
  //    node_key: the root key of a split path
  //    path_indices: new availabe path indices
  //    input_start_node_to_splittable_paths: the updated
  //    start_node_to_splittable_paths hash table
  //  Output:
  //   a bool value indicates whether the update is successful, the updated
  //   result will be stored in
  //    the given input_start_node_to_splittable_paths parameter
  Status UpdateNewPaths(
      SplitNodeKey node_key, const std::vector<size_t>& path_indices,
      absl::flat_hash_map<SplitNodeKey, std::vector<std::vector<SplitNodeVal>>>&
          input_start_node_to_splittable_paths);

  // Reocrd all instructions in all while_loops
  //  Output:
  //    a status indicates whether the recording is successful
  Status RecordInstructionsInWhileLoop();

  // the pointer to the graph's parent module
  HloModule* parent_module;
  // starting_node -> all possbile best splittable paths of the node (pre-order)
  // splittable_paths doesn't contain start_node itself
  absl::flat_hash_map<SplitNodeKey, std::vector<std::vector<SplitNodeVal>>>
      start_node_to_splittable_paths;
  absl::flat_hash_map<SplitNodeKey, HloInstruction*> start_node_to_start_inst;
  // starting_node -> while_loop_num
  absl::flat_hash_map<SplitNodeKey, int64_t> start_node_to_while_loop_num;
  // a set that sotres all starting_nodes
  absl::flat_hash_set<SplitNodeKey> start_node_set;
  // a vector that sotres all starting_nodes according to post-order traversal
  std::vector<SplitNodeKey> start_node_vector;
  // starting_node -> all decendant starting_nodes
  absl::flat_hash_map<SplitNodeKey, absl::flat_hash_set<SplitNodeKey>>
      start_node_to_decendant_start_nodes;
  // while_loop_num -> all starting_nodes included in the while_loop
  absl::flat_hash_map<int64_t, std::vector<SplitNodeKey>>
      while_loop_num_to_start_node;
  // while_loop_num -> all instructions(including starting_nodes) included in
  // the while_loop
  absl::flat_hash_map<int64_t, absl::flat_hash_set<HloInstruction*>>
      while_loop_num_to_instructions;
  // record how many staring_nodes have been procesed for a while-loop
  absl::flat_hash_map<int64_t, absl::flat_hash_set<int64_t>>
      while_loop_num_to_unmegerable_while_loop_num;
  // absl::flat_hash_map<SplitNodeVal, std::vector<int64_t>>
  //     node_to_while_loop_nums;
  int64_t while_loop_num_counter;
  // The threshold indicates the minimum common path length when merging two
  // while-loops
  float lcs_merge_threshold;
};

// This phase is similar to eXLA-v1 but will use the recorded paths and
// while-loop information to perform splitting and add split nodes to allocated
// while-loops.
// perform real splitting and create while_loops
class TensorSplitterRewriteVisitor : public SplitDeterminer {
 public:
  // Constructor of TensorSplitterRewriteVisitor
  //  Input:
  //    max_size_threshold: a thereshold decides whether an instruction is big
  //      enought to split
  //    target_split_size: the target size of an instruction after splitting
  //    split_size: the split size of the instruction
  //    input_parent_module: the pointer of the parent module
  //    input_start_node_to_splittable_paths: a hashtable from a root node to
  //    all of its splittable paths input_start_node_to_while_loop_num: a
  //    hashtable from a root node to its while_loop_num
  //    input_while_loop_num_to_start_node: a hashtable from a while_loop_num to
  //    all of its splittable root nodes in the while loop
  //    input_while_loop_num_to_instructions: a hashtable from a while_loop_num
  //    to all instructions in the while loop
  //  Output:
  //    a TensorSplitterRewriteVisitor instance
  explicit TensorSplitterRewriteVisitor(
      int64_t max_size_threshold, int64_t target_split_size,
      HloModule* input_parent_module,
      absl::flat_hash_map<SplitNodeKey, std::vector<std::vector<SplitNodeVal>>>&
          input_start_node_to_splittable_paths,
      const absl::flat_hash_map<SplitNodeKey, int64_t>&
          input_start_node_to_while_loop_num,
      absl::flat_hash_map<int64_t, std::vector<SplitNodeKey>>&
          input_while_loop_num_to_start_node,
      absl::flat_hash_map<int64_t, absl::flat_hash_set<HloInstruction*>>&
          input_while_loop_num_to_instructions)
      : SplitDeterminer(max_size_threshold, target_split_size),
        parent_module(input_parent_module),
        start_node_to_splittable_paths(input_start_node_to_splittable_paths),
        start_node_to_while_loop_num(input_start_node_to_while_loop_num),
        while_loop_num_to_start_node(input_while_loop_num_to_start_node),
        while_loop_num_to_instructions(input_while_loop_num_to_instructions) {
    while_loop_num_to_info.clear();
  }
  // Deprecated function in V2
  HloComputation* CreateWhileSplitCondition(const std::string& name,
                                            const Shape& parameters_shape,
                                            int64_t stop_at);

  // Deprecated function in V2
  std::vector<HloInstruction*> CreateWhileSplitWithResults(
      HloComputation* parent_comp, HloComputation* condition,
      HloComputation* body, std::vector<HloInstruction*> parameters,
      std::vector<std::tuple<int64_t, Shape>> ids_and_shapes);

  // Splitting handle function when visiting a dot instruction
  //  Input:
  //    dot: currently visited dot instruction
  //  Output:
  //    a status indicates if the splitting process is successful
  Status HandleDot(HloInstruction* dot) override;

  // Splitting handle function when visiting a reduce instruction
  //  Input:
  //    reduce: currently visited reduce instruction
  //  Output:
  //    a status indicates if the splitting process is successful
  Status HandleReduce(HloInstruction* reduce) override;

  // Splitting handle function when visiting a sort instruction
  //  Input:
  //    sort: currently visited sort instruction
  //  Output:
  //    a status indicates if the splitting process is successful
  Status HandleSort(HloInstruction* sort) override;

  // a tuple<instruction_id, split_dim, split_size> used to identify
  // instructions after splitting
  using VisitedInstructionKey = std::tuple<int, int64_t, int64_t>;

  // a hashtable from a while_loop_num to a hashtable(from VisitedInstructionKey
  // to origianl instruction) to avoid duplicate splitting
  absl::flat_hash_map<
      int64_t, absl::flat_hash_map<VisitedInstructionKey, HloInstruction*>>
      while_loop_num_to_visited_instructions;

  // Represents the output information of a single split path in a while-loop
  class SubOutputInfo {
   public:
    SubOutputInfo() {}
    // Constructor of SubOutputInfo
    //  Input:
    //    output_index: the index of the suboutput in the output of the while
    //    loop node_inst: the instructon of the original instruciton of the
    //    suboutput s_dim: the split dimension ids_and_shapes: only used for
    //    sort instruction, empty vector for other cases sort_leafs: only used
    //    for sort instruction, empty vector for other cases combine_with_sum:
    //    only used for dot instruction, a flag indicating wether to combine
    //    part results by sum combine_with_sum: only used for reduce
    //    instruction, a flag indicating wether to combine part results by
    //    reduce dot_s_dim: only used when splitting one side of a dot
    //    instruction and the split dimension is a non-contracting dimension.
    //      dot_s_dim is the split dimension of the original dot instruction
    //  Output:
    //    a SubOutputInfo instance
    SubOutputInfo(int64_t output_index, HloInstruction* node_inst,
                  int64_t s_dim,
                  std::vector<std::tuple<int64_t, Shape>>& ids_and_shapes,
                  std::vector<HloInstruction*>& sort_leafs,
                  int64_t split_rest_count = 0, int64_t rest_index = -1,
                  bool combine_with_sum = false, bool split_reduce_dim = false,
                  int64_t dot_s_dim = -1)
        : result_index(output_index),
          result_rest_index(rest_index),
          starting_node_inst(node_inst),
          split_dim(s_dim),
          split_rest(split_rest_count),
          combine_parts_with_sum(combine_with_sum),
          split_along_reduce_dim(split_reduce_dim),
          sort_ids_and_shapes(ids_and_shapes),
          sort_split_leafs(sort_leafs),
          dot_split_dim(dot_s_dim) {
      result_rest = nullptr;
    }
    // index of result in the while_loop's output
    int64_t result_index;
    // index of rest part result in the while_loop's rest output
    int64_t result_rest_index;  // * only valid if the split_rest >0
    // the pointer to the original instruction before splitting
    HloInstruction* starting_node_inst;
    // the pointer to the rest part of the sub-result
    HloInstruction* result_rest;

    std::vector<std::tuple<int64_t, Shape>>
        sort_ids_and_shapes;  // * only valid if starting_node_inst is sort
    std::vector<HloInstruction*>
        sort_split_leafs;  // * only valid if starting_node_inst is sort
                           // * and split_rest >0
    // the rest part size of the sub-output
    int64_t split_rest;
    // split dimension of the sub-output
    int64_t split_dim;  // * invalid and useless for dot case2
    // indicate whether part results should be combied by sum
    bool combine_parts_with_sum;  // * only valid if starting_node_inst is dot
                                  // * and split_rest >0
    // the split dimension of a dot instruction
    int64_t
        dot_split_dim;  // * only valid if starting_node_inst is dot case 3
                        // * and split_rest >0 and combine_parts_with_sum=false

    // indicate whether part results should be combied by reduce
    bool split_along_reduce_dim;  // * only valid if starting_node_inst is
                                  // * reduce and split_rest >0
  };

  // Represents the output information of a while-loop
  class WhileLoopInfo {
   public:
    WhileLoopInfo() { final_sub_main_output_elements.resize(0); }
    // Constructor of WhileLoopInfo
    //  Input:
    //    loop_num: the while_loop_num of a while loop
    //    size: split size
    //    count: split count
    //    rest: the rest size after splitting
    //    builder: builder for the while loop
    //    rest_builder: the builder of the rest part if there is a rest part
    //  Output:
    //    a WhileLoopInfo instance
    WhileLoopInfo(int64_t loop_num, int64_t size, int64_t count, int64_t rest,
                  std::unique_ptr<HloComputation::Builder>&& builder,
                  std::unique_ptr<HloComputation::Builder>&& rest_builder)
        : while_loop_num(loop_num),
          split_size(size),
          split_count(count),
          split_rest(rest),
          while_builder(std::move(builder)),
          while_rest_builder(std::move(rest_builder)) {
      final_sub_main_output_elements.resize(0);
      while_loop_parameters.resize(0);
      while_loop_output_elements.resize(0);
      while_rest_output_elements.resize(0);
      while_loop_param = nullptr;
    }

    // Add suboutput to the output of the while loop
    // Input:
    //    sub_output: the sub-output to be added
    void AddSubOutput(SubOutputInfo sub_output) {
      final_sub_main_output_elements.emplace_back(sub_output);
    }
    // final output for the while-loop, including all outputs from all
    // staring_nodes included in the while-loop
    std::vector<SubOutputInfo> final_sub_main_output_elements;
    // builder of the while loop computation
    std::unique_ptr<HloComputation::Builder> while_builder;
    // builder of the rest part computation of the while loop
    std::unique_ptr<HloComputation::Builder> while_rest_builder;
    // single tuple param instruction
    HloInstruction* while_loop_param;
    // single tuple param instruction
    HloInstruction* while_rest_param;
    // tuple parameter elements for the while loop
    std::vector<HloInstruction*> while_loop_parameters;
    // tuple parameter elements for the rest part of the while loop
    std::vector<HloInstruction*> while_rest_parameters;
    // output elements for the while loop
    std::vector<HloInstruction*> while_loop_output_elements;
    // output elements for the rest part of the while loop
    std::vector<HloInstruction*> while_rest_output_elements;
    // used in while-loop
    absl::flat_hash_map<HloInstruction*, HloInstruction*>
        starting_node_inst_to_cloned_inst;
    // used in the rest part
    absl::flat_hash_map<HloInstruction*, HloInstruction*>
        rest_starting_node_inst_to_cloned_inst;
    // identifier number of the while loop
    int64_t while_loop_num;
    // rest part size of the while loop
    int64_t split_rest;
    // split size of the while loop
    int64_t split_size;
    // split count of the while loop
    int64_t split_count;
  };

  // Collect computation for the instruction we want to split
  // and split the parameters. The parameters are returned pre-
  // split such that they can be used verbatim inside a call.
  // The returned instruction is the root instruction of the
  // computation.
  class Splitter {
    HloInstruction* param_;   // single tuple param instruction
    HloInstruction* offset_;  // get offset from tuple param
    std::vector<HloInstruction*>&
        parameters_;  // initialize tuple parameter elements
    // builder used when performing splitting
    HloComputation::Builder& builder_;
    // leaf nodes of the split path
    absl::Span<HloInstruction*> leafs_;
    // a hashtable used to recorde instructions which have already been split
    absl::flat_hash_map<VisitedInstructionKey, HloInstruction*>&
        visited_instructions_;
    // a hastable from a while_loop_num to all instructions involved in the
    // while loop
    absl::flat_hash_map<int64_t, absl::flat_hash_set<HloInstruction*>>&
        while_loop_num_to_instructions_;
    // a flag indicating whether to merge the rest part into a dynamic loop
    bool merged_splitter_flag;
    // the while loop info of the split path
    WhileLoopInfo* while_loop_info;

   public:
    // the output of the while loop of the split path
    std::vector<HloInstruction*>& while_loop_output_elements;
    int64_t param_start_index;
    // Constructor of Splitter
    //  Input:
    //    merged_flag: the flag to indicate wheter to merge the rest part into a
    //    dynamic while loop loop_info: the information of the while loop
    //    containing the split path loop_parameters: the vector of parameters of
    //    the while loop loop_output_elements: the vector of sub-outputs of the
    //    while loop builder: the builder of the while loop parent: the parent
    //    compution leafs: leaf nodes of the split path visited_instructions: a
    //    map from an instruction to the new instruction after splitting, used
    //    to avoid duplicate splitting while_loop_num_to_instructions: a map
    //    from a while_loop_num to all instructions in the while loop offset:
    //    the start offset when splitting
    //  Output:
    //    a Splitter instance
    explicit Splitter(
        bool merged_flag, WhileLoopInfo* loop_info,
        std::vector<HloInstruction*>& loop_parameters,
        std::vector<HloInstruction*>& loop_output_elements,
        HloComputation::Builder& builder, HloComputation* parent,
        absl::Span<HloInstruction*> leafs,
        absl::flat_hash_map<VisitedInstructionKey, HloInstruction*>&
            visited_instructions,
        absl::flat_hash_map<int64_t, absl::flat_hash_set<HloInstruction*>>&
            while_loop_num_to_instructions,
        int64_t offset = 0)
        : merged_splitter_flag(merged_flag),
          parameters_(loop_parameters),
          while_loop_info(loop_info),
          while_loop_output_elements(loop_output_elements),
          builder_(builder),
          leafs_(leafs),
          visited_instructions_(visited_instructions),
          while_loop_num_to_instructions_(while_loop_num_to_instructions) {
      std::stringstream msg;

      msg << "@@@ leafs=[";
      for (auto leaf : leafs_) {
        msg << leaf->name() << ",";
      }
      msg << "]";
      LOG(INFO) << "\n> " << msg.str();
      param_start_index = parameters_.size();
      auto tmp_offset = CREATE_CONSTANT_INT32(offset);
      Shape param_shape = ShapeUtil::MakeTupleShape({tmp_offset->shape()});
      if (merged_splitter_flag) {
        // used for while_loop
        if (param_start_index == 0) {
          // create the parameter for the while_loop
          // Make a param, the shape can be added to over time to get correct
          // shape

          while_loop_info->while_loop_param =
              builder.AddInstruction(HloInstruction::CreateParameter(
                  0, param_shape,
                  "merged_loop_" +
                      std::to_string(while_loop_info->while_loop_num) +
                      "_param"));
          param_ = while_loop_info->while_loop_param;

        } else {
          param_ = while_loop_info->while_loop_param;
        }
      } else {
        // used for rest
        if (param_start_index == 0) {
          // create the parameter for the rest
          // Make a param, the shape can be added to over time to get correct
          // shape
          while_loop_info->while_rest_param =
              builder.AddInstruction(HloInstruction::CreateParameter(
                  0, param_shape,
                  "merged_rest_" +
                      std::to_string(while_loop_info->while_loop_num) +
                      "_param"));
          param_ = while_loop_info->while_rest_param;

        } else {
          param_ = while_loop_info->while_rest_param;
        }
      }

      if (param_start_index == 0) {
        HloInstruction* init_offset =
            parent->AddInstruction(CREATE_CONSTANT_INT32(offset));
        parameters_.push_back(init_offset);
      }
      offset_ = builder.AddInstruction(HloInstruction::CreateGetTupleElement(
          tmp_offset->shape(), param_, 0));
    }

    // Split the inst given its split dimension and split size
    // Input:
    //  inst: the pointer of the instruction to be split
    //  split_dim: split dimension of the instruciton
    //  split_dim: split size of inst on split_dim
    // Output:
    //  the pointer of the new instruction after splitting
    StatusOr<HloInstruction*> SplitInstruction(HloInstruction* inst,
                                               int64_t split_dim,
                                               int64_t split_size);

    // Split the leaf dot instuction given its split dimension and split size
    // Input:
    //  dot: the pointer of the leaf dot instruction
    //  split_dim: split dimension of the dot operation
    //  split_dim: split size of dot on split_dim
    // Output:
    //  the pointer of the new dot instruction after splitting
    StatusOr<HloInstruction*> SplitLeafDot(HloInstruction* dot,
                                           int64_t split_dim,
                                           int64_t split_size);

    // Split the leaf broadcast instuction given its split dimension and split
    // size Input:
    //  broadcast: the pointer of the leaf broadcast instruction
    //  split_dim: split dimension of the broadcast instruction
    //  split_dim: split size of broadcast on split_dim
    // Output:
    //  the pointer of the new broadcast instruction after splitting
    StatusOr<HloInstruction*> SplitLeafBroadcast(HloInstruction* broadcast,
                                                 int64_t split_dim,
                                                 int64_t split_size);

    // Split the leaf parameter instuction given its split dimension and split
    // size Input:
    //  parameter: the pointer of the leaf parameter instruction
    //  split_dim: split dimension of the parameter instruction
    //  split_dim: split size of parameter on split_dim
    // Output:
    //  the pointer of the new parameter instruction after splitting
    StatusOr<HloInstruction*> SplitLeafParameter(HloInstruction* parameter,
                                                 int64_t split_dim,
                                                 int64_t split_size);

    // Split the leaf iota instuction given its split dimension and split size
    // Input:
    //  parameter: the pointer of the iota parameter instruction
    //  split_dim: split dimension of the iota instruction
    //  split_dim: split size of iota on split_dim
    // Output:
    //  the pointer of the new iota instruction after splitting
    StatusOr<HloInstruction*> SplitLeafIota(HloInstruction* iota,
                                            int64_t split_dim,
                                            int64_t split_size);

    // A helper function to generate a VisitedInstructionKey instance
    //  Input:
    //    inst: the target instruction
    //    split_dim: split dimension
    //    split_size: split size
    //  Output:
    //    a VisitedInstructionKey instance
    VisitedInstructionKey MakeVisitedInstructionKey(const HloInstruction* inst,
                                                    int64_t split_dim,
                                                    int64_t split_size) {
      int unique_id = inst->unique_id();
      return std::make_tuple(unique_id, split_dim, split_size);
    }

    // Add the parameter and returnd it's index in the tuple. If get_tuple
    // is passed, it will also create an accessor for the parameter.
    //  Input:
    //    inst: the pointer of the parameter instruction
    //    get_tuple: a instruction pointer used to store the accessor for the
    //    parameter
    //  Output:
    //    the index of the parameter instruction in the input tuple
    int64_t AddParameter(HloInstruction* inst,
                         HloInstruction** get_tuple = nullptr) {
      if (merged_splitter_flag &&
          while_loop_info->starting_node_inst_to_cloned_inst.contains(inst)) {
        // for the  while-loop
        LOG(INFO)
            << "[AddParameter] While-loop Use an in-loop-inst="
            << while_loop_info->starting_node_inst_to_cloned_inst[inst]->name()
            << " to replace in_loop starting_node=" << inst->name();
        *get_tuple = while_loop_info->starting_node_inst_to_cloned_inst[inst];
        // * for now it is safe since all places ussing the return index are to
        // add a newly created
        // * instruction, so cannot get into this if statement.
        return -1;
      } else if (!merged_splitter_flag &&
                 while_loop_info->rest_starting_node_inst_to_cloned_inst
                     .contains(inst)) {
        // for the  rest_part
        // for the  while-loop
        LOG(INFO) << "[AddParameter] RestPart Use an in-remainder-inst="
                  << while_loop_info
                         ->rest_starting_node_inst_to_cloned_inst[inst]
                         ->name()
                  << " to replace in_loop starting_node=" << inst->name();
        *get_tuple =
            while_loop_info->rest_starting_node_inst_to_cloned_inst[inst];
        // * for now it is safe since all places ussing the return index are to
        // add a newly created
        // * instruction, so cannot get into this if statement.
        return -1;
      }
      int64_t idx = parameters_size();
      parameters_.push_back(inst);
      param_->mutable_shape()->mutable_tuple_shapes()->push_back(inst->shape());
      if (get_tuple != nullptr) {
        *get_tuple = builder_.AddInstruction(
            HloInstruction::CreateGetTupleElement(inst->shape(), param_, idx));
      }
      return idx;
    }

    // Generates the rest part output tuple from the given root
    // computation part. Return the result index in the whole_loop_output
    //  Input:
    //    split_dim: the split dimension of the original insturction
    //    split_size: the split size of the original insturction
    //    original: the pointer of the original instruction
    //    part: the pointer of the new part result instruction after splitting
    //    combine_with_sum: a flag which indicates whether part results should
    //    be combied by sum combine_with_reduce: a flag which indicates whether
    //    part results should be combied by reduce operation
    //  Output:
    //    the index of the rest part result in the whole_loop_output
    int64_t BuildRestOutputTuple(int64_t split_dim, int64_t split_size,
                                 HloInstruction* original, HloInstruction* part,
                                 bool combine_with_sum = false,
                                 bool combine_with_reduce = false);

    // Generates the output tuple from the given root
    // computation part. Return the result index in the whole_loop_output
    //  Input:
    //    split_dim: the split dimension of the original insturction
    //    split_size: the split size of the original insturction
    //    original: the pointer of the original instruction
    //    part: the pointer of the new part result instruction after splitting
    //    combine_with_sum: a flag which indicates whether part results should
    //    be combied by sum combine_with_reduce: a flag which indicates whether
    //    part results should be combied by reduce operation
    //  Output:
    //    the index of the result in the whole_loop_output
    int64_t BuildMergedLoopOutput(int64_t split_dim, int64_t split_size,
                                  HloInstruction* original,
                                  HloInstruction* part,
                                  bool combine_with_sum = false,
                                  bool combine_with_reduce = false);

    // Return the parameter size in the while loop
    int64_t parameters_size() { return parameters_.size(); }

    // Return the paramter located in the given index
    //  Input:
    //    idx: the index of the parameter
    //  Output:
    //    the parameter instruction in the given idx
    HloInstruction* parameters(int64_t idx) { return parameters_.at(idx); }

    // Return the tuple paramter
    //  Output:
    //    the tuple parameter
    HloInstruction* tuple_parameters() { return param_; }

    // Return the vector of all parameters
    //  Output:
    //    the vector of all parameters
    std::vector<HloInstruction*>& parameters() { return parameters_; }

    // Return the offset instruction
    //  Output:
    //    offset instruction
    HloInstruction* offset() { return offset_; }
  };

 private:
  // Generate while loop name of the given loop_num
  //  Input:
  //    loop_num: a while loop number
  //  Output:
  //    the name of the while loop with loop_num
  static std::string GenerateBuilderName(int64_t loop_num) {
    return std::string("merged_while_loop" + std::to_string(loop_num));
  }

  // Check whether can finish a merged while-loop, we can build a while loop if
  // all split paths in the loop have been split
  //  Input:
  //    while_loop_num: a while loop number
  //  Output:
  //    a bool value indicating if we can build the while loop computation with
  //    the given while_loop_num
  bool CanFinishMergedWhileLoop(int64_t while_loop_num);

  // Create a merged while-loop computation and build the correspoinding output
  //  Input:
  //    while_loop_num: a while loop number
  //  Output:
  //    a status indicating if the while loop has been built successfully
  Status FinalizeMergedWhileLoop(int64_t while_loop_num);
  // // return (output_inst,init_inst)
  // StatusOr<std::pair<HloInstruction*, HloInstruction*>> BuildSubOutput(
  //     SubOutputInfo& sub_info);

  // Build the final output instruction of the while loop
  //  Input:
  //    loop_info: the information of the while loop
  //    loop: the pointer of the while loop instruction
  //  Output:
  //    a status indicating if the output instruction of the while loop has been
  //    built successfully
  Status BuildFinalOutput(WhileLoopInfo& loop_info, HloInstruction* loop);

  // Add the information of a splittable dot instruction to its while-loop info
  //  Input:
  //    dot: the pointer of the dot instruction
  //    lhs: the pointer of the left hand side instruction of dot
  //    rhs: the pointer of the right hand side instruction of dot
  //  Output:
  //    a status indicating if the dot splitting information has been added to
  //    its while loop inforamtion successfully
  Status AddDotToMergedWhileLoop(HloInstruction* dot, HloInstruction* lhs,
                                 HloInstruction* rhs);

  // Add the information of a splittable reduce instruction to its while-loop
  // info
  //  Input:
  //    reduce: the pointer of the reduce instruction
  //  Output:
  //    a status indicating if the reduce splitting information has been added
  //    to its while loop inforamtion successfully
  Status AddReduceToMergedWhileLoop(HloInstruction* reduce);

  // Add the information of a splittable sort instruction to its while-loop info
  //  Input:
  //    sort: the pointer of the sort instruction
  //  Output:
  //    a status indicating if the sort splitting information has been added to
  //    its while loop inforamtion successfully
  Status AddSortToMergedWhileLoop(HloInstruction* sort);
  // the instruction of the graph's parent module
  HloModule* parent_module;
  // A hashtable from SplitNodeKey to all possible split paths of the key
  // Note: splittable_paths doesn't contain start_node itself
  absl::flat_hash_map<SplitNodeKey, std::vector<std::vector<SplitNodeVal>>>
      start_node_to_splittable_paths;
  // A hashtable from SplitNodeKey to its while loop number
  absl::flat_hash_map<SplitNodeKey, int64_t> start_node_to_while_loop_num;
  // A hashtable from while loop number to all SplitNodeKeys contained in the
  // while loop
  absl::flat_hash_map<int64_t, std::vector<SplitNodeKey>>
      while_loop_num_to_start_node;

  // record how many staring_nodes have been procesed for a while-loop
  absl::flat_hash_map<int64_t, int64_t> while_loop_num_to_processed_count;
  // A hashtable from while loop number to the while loop information of the
  // loop
  absl::flat_hash_map<int64_t, WhileLoopInfo> while_loop_num_to_info;
  // while_loop_num -> all instructions(including starting_nodes) included in
  // the while_loop
  absl::flat_hash_map<int64_t, absl::flat_hash_set<HloInstruction*>>
      while_loop_num_to_instructions;
};

}  // namespace

bool SplitDeterminer::OperandShouldBeSplit(HloInstruction* inst) {
  if (!inst->shape().IsArray()) {
    return false;
  }
  return ShapeUtil::ByteSizeOfElements(inst->shape()) > max_size_threshold;
}

bool SplitDeterminer::OperandCanBeSplit(
    HloInstruction* inst, std::vector<HloInstruction*>* split_leafs,
    std::vector<int64_t>* original_dimensions,
    std::vector<int64_t>* exclude_dimensions) {
  std::stringstream msg;
  msg << "\n<---><---><---><---> orig=[";
  for (auto dim : *original_dimensions) {
    msg << dim << ",";
  }
  msg << "], excl=[";
  for (auto dim : *exclude_dimensions) {
    msg << dim << ",";
  }
  msg << "]";
  msg << "\n> Can be split for '" << inst->name() << "'";
  HloInstruction *next, *lhs, *rhs;
  std::vector<HloInstruction*> next_vec;
  if (Match(inst, m::Dot(m::Op(&lhs), m::Op(&rhs)))) {
    bool do_split_lhs = OperandShouldBeSplit(lhs);
    bool do_split_rhs = OperandShouldBeSplit(rhs);

    msg << ", do_split_lhs=" << do_split_lhs;
    msg << ", do_split_rhs=" << do_split_rhs;

    if (do_split_lhs && do_split_rhs) {
      // We can only split one dimension, so this is impossible
      LOG(INFO) << msg.str();
      return false;
    } else if (do_split_lhs) {
      msg << ", split LHS;";
      LOG(INFO) << msg.str();
      // Exclude all rhs dims from split
      for (int64_t i = lhs->shape().dimensions_size() - 1;
           i < original_dimensions->size(); i++) {
        exclude_dimensions->push_back((*original_dimensions)[i]);
      }
      // exclude batch dimensions
      auto& dnums = inst->dot_dimension_numbers();
      for (int64_t i = 0; i < dnums.lhs_batch_dimensions_size(); ++i) {
        exclude_dimensions->push_back(
            (*original_dimensions)[dnums.lhs_batch_dimensions(i)]);
        LOG(INFO) << "\n ----< [OperandCanBeSplit]  HandleDot for '"
                  << inst->name() << " skip lhs.batch_dimension="
                  << dnums.lhs_batch_dimensions(i) << " orig_dim = "
                  << (*original_dimensions)[dnums.lhs_batch_dimensions(i)];
      }
      // Make a dimensions which is only for the lhs
      std::vector<int64_t> lhs_original_dims;
      int64_t lhs_cdim =
          inst->dot_dimension_numbers().lhs_contracting_dimensions(0);
      for (int64_t i = 0; i < lhs->shape().dimensions_size(); i++) {
        if (i == lhs_cdim) {
          lhs_original_dims.push_back(-1);  // this has no original dim...
        } else if (i < lhs_cdim) {
          lhs_original_dims.push_back((*original_dimensions)[i]);
        } else if (i > lhs_cdim) {
          lhs_original_dims.push_back((*original_dimensions)[i - 1]);
        }
      }
      // Check if can split
      bool can_split = OperandCanBeSplit(lhs, split_leafs, &lhs_original_dims,
                                         exclude_dimensions);
      lhs_original_dims.clear();
      return can_split;  // not tail recursive to keep fresh orig dims
    } else if (do_split_rhs) {
      msg << ", split RHS;";
      LOG(INFO) << msg.str();
      // Exclude all lhs dims from split
      for (int64_t i = 0; i < lhs->shape().dimensions_size() - 1; i++) {
        exclude_dimensions->push_back((*original_dimensions)[i]);
      }
      // exclude batch dimensions
      auto& dnums = inst->dot_dimension_numbers();
      int64_t offset = lhs->shape().dimensions_size() - 1;
      for (int64_t i = 0; i < dnums.rhs_batch_dimensions_size(); ++i) {
        exclude_dimensions->push_back(
            (*original_dimensions)[offset + dnums.rhs_batch_dimensions(i)]);
        LOG(INFO)
            << "\n ----< [OperandCanBeSplit]  HandleDot for '" << inst->name()
            << " skip rhs.batch_dimension=" << dnums.rhs_batch_dimensions(i)
            << " orig_dim = "
            << (*original_dimensions)[offset + dnums.rhs_batch_dimensions(i)];
      }

      // Make a dimensions which is only for the rhs
      std::vector<int64_t> rhs_original_dims;
      int64_t rhs_cdim =
          inst->dot_dimension_numbers().rhs_contracting_dimensions(0);
      int64_t rhs_start = lhs->shape().dimensions_size() - 1;
      for (int64_t i = 0; i < rhs->shape().dimensions_size(); i++) {
        if (i == rhs_cdim) {
          rhs_original_dims.push_back(-1);  // this has no original dim...
        } else if (i < rhs_cdim) {
          rhs_original_dims.push_back((*original_dimensions)[rhs_start + i]);
        } else if (i > rhs_cdim) {
          rhs_original_dims.push_back(
              (*original_dimensions)[rhs_start + i - 1]);
        }
      }
      // Check if can split
      bool can_split = OperandCanBeSplit(rhs, split_leafs, &rhs_original_dims,
                                         exclude_dimensions);
      rhs_original_dims.clear();
      return can_split;  // not tail recursive to keep fresh orig dims
    } else {
      msg << ", dot base case;";
      LOG(INFO) << msg.str();
      // Base case: A Dot produces this large intermediate tensor

      // exclude batch dimensions
      auto& dnums = inst->dot_dimension_numbers();
      for (int64_t i = 0; i < dnums.lhs_batch_dimensions_size(); ++i) {
        exclude_dimensions->push_back(
            (*original_dimensions)[dnums.lhs_batch_dimensions(i)]);
        LOG(INFO) << "\n ----< [OperandCanBeSplit]  HandleDot for '"
                  << inst->name() << " skip lhs.batch_dimension="
                  << dnums.lhs_batch_dimensions(i) << " orig_dim = "
                  << (*original_dimensions)[dnums.lhs_batch_dimensions(i)];
      }
      int64_t offset = lhs->shape().dimensions_size() - 1;
      for (int64_t i = 0; i < dnums.rhs_batch_dimensions_size(); ++i) {
        exclude_dimensions->push_back(
            (*original_dimensions)[offset + dnums.rhs_batch_dimensions(i)]);
        LOG(INFO)
            << "\n ----< [OperandCanBeSplit]  HandleDot for '" << inst->name()
            << " skip rhs.batch_dimension=" << dnums.rhs_batch_dimensions(i)
            << " orig_dim = "
            << (*original_dimensions)[offset + dnums.rhs_batch_dimensions(i)];
      }
      if (split_leafs != nullptr) {
        split_leafs->push_back(inst);
      }
      return true;
    }
  } else if (Match(inst, m::Broadcast(m::Op()))) {
    LOG(INFO) << msg.str();
    // Base case: A broadcast can be split
    if (split_leafs != nullptr) {
      split_leafs->push_back(inst);
    }
    return true;
  } else if (Match(inst, m::Iota())) {
    LOG(INFO) << msg.str();
    // Base case: An Iota can be split!
    if (split_leafs != nullptr) {
      split_leafs->push_back(inst);
    }
    return true;
  } else if (Match(inst, m::Parameter())) {
    LOG(INFO) << msg.str();
    LOG(INFO) << "\n>---<>---<  Exit. Parameter will be split '" << inst->name()
              << "'";
    // TODO(awav)
    if (split_leafs != nullptr) {
      split_leafs->push_back(inst);
    }
    return true;
    // TODO(awav)
    // return false;
  } else if (Match(inst, m::Transpose(m::Op(&next)))) {
    // A transpose changes the dimensions, so we need to
    // update the original_dimensions array.

    // if (original_dimensions != nullptr) {
    //   std::vector<int64_t>
    //   old_original_dimensions(original_dimensions->begin(),
    //                                                original_dimensions->end());
    //   for (int64_t i = 0; i < original_dimensions->size(); i++) {
    //     (*original_dimensions)[i] =
    //         old_original_dimensions[inst->dimensions(i)];
    //   }
    // }
    // return OperandCanBeSplit(next, split_leafs, original_dimensions,
    //                          exclude_dimensions);

    if (original_dimensions == nullptr) {
      LOG(INFO) << msg.str();
      return OperandCanBeSplit(next, split_leafs, original_dimensions,
                               exclude_dimensions);
    }

    msg << ", transpose original dims to [";
    std::vector<int64_t> transposed_dimensions(original_dimensions->begin(),
                                               original_dimensions->end());
    for (int64_t i = 0; i < original_dimensions->size(); i++) {
      transposed_dimensions[i] = original_dimensions->at(inst->dimensions(i));
      msg << transposed_dimensions[i] << ",";
    }
    msg << "]";
    LOG(INFO) << msg.str();
    return OperandCanBeSplit(next, split_leafs, &transposed_dimensions,
                             exclude_dimensions);
  } else if (MatchSupportedNestedReduce(inst)) {
    msg << ", split nested reduce;";
    LOG(INFO) << msg.str();
    return OperandCanBeSplit(inst->mutable_operand(0), split_leafs,
                             original_dimensions, exclude_dimensions);
  } else if (inst->opcode() == HloOpcode::kTriangularSolve) {
    msg << ", split triangular solve;";
    LOG(INFO) << msg.str();
    // We can split a triangular solve on some (but not all)
    // dims
    if (original_dimensions != nullptr && exclude_dimensions != nullptr) {
      if (inst->triangular_solve_options().left_side()) {
        // exclude second to last : Ax = y
        exclude_dimensions->push_back(
            original_dimensions->at(original_dimensions->size() - 2));
      } else {
        // exclude last : xA = y
        exclude_dimensions->push_back(
            original_dimensions->at(original_dimensions->size() - 1));
      }
    }
    // We can't split the matrix for now, so ignore it
    return OperandCanBeSplit(inst->mutable_operand(1), split_leafs,
                             original_dimensions, exclude_dimensions);
  } else if (MatchPointwiseUnary(inst, &next)) {
    msg << ", split pointwise unary;";
    LOG(INFO) << msg.str();
    // This is a special case seperate from nary,
    // since we can make it tail recursive :)
    return OperandCanBeSplit(next, split_leafs, original_dimensions,
                             exclude_dimensions);
  } else if (MatchPointwiseNary(inst, &next_vec)) {
    msg << ", split pointwise nary;";
    LOG(INFO) << msg.str();

    for (HloInstruction* next : next_vec) {
      // this path is not tail recursive :(
      if (!OperandCanBeSplit(next, split_leafs, original_dimensions,
                             exclude_dimensions)) {
        LOG(INFO) << "\n>---<>---<  Exit. Cannot be split 1 for '"
                  << next->name() << "'";
        ;
        return false;
      }
    }
    return true;
  } else {
    LOG(INFO) << msg.str();
    LOG(INFO) << "\n>---<>---<  Exit. Cannot be split 0 for '" << inst->name()
              << "'";
    return false;
  }
}

bool SplitDeterminer::MatchPointwiseUnary(HloInstruction* inst,
                                          HloInstruction** operand) {
  if (inst->IsElementwise() && !inst->HasSideEffect() &&
      inst->operand_count() == 1) {
    if (operand != nullptr) {
      *operand = inst->mutable_operand(0);
    }
    return true;
  } else {
    return false;
  }
}

bool SplitDeterminer::MatchPointwiseNary(
    HloInstruction* inst, std::vector<HloInstruction*>* operands) {
  if (inst->IsElementwise() && !inst->HasSideEffect() &&
      inst->operand_count() > 0) {
    if (operands != nullptr) {
      for (int64_t i = 0; i < inst->operand_count(); i++)
        operands->push_back(inst->mutable_operand(i));
    }
    return true;
  } else {
    return false;
  }
}

bool SplitDeterminer::MatchSupportedReduce(HloInstruction* inst) {
  if (inst->opcode() == HloOpcode::kReduce) {
    int64_t opt_count = inst->operand_count() / 2;
    if (opt_count < 1) return false;

    for (int64_t i = 1; i < opt_count; i++)
      if (!ShapeUtil::EqualIgnoringElementType(inst->operand(0)->shape(),
                                               inst->operand(i)->shape()))
        return false;

    for (int64_t i = 0; i < opt_count; i++)
      if (!ShapeUtil::IsScalar(inst->operand(opt_count + i)->shape()))
        return false;

    return true;
  } else {
    return false;
  }
}

bool SplitDeterminer::MatchSupportedNestedReduce(HloInstruction* inst) {
  return MatchSupportedReduce(inst) && inst->operand_count() == 2;
}

std::vector<int64_t> SplitDeterminer::BestSplitDim(
    HloInstruction* inst, absl::Span<const int64_t> excluded) {
  const Shape& shape = inst->shape();
  std::vector<int64_t> best_dims;
  // (dim,best_even_split_size)
  std::vector<std::pair<int64_t, int64_t>> split_dims;
  int64_t best_dim = -1,
          best_split = 0;  // ShapeUtil::ElementsIn(inst->shape());
  for (int64_t i = 0; i < shape.dimensions_size(); i++) {
    if (absl::c_linear_search(excluded, i)) {
      continue;
    }
    int64_t split = BestEvenSplitSize(inst, i);
    Shape new_shape = ShapeUtil::MakeShape(inst->shape().element_type(),
                                           inst->shape().dimensions());
    new_shape.set_dimensions(i, split);
    if (ShapeUtil::ByteSizeOfElements(new_shape) > max_size_threshold) {
      // cannot select slilt which makes inst.size after splitting still greater
      // than max_size_threshold
      continue;
    }
    split_dims.push_back(std::make_pair(i, split));
    if (split == -1 || split <= best_split) {
      continue;
    }
    best_split = split;
    best_dim = i;
  }
  if (best_dim == -1) {
    best_dims.push_back(best_dim);
  } else {
    // record all best_dims
    for (auto p : split_dims) {
      if (p.second == best_split) {
        best_dims.push_back(p.first);
      }
    }
  }
  std::stringstream ss;

  ss << "[BestSplitDim] "
     << "inst.name=" << inst->name() << " best_split=" << best_split
     << " best_dims.size=" << best_dims.size() << " best_dims=(";
  for (auto i = 0; i < best_dims.size(); ++i) {
    if (i + 1 == best_dims.size()) {
      ss << best_dims[i] << ")";
    } else {
      ss << best_dims[i] << ", ";
    }
  }
  LOG(INFO) << ss.str();
  return best_dims;
}

int64_t SplitDeterminer::BestEvenSplitSizeFold(int64_t (&factors)[PRIME_SIZE],
                                               int offset, int64_t current,
                                               int64_t best, int64_t size,
                                               int64_t max_size) {
  if (offset >= PRIME_SIZE) {
    return best;
  } else {
    if (factors[offset] > 0) {
      int64_t current_prime = prime_numbers[offset] * current;
      if (size / current_prime <= max_size && current_prime < best) {
        best = current_prime;
      }
      factors[offset]--;
      best = BestEvenSplitSizeFold(factors, offset, current_prime, best, size,
                                   max_size);
      factors[offset]++;
    }
    return BestEvenSplitSizeFold(factors, offset + 1, current, best, size,
                                 max_size);
  }
}

int64_t SplitDeterminer::BestEvenSplitSize(HloInstruction* inst,
                                           int64_t split_dim) {
  // find list of prime factors
  int64_t factors[PRIME_SIZE];
  int64_t tmp_size = inst->shape().dimensions(split_dim);
  for (int i = 0; i < PRIME_SIZE; i++) {
    factors[i] = 0;
    while (tmp_size % prime_numbers[i] == 0) {
      factors[i]++;
      tmp_size /= prime_numbers[i];
    }
  }

  int64_t size = inst->shape().dimensions(split_dim);
  int64_t full_size_bytes =
      ShapeUtil::ByteSizeOfPrimitiveType(inst->shape().element_type()) *
      ShapeUtil::ElementsIn(inst->shape());
  int64_t max_size = target_split_size * size / full_size_bytes;
  int64_t factor = BestEvenSplitSizeFold(factors, 0, 1, size, size, max_size);
  return size / factor;
}

TensorSplitProperties SplitDeterminer::DetermineSplitSize(HloInstruction* inst,
                                                          int64_t split_dim) {
  const Shape& inst_shape = inst->shape();
  int64_t best_even_split = BestEvenSplitSize(inst, split_dim);
  int64_t split_dim_size = inst_shape.dimensions(split_dim);
  int64_t primitive_type_size =
      ShapeUtil::ByteSizeOfPrimitiveType(inst_shape.element_type());
  int64_t max_elements = target_split_size / primitive_type_size;
  int64_t max_dim_size =
      max_elements * split_dim_size / ShapeUtil::ElementsIn(inst_shape);
  LOG(INFO) << "[DetermineSplitSize] "
            << "inst.name=" << inst->name() << " split_dim=" << split_dim
            << " split_dim_size=" << split_dim_size
            << " target_split_size=" << target_split_size
            << " primitive_type_size=" << primitive_type_size
            << " max_elements=" << max_elements
            << " max_dim_size=" << max_dim_size
            << " max_dim_size*8/10=" << max_dim_size * 8 / 10
            << " best_even_split=" << best_even_split;
  int64_t size, count, rest;
  if (best_even_split >= max_dim_size * 8 / 10) {
    // even split is prefered
    size = best_even_split;
    count = split_dim_size / size;
    rest = 0;
    LOG(INFO) << "[DetermineSplitSize] "
              << "inst.name=" << inst->name() << " split_dim=" << split_dim
              << " split_dim_size=" << split_dim_size
              << " even split is prefered, split_size=" << size
              << " split_count=" << count << " split_rest=" << rest;
  } else {
    // uneven split is prefered
    size = max_dim_size;
    count = split_dim_size / size;
    rest = split_dim_size - size * count;
    LOG(INFO) << "[DetermineSplitSize] "
              << "inst.name=" << inst->name() << " split_dim=" << split_dim
              << " split_dim_size=" << split_dim_size
              << " uneven split is prefered, split_size=" << size
              << " split_count=" << count << " split_rest=" << rest;
  }
  return std::make_tuple(size, count, rest);
}

int64_t SplittablePathRecorder::InferDotResultSplitDim(
    HloInstruction* dot, int64_t operand_split_diim, bool split_is_lhs) {
  int64_t result_split_dim = operand_split_diim;
  HloInstruction *lhs, *rhs;
  CHECK(Match(dot, m::Dot(m::Op(&lhs), m::Op(&rhs))));
  auto& dnums = dot->dot_dimension_numbers();
  int64_t dims_lhs =
      lhs->shape().rank() - dnums.lhs_contracting_dimensions_size();

  if (split_is_lhs) {
    for (int64_t i = 0; i < dnums.lhs_contracting_dimensions_size(); i++) {
      if (result_split_dim >= dnums.lhs_contracting_dimensions(i))
        result_split_dim -= 1;
    }
  } else {
    result_split_dim += dims_lhs;
    for (int64_t i = 0; i < dnums.rhs_contracting_dimensions_size(); i++) {
      if (result_split_dim >= dnums.rhs_contracting_dimensions(i))
        result_split_dim -= 1;
    }
  }

  return result_split_dim;
}
int64_t SplittablePathRecorder::InferReduceResultSplitDim(
    HloInstruction* reduce, int64_t operand_split_diim) {
  int64_t result_split_dim = operand_split_diim;  // split dim after reduce
  for (int64_t r_dim : reduce->dimensions()) {
    if (r_dim < operand_split_diim) {
      result_split_dim--;
    }
  }
  return result_split_dim;
}
Status SplittablePathRecorder::CreateNewEmptyPath(SplitNodeKey key,
                                                  HloInstruction* inst) {
  // It's not possible to have different paths which have the same starting inst
  // and same splitting info but are different paths. So we could use
  // (start_inst, split_dim, split_size) to distinguish different paths
  if (start_node_to_splittable_paths.contains(key) &&
      start_node_to_start_inst.contains(key)) {
    LOG(INFO) << "[SplittablePathRecorder::CreateNewEmptyPath] "
              << " start_node_key=" << key << " exists, current_paths.size="
              << start_node_to_splittable_paths[key].size()
              << "Add new_path to existing starting_node: " << inst->name()
              << " start_node_to_start_inst[" << key
              << "]=" << start_node_to_start_inst[key];
  } else {
    // find a new starting_node, insert an empty path
    std::vector<std::vector<SplitNodeVal>> tmp(0);
    start_node_to_splittable_paths[key] = tmp;
    start_node_to_start_inst[key] = inst;
  }
  std::vector<SplitNodeVal> tmp(0);
  start_node_to_splittable_paths[key].push_back(tmp);
  // find a new starting_node, insert an empty path
  LOG(INFO) << "[SplittablePathRecorder::CreateNewEmptyPath] "
            << "Finish, Starting_node: " << inst->name()
            << " paths.size=" << start_node_to_splittable_paths[key].size();
  return OkStatus();
}

Status SplittablePathRecorder::AppendToPath(
    SplitNodeKey path_key, int64_t path_index, HloInstruction* inst,
    int64_t split_dim, int64_t split_size, HloInstruction* parent_inst) {
  if (!start_node_to_splittable_paths.contains(path_key)) {
    std::stringstream msg;
    msg << "[SplittablePathRecorder::AppendToPath] "
        << "Path StartingNode: <" << start_node_to_start_inst[path_key]->name()
        << "> Doesn't exist ";
    LOG(ERROR) << msg.str();
    CHECK(false);
  }
  if (start_node_to_splittable_paths[path_key].size() <= path_index) {
    // error usage
    LOG(ERROR) << "[SplittablePathRecorder::AppendToPath] "
               << "path_index is too large, error usage";
    CHECK(false);
  }
  auto node_val = MakeSplitNodeVal(inst, split_dim, split_size, parent_inst);
  start_node_to_splittable_paths[path_key][path_index].emplace_back(node_val);

  return OkStatus();
}
Status SplittablePathRecorder::FinishRecordLeafDot(SplitNodeKey path_key,
                                                   int64_t path_index,
                                                   HloInstruction* dot,
                                                   int64_t split_dim,
                                                   int64_t split_size) {
  std::string prefix = "[FinishRecordLeafDot] ";
  HloInstruction *lhs, *rhs;
  CHECK(Match(dot, m::Dot(m::Op(&lhs), m::Op(&rhs))));
  auto& dnums = dot->dot_dimension_numbers();
  int64_t dims_lhs =
      lhs->shape().rank() - dnums.lhs_contracting_dimensions_size();

  HloInstruction* split_op;
  bool split_is_lhs;
  if (split_dim < dims_lhs) {
    // We are splitting up the lhs
    split_is_lhs = true;
    split_op = lhs;
    for (int64_t i = 0; i < dnums.lhs_contracting_dimensions_size(); i++) {
      if (split_dim >= dnums.lhs_contracting_dimensions(i)) split_dim += 1;
    }
  } else {
    // We are splitting up the rhs
    split_is_lhs = false;
    split_dim -= dims_lhs;
    split_op = rhs;
    for (int64_t i = 0; i < dnums.rhs_contracting_dimensions_size(); i++) {
      if (split_dim >= dnums.rhs_contracting_dimensions(i)) split_dim += 1;
    }
  }
  bool is_start_node = false;
  SplitNodeKey split_op_key = MakeSplitNodeKey(split_op);
  for (const auto& kv : start_node_to_start_inst) {
    if (split_op_key == kv.first) {
      is_start_node = true;
      break;
    }
  }
  if (is_start_node) {
    LOG(INFO) << prefix << " path_key=" << path_key
              << " path_index=" << path_index
              << " Add extra inst.name=" << split_op->name()
              << " split_dim=" << split_dim << " split_size=" << split_size
              << " parent_inst.name=" << dot->name();
    AppendToPath(path_key, path_index, split_op, split_dim, split_size, dot);
  }
  return OkStatus();
}
Status SplittablePathRecorder::FinishRecordLeafBroadcast(
    SplitNodeKey path_key, int64_t path_index, HloInstruction* broadcast,
    int64_t split_dim, int64_t split_size) {
  HloInstruction* operand;
  CHECK(Match(broadcast, m::Broadcast(m::Op(&operand))));
  std::string prefix = "[FinishRecordLeafBroadcast] ";
  // For a broadcast, we identify if we have to
  // create slices of the underlying operand tensor.

  bool split_on_original_dim =
      absl::c_linear_search(broadcast->dimensions(), split_dim);

  if (split_on_original_dim) {
    // we need to slice the parameter ...
    int64_t operand_split_dim;
    for (int64_t i = 0; i < broadcast->dimensions().size(); i++) {
      if (broadcast->dimensions(i) == split_dim) {
        operand_split_dim = i;
        break;
      }
    }
    bool is_start_node = false;
    SplitNodeKey operand_key = MakeSplitNodeKey(operand);
    for (const auto& kv : start_node_to_start_inst) {
      if (operand_key == kv.first) {
        is_start_node = true;
        break;
      }
    }
    if (is_start_node) {
      LOG(INFO) << prefix << " path_key=" << path_key
                << " path_index=" << path_index
                << " Add extra inst.name=" << operand->name()
                << " split_dim=" << operand_split_dim
                << " split_size=" << split_size
                << " parent_inst.name=" << broadcast->name();
      AppendToPath(path_key, path_index, operand, operand_split_dim, split_size,
                   broadcast);
    }
  }
  return OkStatus();
}
Status SplittablePathRecorder::RecordPath(
    SplitNodeKey path_key, int64_t path_index, HloInstruction* inst,
    int64_t split_dim, int64_t split_size,
    std::vector<HloInstruction*>& split_leafs, HloInstruction* parent_inst) {
  std::string prefix = "[SplittablePathRecorder::RecordPath]";
  LOG(INFO) << prefix << " path_key=" << path_key
            << " path_index=" << path_index << " inst.name=" << inst->name()
            << " split_dim=" << split_dim << " split_size=" << split_size
            << " parent_inst.name=" << parent_inst->name();
  // record current node
  AppendToPath(path_key, path_index, inst, split_dim, split_size, parent_inst);
  if (absl::c_linear_search(split_leafs, inst)) {
    LOG(INFO) << prefix << " Reach a leaf: '" << inst->name();
    if (Match(inst, m::Dot())) {
      TF_RETURN_IF_ERROR(FinishRecordLeafDot(path_key, path_index, inst,
                                             split_dim, split_size));
    } else if (Match(inst, m::Broadcast())) {
      TF_RETURN_IF_ERROR(FinishRecordLeafBroadcast(path_key, path_index, inst,
                                                   split_dim, split_size));
    }
  } else {
    HloInstruction *operand, *lhs, *rhs;
    std::vector<HloInstruction*> operands;

    if (Match(inst, m::Transpose(m::Op(&operand)))) {
      // For a transpose, the transpose might change which dimension is
      // being split. So we obtain the new split dimension and then
      // recursively a new operand to make a clone.
      int64_t operand_split_dim = inst->dimensions(split_dim);
      LOG(INFO) << prefix << " Record 'Transpose:Op' instruction '"
                << inst->name() << "'"
                << " split_dim=" << split_dim
                << " operand_split_dim=" << operand_split_dim;
      TF_RETURN_IF_ERROR(RecordPath(path_key, path_index, operand,
                                    operand_split_dim, split_size, split_leafs,
                                    inst));

    } else if (MatchSupportedNestedReduce(inst)) {
      // For a reduce, split the 0th and only operand
      // (the initializer a scalar, so all we need to do
      // is update the shape and clone the operand with new
      // inputs)

      LOG(INFO) << prefix << " Record 'NestedReduce' instruction '"
                << inst->name() << "'";
      int64_t operand_split_dim = split_dim;  // split dim in operand
      if (inst->dimensions(0) <= split_dim) {
        operand_split_dim += 1;
      }
      TF_RETURN_IF_ERROR(RecordPath(path_key, path_index,
                                    inst->mutable_operand(0), operand_split_dim,
                                    split_size, split_leafs, inst));
    } else if (inst->opcode() == HloOpcode::kTriangularSolve) {
      LOG(INFO) << prefix << " Record 'TriangularSolve' instruction '"
                << inst->name() << "'";
      TF_RETURN_IF_ERROR(RecordPath(path_key, path_index,
                                    inst->mutable_operand(1), split_dim,
                                    split_size, split_leafs, inst));
    } else if (Match(inst, m::Dot(m::Op(&lhs), m::Op(&rhs)))) {
      std::stringstream msg;
      msg << prefix << " Record 'Dot(Op, Op)' instruction '" << inst->name()
          << "'";
      // For an intermediate dot, split the correct operand and assemble
      // a new dot.
      bool split_lhs = ShapeUtil::ElementsIn(lhs->shape()) >
                       ShapeUtil::ElementsIn(
                           rhs->shape());  // this works, since only one of the
                                           // operands needs to be split
      msg << ", split_dim=" << split_dim << ", split_size=" << split_size
          << ", is_lhs=" << (split_lhs ? "yes" : "no");
      LOG(INFO) << msg.str();

      if (split_lhs) {
        CHECK(split_dim < lhs->shape().dimensions_size() - 1);
        int64_t lhs_contr_dim =
            inst->dot_dimension_numbers().lhs_contracting_dimensions(0);
        int64_t lhs_split_dim =
            split_dim >= lhs_contr_dim ? split_dim + 1 : split_dim;

        TF_RETURN_IF_ERROR(RecordPath(path_key, path_index, lhs, lhs_split_dim,
                                      split_size, split_leafs, inst));
      } else {
        int64_t rhs_start = lhs->shape().dimensions_size() - 1;
        CHECK(split_dim >= rhs_start);
        int64_t rhs_contr_dim =
            inst->dot_dimension_numbers().rhs_contracting_dimensions(0);
        int64_t rhs_split_dim = split_dim - rhs_start >= rhs_contr_dim
                                    ? split_dim + 1 - rhs_start
                                    : split_dim - rhs_start;

        TF_RETURN_IF_ERROR(RecordPath(path_key, path_index, rhs, rhs_split_dim,
                                      split_size, split_leafs, inst));
      }
    } else if (MatchPointwiseNary(inst, &operands)) {
      // For a pointwise operation recursively obtain the new operands and
      // clone the operation.
      LOG(INFO) << prefix << " Record 'PointwiseNary' instruction '"
                << inst->name() << "'";
      for (HloInstruction* operand : operands) {
        TF_RETURN_IF_ERROR(RecordPath(path_key, path_index, operand, split_dim,
                                      split_size, split_leafs, inst));
      }
    } else {
      // Invariant violation
      // TODO: Is there a more idiomatic way to return a bad status?
      LOG(ERROR) << prefix << "Trying to split invalid operation '"
                 << inst->name() << "'";
      CHECK(false);
    }
  }
  return OkStatus();
}

Status SplittablePathRecorder::HandleDot(HloInstruction* dot) {
  HloInstruction *lhs, *rhs;
  CHECK(Match(dot, m::Dot(m::Op(&lhs), m::Op(&rhs))));
  auto& dnums = dot->dot_dimension_numbers();

  std::stringstream ss;

  LOG(INFO) << "\n ---------------------------"
            << "\n ----> [SplittablePathRecorder] Enter HandleDot for '"
            << dot->name() << "'";

  if (OperandShouldBeSplit(dot)) {
    ss << "\n ----< [SplittablePathRecorder] Exit HandleDot for '"
       << dot->name() << "' FAILURE";
    LOG(INFO) << ss.str();
    return OkStatus();
  }
  // TODO: Handle the case where both operands can be
  //       split in a better way.

  // Cases we handle:
  // 1. lhs can split
  // 2. rhs can split
  // 3. lhs = rhs + can split
  // 3.1 lhs + rhs can split on respective contracted dim

  bool can_split = false;
  bool rhs_is_lhs = lhs == rhs;

  std::vector<HloInstruction*> split_leafs_lhs;
  std::vector<HloInstruction*> split_leafs_rhs;
  std::vector<int64_t> exclude_dims_lhs;
  std::vector<int64_t> exclude_dims_rhs;

  auto lhs_dim_size = lhs->shape().dimensions_size();
  std::vector<int64_t> original_dims_lhs(lhs_dim_size);
  std::iota(original_dims_lhs.begin(), original_dims_lhs.end(), 0);

  bool can_split_lhs = OperandShouldBeSplit(lhs) &&
                       OperandCanBeSplit(lhs, &split_leafs_lhs,
                                         &original_dims_lhs, &exclude_dims_lhs);

  auto rhs_dim_size = rhs->shape().dimensions_size();
  std::vector<int64_t> original_dims_rhs(rhs_dim_size);
  std::iota(original_dims_rhs.begin(), original_dims_rhs.end(), 0);

  bool can_split_rhs = OperandShouldBeSplit(rhs) &&
                       OperandCanBeSplit(rhs, &split_leafs_rhs,
                                         &original_dims_rhs, &exclude_dims_rhs);

  auto path_key = MakeSplitNodeKey(dot);
  if (can_split_lhs && can_split_rhs && rhs_is_lhs) {
    //
    // Case :: Self dot
    //
    CHECK(dnums.lhs_contracting_dimensions().size() == 1);
    if (dnums.lhs_contracting_dimensions()[0] !=
        dnums.rhs_contracting_dimensions()[0]) {
      return OkStatus();
    }

    int64_t split_dim = dnums.lhs_contracting_dimensions()[0];
    if (absl::c_linear_search(exclude_dims_lhs, split_dim)) {
      LOG(WARNING) << "Failed to split self dot '" << dot->name()
                   << "' as contracted dimension is excluded.";
      ss << "\n ----< Exit HandleDot for '" << dot->name() << "' FAILURE";
      LOG(INFO) << ss.str();
      return OkStatus();
    }

    int64_t split_size, split_count, split_rest;
    std::tie(split_size, split_count, split_rest) =
        DetermineSplitSize(lhs, split_dim);

    auto main_split_size = split_count * split_size;
    CHECK(main_split_size + split_rest == lhs->shape().dimensions(split_dim));

    ss << "<<< "
       << "Splitting self dot " << dot->name()
       << " operand will be split at dimension " << split_dim
       << " with split size " << split_size << " and rest size " << split_rest;

    TF_RETURN_IF_ERROR(CreateNewEmptyPath(path_key, dot));
    start_node_vector.push_back(path_key);
    // split on contracting dim so record dot's split_dim = -1
    AppendToPath(path_key, start_node_to_splittable_paths[path_key].size() - 1,
                 dot, -1, -1, nullptr);
    // record the newly created empty path
    TF_RETURN_IF_ERROR(RecordPath(
        path_key, start_node_to_splittable_paths[path_key].size() - 1, lhs,
        split_dim, split_size, split_leafs_lhs, dot));

  } else if (!rhs_is_lhs && can_split_lhs && can_split_rhs) {
    //
    // CASE :: both lhs and rhs need split
    //
    CHECK(dnums.lhs_contracting_dimensions().size() == 1);

    int64_t split_dim_lhs = dnums.lhs_contracting_dimensions()[0];
    if (absl::c_linear_search(exclude_dims_lhs, split_dim_lhs)) {
      LOG(WARNING) << "Failed to split both sides of dot '" << dot->name()
                   << "' as LHS contracted dimension is excluded.";
      ss << "\n ----< Exit HandleDot for '" << dot->name() << "' FAILURE";
      return OkStatus();
    }
    int64_t split_dim_rhs = dnums.rhs_contracting_dimensions()[0];
    if (absl::c_linear_search(exclude_dims_rhs, split_dim_rhs)) {
      LOG(WARNING) << "Failed to split both sides of dot '" << dot->name()
                   << "' as RHS contracted dimension is excluded.";
      ss << "\n ----< Exit HandleDot for '" << dot->name() << "' FAILURE";
      LOG(INFO) << ss.str();
      return OkStatus();
    }

    CHECK(lhs->shape().dimensions(split_dim_lhs) ==
          rhs->shape().dimensions(split_dim_rhs));

    int64_t split_size, split_count, split_rest;
    std::tie(split_size, split_count, split_rest) =
        DetermineSplitSize(lhs, split_dim_lhs);

    auto main_split_size = split_count * split_size;
    CHECK(main_split_size + split_rest ==
          lhs->shape().dimensions(split_dim_lhs));

    ss << "<<< "
       << "Splitting dot " << dot->name()
       << " lhs and rhs will be split on contracted dimension with split size "
       << split_size << " and split rest " << split_rest;

    TF_RETURN_IF_ERROR(CreateNewEmptyPath(path_key, dot));
    start_node_vector.push_back(path_key);
    // record the newly created empty path
    // split on contracting dim so record dot's split_dim = -1
    AppendToPath(path_key, start_node_to_splittable_paths[path_key].size() - 1,
                 dot, -1, -1, nullptr);
    ss << "\n> Record LHS Split Path'" << lhs->name() << "'";
    TF_RETURN_IF_ERROR(RecordPath(
        path_key, start_node_to_splittable_paths[path_key].size() - 1, lhs,
        split_dim_lhs, split_size, split_leafs_lhs, dot));

    ss << "\n> Record RHS Split Path '" << rhs->name() << "'";
    TF_RETURN_IF_ERROR(RecordPath(
        path_key, start_node_to_splittable_paths[path_key].size() - 1, rhs,
        split_dim_rhs, split_size, split_leafs_rhs, dot));

  } else if ((can_split_lhs && !can_split_rhs) ||
             (!can_split_lhs && can_split_rhs)) {
    //
    // CASE :: one of lhs / rhs is split
    //
    bool split_is_lhs = can_split_lhs;
    HloInstruction* split_inst = split_is_lhs ? lhs : rhs;
    for (int64_t i = 0; i < dnums.lhs_batch_dimensions_size(); ++i) {
      exclude_dims_lhs.push_back(dnums.lhs_batch_dimensions(i));
      exclude_dims_rhs.push_back(dnums.rhs_batch_dimensions(i));
      LOG(INFO) << "\n ----< [SplittablePathRecorder]  HandleDot for '"
                << dot->name()
                << " skip lhs.batch_dimension=" << dnums.lhs_batch_dimensions(i)
                << " skip rhs.batch_dimension="
                << dnums.rhs_batch_dimensions(i);
    }
    std::vector<int64_t> split_dims = BestSplitDim(
        split_inst,
        absl::MakeSpan(split_is_lhs ? exclude_dims_lhs : exclude_dims_rhs));
    if (split_dims[0] == -1) {
      // Bail, we can't split this tensor into equally sized parts.
      ss << "\n ----< Exit HandleDot for '" << dot->name() << "' FAILURE";
      LOG(INFO) << ss.str();
      return OkStatus();
    }

    // record all splittable paths
    start_node_vector.push_back(path_key);
    for (auto split_dim : split_dims) {
      bool combine_parts_with_sum = absl::c_linear_search(
          split_is_lhs ? dnums.lhs_contracting_dimensions()
                       : dnums.rhs_contracting_dimensions(),
          split_dim);

      int64_t split_size, split_count, split_rest;
      std::tie(split_size, split_count, split_rest) =
          DetermineSplitSize(split_inst, split_dim);

      auto main_split_size = split_count * split_size;
      CHECK(main_split_size + split_rest ==
            split_inst->shape().dimensions(split_dim));

      ss << "\n <<< " << start_node_to_splittable_paths[path_key].size()
         << "th Recording splitting dot '" << dot->name() << "' "
         << (split_is_lhs ? "lhs" : "rhs") << " will be split on " << split_dim
         << " with split size " << split_size << " and rest size "
         << split_rest;
      TF_RETURN_IF_ERROR(CreateNewEmptyPath(path_key, dot));
      if (combine_parts_with_sum) {
        // split on contracting dim so record dot's split_dim = -1
        AppendToPath(path_key,
                     start_node_to_splittable_paths[path_key].size() - 1, dot,
                     -1, -1, nullptr);
      } else {
        int64_t dot_result_split_dim =
            InferDotResultSplitDim(dot, split_dim, split_is_lhs);
        AppendToPath(path_key,
                     start_node_to_splittable_paths[path_key].size() - 1, dot,
                     dot_result_split_dim, split_size, nullptr);
      }
      TF_RETURN_IF_ERROR(RecordPath(
          path_key, start_node_to_splittable_paths[path_key].size() - 1,
          split_inst, split_dim, split_size,
          split_is_lhs ? split_leafs_lhs : split_leafs_rhs, dot));
    }
  }

  ss << "\n ----< Exit HandleDot for '" << dot->name();
  LOG(INFO) << ss.str();
  return OkStatus();
}

Status SplittablePathRecorder::HandleReduce(HloInstruction* reduce) {
  if (!MatchSupportedReduce(reduce)) {
    return OkStatus();
  }

  std::stringstream ss;

  LOG(INFO) << "\n =============================="
            << "\n ----> Enter HandleReduce for '" << reduce->name() << "'";

  // MatchSupportedReduce enforces that all inputs are of the
  // same shape, and that there is at least one operand!
  if (!OperandShouldBeSplit(reduce->mutable_operand(0))) {
    ss << "\n<<< Reduce '" << reduce->name()
       << "' cannot be split. Something is not splittable on the way up.";
    ss << "\n <---- Exit HandleReduce for '" << reduce->name() << "' FAILURE";
    LOG(INFO) << ss.str();
    return OkStatus();
  }

  // TODO: This is a hack, I need to more seriously rethink the
  //       two pass system, to mark elements in a first pass and combine
  //       sections properly ...
  if (OperandShouldBeSplit(reduce)) {
    ss << "\n<<< Looks like reduce '" << reduce->name()
       << "' cannot be split after all";
    ss << "\n <---- Exit HandleReduce for '" << reduce->name() << "' FAILURE";
    LOG(INFO) << ss.str();
    return OkStatus();
  }

  // If this is a multi-argument reduce, check if only one
  // result is used.
  if (reduce->shape().IsTuple() && reduce->user_count() > 1) {
    ss << "\n<<< Nah, looks like reduce '" << reduce->name()
       << "' cannot be split after all";
    ss << "\n <---- Exit HandleReduce for '" << reduce->name() << "' FAILURE";
    LOG(INFO) << ss.str();
    return OkStatus();
  }

  // MatchSupportedReduce enforces that all initializers are
  // scalars, so we only need to split the operands to the
  // reduce itself.
  int64_t op_count = reduce->operand_count() / 2;
  std::vector<HloInstruction*> split_leafs;
  std::vector<int64_t> orig_dims;
  std::vector<int64_t> exclude_dims;
  for (int64_t i = 0; i < op_count; i++) {
    orig_dims.clear();
    for (int64_t j = 0; j < reduce->operand(i)->shape().dimensions_size();
         j++) {
      orig_dims.push_back(j);
    }

    if (!OperandCanBeSplit(reduce->mutable_operand(i), &split_leafs, &orig_dims,
                           &exclude_dims)) {
      ss << "\n<<< Again, looks like reduce '" << reduce->name()
         << "' cannot be split because of '"
         << reduce->mutable_operand(i)->name() << "'";
      ss << "\n <---- Exit HandleReduce for '" << reduce->name() << "' FAILURE";
      LOG(INFO) << ss.str();
      return OkStatus();
    }
  }

  if (reduce->shape().IsTuple()) {
    for (int64_t reduce_dim : reduce->dimensions()) {
      exclude_dims.push_back(reduce_dim);
    }
  }

  std::vector<int64_t> split_dims =
      BestSplitDim(reduce->mutable_operand(0), absl::MakeSpan(exclude_dims));
  if (split_dims[0] == -1) {
    // Bail, we can't split this tensor into equally sized parts.
    ss << "\n<<< Looks like reduce '" << reduce->name()
       << "' cannot be split into equally sized parts";
    ss << "\n <---- Exit HandleReduce for '" << reduce->name() << "' FAILURE";
    LOG(INFO) << ss.str();
    return OkStatus();
  }

  auto path_key = MakeSplitNodeKey(reduce);
  start_node_vector.push_back(path_key);

  for (auto split_dim : split_dims) {
    TF_RETURN_IF_ERROR(CreateNewEmptyPath(path_key, reduce));

    bool split_along_reduce_dim =
        absl::c_linear_search(reduce->dimensions(), split_dim);

    int64_t split_size, split_count, split_rest;
    std::tie(split_size, split_count, split_rest) =
        DetermineSplitSize(reduce->mutable_operand(0), split_dim);

    auto main_split_size = split_count * split_size;
    CHECK(main_split_size + split_rest ==
          reduce->mutable_operand(0)->shape().dimensions(split_dim));

    ss << "\n <<< " << start_node_to_splittable_paths[path_key].size()
       << "th Recording splitting reduce " << reduce->name()
       << " operands will be split at dimension " << split_dim
       << " with split size " << split_size << " and rest size " << split_rest;

    if (split_along_reduce_dim) {
      // split on reduce dim so record reduce's split_dim = -1
      AppendToPath(path_key,
                   start_node_to_splittable_paths[path_key].size() - 1, reduce,
                   -1, -1, nullptr);
    } else {
      int64_t reduce_result_split_dim =
          InferReduceResultSplitDim(reduce, split_dim);
      AppendToPath(path_key,
                   start_node_to_splittable_paths[path_key].size() - 1, reduce,
                   reduce_result_split_dim, split_size, nullptr);
    }
    for (int64_t i = 0; i < op_count; i++) {
      TF_RETURN_IF_ERROR(RecordPath(
          path_key, start_node_to_splittable_paths[path_key].size() - 1,
          reduce->mutable_operand(i), split_dim, split_size, split_leafs,
          reduce));
    }
  }

  ss << "\n <---- Exit HandleReduce for '" << reduce->name() << "' SUCCESS";
  LOG(INFO) << ss.str();
  return OkStatus();
}

Status SplittablePathRecorder::HandleSort(HloInstruction* sort) {
  CHECK(Match(sort, m::Sort()));
  if (!Match(sort, m::Sort(m::Op(), m::Iota()))) {
    LOG(WARNING) << "Splitter cannot transform '" << sort->name()
                 << "' which is not supported sort-type operation. "
                 << "Iota is expected as the second parameter";
    return OkStatus();
  }
  HloInstruction* array = sort->mutable_operand(0);
  HloInstruction* indices = sort->mutable_operand(1);

  std::stringstream msg;

  if (!ShapeUtil::CompatibleIgnoringElementType(array->shape(),
                                                indices->shape())) {
    return OkStatus();
  }

  HloInstruction* get_array;
  HloInstruction* get_indices;
  if (sort->shape().IsTuple() && sort->shape().tuple_shapes_size() == 2 &&
      sort->user_count() == 2) {
    get_array = sort->users()[0];
    get_indices = sort->users()[1];
    if (get_array->user_count() != 1 || get_indices->user_count() != 1) {
      LOG(WARNING) << "Splitting pattern doesn't match sort operation as "
                   << "number of users on the left is "
                   << get_array->user_count() << " and on the right is "
                   << get_indices->user_count();
      return OkStatus();
    }
  }

  HloInstruction* array_slice = get_array->users()[0];
  HloInstruction* indices_slice = get_indices->users()[0];
  auto left_is_slice =
      Match(array_slice, m::Slice(m::GetTupleElement(m::Sort())));
  auto right_is_slice =
      Match(indices_slice, m::Slice(m::GetTupleElement(m::Sort())));
  if (!(left_is_slice && right_is_slice)) {
    return OkStatus();
  }

  if (!ShapeUtil::CompatibleIgnoringElementType(array_slice->shape(),
                                                indices_slice->shape())) {
    return OkStatus();
  }

  // Checks that the operation can be split
  auto array_dims_size = array->shape().dimensions().size();
  std::vector<int64_t> original_dims(array_dims_size);
  std::iota(original_dims.begin(), original_dims.end(), 0);

  std::vector<int64_t> exclude_dims;
  std::vector<HloInstruction*> split_leafs;
  std::vector<HloInstruction*> indices_split_leafs;

  bool can_split_array =
      OperandShouldBeSplit(array) &&
      OperandCanBeSplit(array, &split_leafs, &original_dims, &exclude_dims);

  bool can_split_indices = OperandCanBeSplit(indices, &indices_split_leafs,
                                             &original_dims, &exclude_dims);

  if (!(can_split_array && can_split_indices)) {
    LOG(WARNING) << "Operation '" << sort->name()
                 << "' either does not require splitting "
                 << "or cannot be splitted";
    return OkStatus();
  }

  std::vector<int64_t> split_dims =
      BestSplitDim(array, absl::MakeSpan(exclude_dims));
  if (split_dims[0] == -1) {
    LOG(WARNING) << "Failed to find best split dimension for '" << sort->name()
                 << "'";
    return OkStatus();
  }
  std::vector<int64_t> excluded_split_dims = {};
  for (auto split_dim : split_dims) {
    // remove all excluded dim
    if (absl::c_linear_search(exclude_dims, split_dim)) {
      continue;
    }
    excluded_split_dims.emplace_back(split_dim);
  }

  if (excluded_split_dims.empty()) {
    LOG(WARNING) << "Failed to find best split dimension for '" << sort->name()
                 << "'";
    return OkStatus();
  }

  split_dims = excluded_split_dims;

  auto last_dim = array_dims_size - 1;

  if (!absl::c_linear_search(split_dims, last_dim)) {
    LOG(WARNING) << "Best split dimension for '" << sort->name()
                 << "' can only be the last dimension. "
                 << "Best found split dimensions doesn't inculde it ";
    return OkStatus();
  } else {
    // can only be the last dimension
    split_dims = {int64_t(last_dim)};
  }

  auto path_key = MakeSplitNodeKey(sort);
  start_node_vector.push_back(path_key);
  for (auto split_dim : split_dims) {
    int64_t split_size, split_count, split_rest;
    std::tie(split_size, split_count, split_rest) =
        DetermineSplitSize(array, split_dim);

    auto slice_k = array_slice->shape().dimensions(last_dim);
    if (slice_k >= split_size) {
      LOG(WARNING) << "Splitting for '" << sort->name()
                   << "' will not benefit user as the slicing dimension ("
                   << slice_k << ") "
                   << "is larger or equal to the split size (" << split_size
                   << ")";
      return OkStatus();
    }

    auto main_split_size = split_count * split_size;
    CHECK(main_split_size + split_rest == array->shape().dimensions(split_dim));

    TF_RETURN_IF_ERROR(CreateNewEmptyPath(path_key, sort));
    AppendToPath(path_key, start_node_to_splittable_paths[path_key].size() - 1,
                 sort, -1, -1, nullptr);
    TF_RETURN_IF_ERROR(RecordPath(
        path_key, start_node_to_splittable_paths[path_key].size() - 1, array,
        split_dim, split_size, split_leafs, sort));

    TF_RETURN_IF_ERROR(RecordPath(
        path_key, start_node_to_splittable_paths[path_key].size() - 1, indices,
        split_dim, split_size, indices_split_leafs, sort));
  }

  return OkStatus();
}

bool SplittablePathRecorder::SizeCanMerge(SplitNodeKey first_key,
                                          SplitNodeKey second_key) {
  // second_key's while_loop can be merged into first_key's while_loop  only if
  // after merging, the result size of the while_loop is less than or equal to
  // max_size_threshold.
  int64_t first_total_size = 0, second_total_size = 0;
  int64_t first_while_loop_num = start_node_to_while_loop_num[first_key],
          second_while_loop_num = start_node_to_while_loop_num[second_key];
  for (const auto& key : while_loop_num_to_start_node[first_while_loop_num]) {
    first_total_size +=
        ShapeUtil::ByteSizeOfElements(start_node_to_start_inst[key]->shape());
  }
  for (const auto& key : while_loop_num_to_start_node[second_while_loop_num]) {
    second_total_size +=
        ShapeUtil::ByteSizeOfElements(start_node_to_start_inst[key]->shape());
  }
  return first_total_size + second_total_size <= max_size_threshold;
}

bool SplittablePathRecorder::Merged(SplitNodeKey first_key,
                                    SplitNodeKey second_key) {
  if (start_node_to_while_loop_num[first_key] ==
      start_node_to_while_loop_num[second_key]) {
    LOG(INFO) << "[Merged] first_key=" << first_key
              << " second_key=" << second_key << " are already merged";
  }

  return start_node_to_while_loop_num[first_key] ==
         start_node_to_while_loop_num[second_key];
}

bool SplittablePathRecorder::ShouldMerge(int64_t lcs_len,
                                         int64_t first_path_len,
                                         int64_t second_path_len) {
  // two path should be merged iff the common path len is greather than
  // half length of each path length.
  if ((!first_path_len) || (!second_path_len)) return false;
  return float(lcs_len) / float(first_path_len) > lcs_merge_threshold &&
         float(lcs_len) / float(second_path_len) > lcs_merge_threshold;
}

// For two Reducers that are not merged and the sum of their result sizes does
// not exceed the memory limit, the length of the longest common subsequence of
// the two Reducer is calculated, i.e. the length of their longest common path.
// Since multiple split paths are recorded for each Reducer, we need to traverse
// all their path combinations and try to find all the paths that have the
// longest common path. When computing the common path of two paths, a node is
// considered to be the common node of both paths if and only if they would be
// split from the same node of the original computational graph with the same
// split_dim and split_size.
int64_t SplittablePathRecorder::LCS(
    const std::vector<SplitNodeVal>& first_path,
    const std::vector<SplitNodeVal>& second_path) {
  if (first_path.empty() || second_path.empty()) return 0;
  int n = first_path.size();
  int m = second_path.size();
  std::vector<int> dp(m + 1, 0);
  for (int i = 1; i <= n; i++) {
    int upLeft = dp[0];
    for (int j = 1; j <= m; j++) {
      // speed the calculation
      if (IsSplitNodeValSameInstDifferentDim(first_path[i - 1],
                                             second_path[j - 1])) {
        // * cannot merge paths who has the same node but different split_dim
        // this is to avoid transpose makeing different different
        // dot/reduce/...
        // finally have the same split_dim but actually are not.
        LOG(INFO) << "[LCS] first_path_node="
                  << std::get<0>(first_path[i - 1])->name()
                  << " real_split_dim=" << std::get<1>(first_path[i - 1])
                  << " second_path_node="
                  << std::get<0>(second_path[j - 1])->name()
                  << " real_split_dim=" << std::get<1>(second_path[j - 1])
                  << " the two paths cannot be merged";
        return 0;
      }
      int tmp = dp[j];
      if (SplitNodeValEqual(first_path[i - 1], second_path[j - 1]))
        dp[j] = upLeft + 1;
      else
        dp[j] = std::max(dp[j - 1], dp[j]);
      upLeft = tmp;
    }
  }
  return dp[m];
}
Status SplittablePathRecorder::RecordUnmergableWhileLoops(int64_t first_num,
                                                          int64_t second_num) {
  if (!while_loop_num_to_unmegerable_while_loop_num.contains(first_num)) {
    while_loop_num_to_unmegerable_while_loop_num[first_num] = {};
  }
  if (!while_loop_num_to_unmegerable_while_loop_num.contains(second_num)) {
    while_loop_num_to_unmegerable_while_loop_num[second_num] = {};
  }

  while_loop_num_to_unmegerable_while_loop_num[first_num].insert(second_num);
  while_loop_num_to_unmegerable_while_loop_num[second_num].insert(first_num);
  return OkStatus();
}

// We will then try to merge the two while-loops of two Reducers using the paths
// with the longest common path identified by the LCS algorithm. When we want to
// merge the while-loops where two Reducers are located, there are various cases
// of dependencies between them and the split_paths they use.
// 1. Two Reducers don't have any dependency relationship. In this case the two
// Reducers can be merged into a single while-loop.
// 2. There is a dependency between the two Reducer, assuming that Reducer_1
// depends on Reducer_2 and Reducer_2 is not on the split_path of Reducer_1. In
// this case although the two Reducers have a part of common split_path that can
// be merged, they cannot be merged because the computation of Reducer_1 depends
// on the final result of Reducer_2.
// 3. There is a dependency between the two Reducer, assuming that Reducer_1
// depends on Reducer_2 and Reducer_2 is on the split_path of Reducer_1 . If the
// split_dim of Reducer_2 is a non-contracting/non-reduce dimension, then it
// means that the result of Reducer_2 calculated in each iteration will be part
// of the final result of Reducer_2, and therefore this part of the result can
// be used to calculate the result of Reducer_1 in each iteration, and the two
// Reducers can be merged; however, if the split_dim of Reducer_2 is a
// contracting/reduce dimension, then this means that the result computed by
// each iteration of Reducer_2 is not part of the final result of Reducer_2 and
// cannot be used in the computation of Reducer_1, so the two Reducers cannot be
// merged.

// There are another special case we must handle when performing merging.
// Assume that there are 4 Reducers on the graph, both Reducer_2 and Reducer_4
// have two split_paths with split_dim=1 and split_dim=2. Reducer_1 has only one
// split_path with split_dim=1 and Reducer_1 has only one split_path with
// split_dim=2 path. This means that we have a total of 4 merge options and they
// will all end up with two while-loops, but one of them causes problems, namely
// merging Reducer_1 and Reducer_4 into one loop using split_path with
// split_dim=1 and merging Reducer_3 and Reducer_2 into a while-loop with
// split_path=1 using split_path with split_dim=2 into another while-loop. This
// case is shown on the right hand side graph. In this case since the
// computation of Reducer_1 needs to use the results of Reducer_2 and the
// computation of Reducer_3 needs to use the results of Reducer_4, this leads to
// an interdependence between these two while-loops and introduces a loop in the
// computational graph, whereas a reasonable computational graph must be a
// directed acyclic graph. To avoid cycles, we need to introduce a rule when
// trying to merge two while-loops in which the Reducers are located. That is,
// if the split_paths we are trying to merge contain other Reducers that are not
// in the two while-loops, the merge is abandoned. With this rule in place, the
// merging algorithm will merge Reducers that are in the same dependency chain
// before attempting to merge with other Reducers that do not have dependencies.

// When we try to merge two while-loops, if all Reducers in the two while-loops
// doesn't offend any dependency relationship and doesn't cause any cycles, the
// two while-loops would be merged.

Status SplittablePathRecorder::TryMergableRelativeWhileLoop(
    SplitNodeKey first_key, SplitNodeKey second_key,
    const std::vector<size_t>& first_path_indices,
    const std::vector<size_t>& second_path_indices) {
  std::string prefix = "[TryMergableRelativeWhileLoop] ";
  // a set to record all start_nodes has descendant relationships
  absl::flat_hash_set<SplitNodeKey> descendant_start_nodes = {};
  int64_t orig_first_while_loop_num = start_node_to_while_loop_num[first_key];
  int64_t orig_second_while_loop_num = start_node_to_while_loop_num[second_key];
  for (auto first_start_node_key :
       while_loop_num_to_start_node[orig_first_while_loop_num]) {
    descendant_start_nodes.insert(first_start_node_key);
  }
  for (auto second_start_node_key :
       while_loop_num_to_start_node[orig_second_while_loop_num]) {
    descendant_start_nodes.insert(second_start_node_key);
  }
  CHECK(first_path_indices.size() == second_path_indices.size());
  std::vector<size_t> selected_first_path_indices;
  std::vector<size_t> selected_second_path_indices;
  for (int i = 0; i < first_path_indices.size(); ++i) {
    bool can_use = true;
    int64_t first_index = first_path_indices[i];
    int64_t second_index = second_path_indices[i];
    for (auto first_start_node_key :
         while_loop_num_to_start_node[orig_first_while_loop_num]) {
      for (auto val :
           start_node_to_splittable_paths[first_start_node_key][first_index]) {
        HloInstruction* cur_inst = std::get<0>(val);
        SplitNodeKey cur_key = MakeSplitNodeKey(cur_inst);
        if (start_node_set.contains(cur_key) &&
            (!descendant_start_nodes.contains(cur_key))) {
          can_use = false;
          LOG(INFO) << prefix << " while_loop_" << orig_first_while_loop_num
                    << " cur_start_node="
                    << start_node_to_start_inst[first_start_node_key]->name()
                    << " cur_path_index=" << first_index
                    << " contain unmerged start_node.name=" << cur_inst->name()
                    << " skip cur_index=" << first_index;
          break;
        }
      }
      if (!can_use) break;
    }
    for (auto second_start_node_key :
         while_loop_num_to_start_node[orig_second_while_loop_num]) {
      for (auto val : start_node_to_splittable_paths[second_start_node_key]
                                                    [second_index]) {
        HloInstruction* cur_inst = std::get<0>(val);
        SplitNodeKey cur_key = MakeSplitNodeKey(cur_inst);
        if (start_node_set.contains(cur_key) &&
            (!descendant_start_nodes.contains(cur_key))) {
          can_use = false;
          LOG(INFO) << prefix << " while_loop_" << orig_second_while_loop_num
                    << " cur_start_node="
                    << start_node_to_start_inst[second_start_node_key]->name()
                    << " cur_path_index=" << second_index
                    << " contain unmerged start_node.name=" << cur_inst->name()
                    << " skip cur_index=" << second_index;
          break;
        }
      }
      if (!can_use) break;
    }
    if (can_use) {
      selected_first_path_indices.push_back(first_index);
      selected_second_path_indices.push_back(second_index);
    }
  }
  if (selected_first_path_indices.empty()) {
    LOG(INFO) << prefix << " Cannot merge while_loop_"
              << orig_first_while_loop_num << " and while_loop_"
              << orig_second_while_loop_num
              << " because multiple chain merge can only be lanuched by "
                 "their end nodes";
    RecordUnmergableWhileLoops(orig_first_while_loop_num,
                               orig_second_while_loop_num);
    // do not perform merging, return
    return OkStatus();
  }

  // record all root nodes of all descendant chains
  absl::flat_hash_set<SplitNodeKey> chain_root_nodes(descendant_start_nodes);
  for (auto first_start_node_key : descendant_start_nodes) {
    for (auto second_start_node_key : descendant_start_nodes) {
      if (first_start_node_key == second_start_node_key) {
        continue;
      }
      if (start_node_to_decendant_start_nodes[first_start_node_key].contains(
              second_start_node_key)) {
        chain_root_nodes.erase(second_start_node_key);
      } else if (start_node_to_decendant_start_nodes[second_start_node_key]
                     .contains(first_start_node_key)) {
        chain_root_nodes.erase(first_start_node_key);
      }
    }
  }
  LOG(INFO) << prefix << "descendant_start_nodes_set.size="
            << descendant_start_nodes.size()
            << " chain_root_nodes_set.size=" << chain_root_nodes.size();

  // chain_nodes include chain_root itself
  absl::flat_hash_map<SplitNodeKey, absl::flat_hash_set<SplitNodeKey>>
      chain_root_to_chain_nodes;
  for (auto chain_root_node : chain_root_nodes) {
    chain_root_to_chain_nodes[chain_root_node] = {chain_root_node};
    for (auto node : descendant_start_nodes) {
      if (start_node_to_decendant_start_nodes[chain_root_node].contains(node)) {
        chain_root_to_chain_nodes[chain_root_node].insert(node);
      }
    }
  }
  absl::flat_hash_map<SplitNodeKey, std::vector<std::vector<SplitNodeVal>>>
      tmp_start_node_to_splittable_paths;
  // perform the temporary update
  TF_RETURN_IF_ERROR(UpdateNewPaths(first_key, selected_first_path_indices,
                                    tmp_start_node_to_splittable_paths));
  TF_RETURN_IF_ERROR(UpdateNewPaths(second_key, selected_second_path_indices,
                                    tmp_start_node_to_splittable_paths));

  absl::flat_hash_map<SplitNodeKey, std::vector<size_t>>
      chain_root_to_new_chain_path_indices;
  // try to find a path in a chain with split_dim in non-reduce/non-contracting
  // dimensions

  for (auto chain_root_node : chain_root_nodes) {
    std::vector<size_t> new_chain_path_indices = {};
    for (auto i = 0;
         i < tmp_start_node_to_splittable_paths[chain_root_node].size(); ++i) {
      absl::flat_hash_set<SplitNodeKey> chain_nodes(
          chain_root_to_chain_nodes[chain_root_node]);
      // skip chain_root, since it will not be used by other node in the
      // while-loop
      chain_nodes.erase(chain_root_node);
      std::queue<SplitNodeVal> path_node_que;
      // skip root it self
      for (auto j = 1;
           j < tmp_start_node_to_splittable_paths[chain_root_node][i].size();
           j++) {
        path_node_que.push(
            tmp_start_node_to_splittable_paths[chain_root_node][i][j]);
      }
      int counter = 0;
      while (!path_node_que.empty()) {
        auto cur_node_val = path_node_que.front();
        path_node_que.pop();
        ++counter;
        HloInstruction* cur_node_inst = std::get<0>(cur_node_val);
        SplitNodeKey cur_node_key = MakeSplitNodeKey(cur_node_inst);
        int64_t cur_node_split_dim = std::get<1>(cur_node_val);
        int64_t cur_node_split_size = std::get<2>(cur_node_val);
        if (cur_node_split_dim == -1 || cur_node_split_size == -1 ||
            (!chain_nodes.contains(cur_node_key))) {
          continue;
        }

        // find a chain node
        SplitNodeVal start_node_val =
            tmp_start_node_to_splittable_paths[cur_node_key][i][0];
        int64_t start_node_split_dim = std::get<1>(start_node_val);
        int64_t start_node_split_size = std::get<2>(start_node_val);
        // need to check both the dim and size
        if (cur_node_split_dim == start_node_split_dim &&
            cur_node_split_size == start_node_split_size) {
          chain_nodes.erase(cur_node_key);
          // since paths can have branches, so once we meet an extendable
          // node,extend the path(skip cur_node)
          for (auto k = 1;
               k < tmp_start_node_to_splittable_paths[cur_node_key][i].size();
               k++) {
            path_node_que.push(
                tmp_start_node_to_splittable_paths[cur_node_key][i][k]);
          }
        }
      }

      if (chain_nodes.empty()) {
        // can use current path
        new_chain_path_indices.push_back(i);
        LOG(INFO) << prefix << " chain_root_node.name="
                  << start_node_to_start_inst[chain_root_node]->name()
                  << " can use cur_path_index=" << i
                  << " new_chain_path_indices.size="
                  << new_chain_path_indices.size();
      } else {
        LOG(INFO) << prefix << " chain_root_node.name="
                  << start_node_to_start_inst[chain_root_node]->name()
                  << " cannot use cur_path_index=" << i;
      }
    }
    if (new_chain_path_indices.empty()) {
      // cannot find a useable path for the root
      // thus cannot merge the two loops
      LOG(INFO) << prefix
                << " Cannot merge while_loop_1=" << orig_first_while_loop_num
                << " while_loop_2=" << orig_second_while_loop_num
                << " because cannot find a usable path for chain_root="
                << start_node_to_start_inst[chain_root_node]->name();
      RecordUnmergableWhileLoops(orig_first_while_loop_num,
                                 orig_second_while_loop_num);
      // do not perform merging, return
      return OkStatus();
    } else {
      chain_root_to_new_chain_path_indices[chain_root_node] =
          new_chain_path_indices;
    }
  }

  // mergable split_dim should be the intersection of chain_root_path_indices
  // and lcs_indices
  std::vector<size_t> first_lcs_indices(selected_first_path_indices);
  std::vector<size_t> second_lcs_indices(selected_second_path_indices);
  for (auto chain_root_node : chain_root_nodes) {
    bool is_first_loop;
    if (start_node_to_while_loop_num[chain_root_node] ==
        orig_first_while_loop_num) {
      is_first_loop = true;
    } else {
      is_first_loop = false;
    }
    std::vector<size_t> new_chain_path_indices = {};
    for (auto lcs_index :
         (is_first_loop ? first_lcs_indices : second_lcs_indices)) {
      for (auto root_path_index :
           chain_root_to_new_chain_path_indices[chain_root_node]) {
        if (lcs_index == root_path_index) {
          // take the intersection of chain_path_indices and lcs_indices
          new_chain_path_indices.push_back(lcs_index);
        }
      }
    }
    if (new_chain_path_indices.empty()) {
      // cannot find usable path, cannot merge
      LOG(INFO) << prefix
                << " Cannot merge while_loop_1=" << orig_first_while_loop_num
                << " while_loop_2=" << orig_second_while_loop_num
                << " because cannot find a intersection between lcs_indices "
                   "and mergable_indices"
                << start_node_to_start_inst[chain_root_node]->name();
      RecordUnmergableWhileLoops(orig_first_while_loop_num,
                                 orig_second_while_loop_num);
      // do not perform merging, return
      return OkStatus();
    } else {
      // use the intersection
      if (is_first_loop) {
        first_lcs_indices = new_chain_path_indices;
      } else {
        second_lcs_indices = new_chain_path_indices;
      }
    }
  }

  // perform the real update
  TF_RETURN_IF_ERROR(UpdateNewPaths(first_key, first_lcs_indices,
                                    start_node_to_splittable_paths));
  TF_RETURN_IF_ERROR(UpdateNewPaths(second_key, second_lcs_indices,
                                    start_node_to_splittable_paths));
  // merge all starting_nodes in the second while_loop into the first
  // while_loop
  for (auto start_node :
       while_loop_num_to_start_node[orig_second_while_loop_num]) {
    start_node_to_while_loop_num[start_node] = orig_first_while_loop_num;
    while_loop_num_to_start_node[orig_first_while_loop_num].emplace_back(
        start_node);
  }
  // delete second_while_loop
  while_loop_num_to_start_node.erase(orig_second_while_loop_num);
  LOG(INFO) << prefix << " Merge while_loop_" << orig_first_while_loop_num
            << " and while_loop_=" << orig_second_while_loop_num
            << " while_loop_" << orig_first_while_loop_num << ".size="
            << while_loop_num_to_start_node[orig_first_while_loop_num].size();
  return OkStatus();
}

Status SplittablePathRecorder::UpdateNewPaths(
    SplitNodeKey node_key, const std::vector<size_t>& path_indices,
    absl::flat_hash_map<SplitNodeKey, std::vector<std::vector<SplitNodeVal>>>&
        input_start_node_to_splittable_paths) {
  int64_t orig_while_loop_num = start_node_to_while_loop_num[node_key];
  // we need to update all noeds because 2 while-loops can be merged means they
  // have some nodes in common, and those common must have other nodes in common
  // in their own loops, so we need to updates paths of all start nodes in
  // current while-loop
  for (SplitNodeKey start_node_key :
       while_loop_num_to_start_node[orig_while_loop_num]) {
    std::vector<std::vector<SplitNodeVal>> new_paths;
    new_paths.reserve(path_indices.size());
    for (auto index : path_indices) {
      new_paths.emplace_back(
          start_node_to_splittable_paths[start_node_key][index]);
    }
    // update all possible paths
    input_start_node_to_splittable_paths[start_node_key] = new_paths;
  }
  return OkStatus();
}

Status SplittablePathRecorder::RecordInstructionsInWhileLoop() {
  std::string prefix = "[RecordInstructionsInWhileLoop] ";
  // first init all starting_node with a unique while_loop
  LOG(INFO) << prefix << "Start Recording";
  for (auto kv : while_loop_num_to_start_node) {
    int64_t while_loop_num = kv.first;
    while_loop_num_to_instructions[while_loop_num] = {};
    for (auto start_node_key : while_loop_num_to_start_node[while_loop_num]) {
      // add starting_node
      while_loop_num_to_instructions[while_loop_num].insert(
          start_node_to_start_inst[start_node_key]);
      // we will use the first path from now on
      std::vector<SplitNodeVal>& final_path =
          start_node_to_splittable_paths[start_node_key][0];
      for (auto& node_val : final_path) {
        while_loop_num_to_instructions[while_loop_num].insert(
            std::get<0>(node_val));
      }
    }
    LOG(INFO) << prefix << "Finish recording, while_loop_num=" << while_loop_num
              << " instructions.size="
              << while_loop_num_to_instructions[while_loop_num].size();
  }

  return OkStatus();
}

Status SplittablePathRecorder::FindAllDescdantStartNodes(
    SplitNodeKey start_node_key) {
  std::string prefix = "[FindAllDescdantStartNodes] ";
  start_node_to_decendant_start_nodes[start_node_key] = {};
  HloInstruction* cur_start_inst = start_node_to_start_inst[start_node_key];
  LOG(INFO) << prefix << " Start: starting_node=" << cur_start_inst->name();
  std::queue<HloInstruction*> descdant_que;
  absl::flat_hash_set<HloInstruction*> descendant_inst_set;
  for (auto operand_inst : cur_start_inst->operands()) {
    descdant_que.push(operand_inst);
    descendant_inst_set.insert(operand_inst);
  }

  while (!descdant_que.empty()) {
    HloInstruction* cur_descdant_inst = descdant_que.front();
    descdant_que.pop();
    SplitNodeKey cur_descdant_key = MakeSplitNodeKey(cur_descdant_inst);
    if (start_node_set.contains(cur_descdant_key)) {
      start_node_to_decendant_start_nodes[start_node_key].insert(
          cur_descdant_key);
    }
    for (auto operand_inst : cur_descdant_inst->operands()) {
      if (!descendant_inst_set.contains(operand_inst)) {
        descdant_que.push(operand_inst);
        descendant_inst_set.insert(operand_inst);
      }
    }
  }
  return OkStatus();
}

Status SplittablePathRecorder::RecordAllDescdantStartNodes() {
  std::string prefix = "[RecordAllDescdantStartNodes] ";
  for (const auto& kv : start_node_to_splittable_paths) {
    start_node_set.insert(kv.first);
  }
  for (const auto& kv : start_node_to_splittable_paths) {
    FindAllDescdantStartNodes(kv.first);
  }
  LOG(INFO) << prefix << "Finish RecordAllDescdantStartNodes";
  return OkStatus();
}

bool SplittablePathRecorder::HasDesedantRelationship(SplitNodeKey first_key,
                                                     SplitNodeKey second_key) {
  int64_t orig_first_while_loop_num = start_node_to_while_loop_num[first_key];
  int64_t orig_second_while_loop_num = start_node_to_while_loop_num[second_key];

  for (auto first_start_node_key :
       while_loop_num_to_start_node[orig_first_while_loop_num]) {
    for (auto second_start_node_key :
         while_loop_num_to_start_node[orig_second_while_loop_num]) {
      if (start_node_to_decendant_start_nodes[first_start_node_key].contains(
              second_start_node_key) ||
          start_node_to_decendant_start_nodes[second_start_node_key].contains(
              first_start_node_key)) {
        // these two while_loop has starting_nodes are descedant of the other
        return true;
      }
    }
  }
  return false;
}
Status SplittablePathRecorder::InitWhileLoops() {
  std::string prefix = "[InitWhileLoops] ";
  for (const auto& kv : start_node_to_splittable_paths) {
    auto while_loop_num = GenerateNewWhileLoopNum();
    start_node_to_while_loop_num[kv.first] = while_loop_num;
    while_loop_num_to_start_node[while_loop_num] = {kv.first};
  }
  LOG(INFO) << prefix << "Finish InitWhileLoops";
  return OkStatus();
}

// After all the paths to be split have been recorded in the previous phase, the
// purpose of this phase is to try to merge the subgraphs that can be merged
// represented by these paths. Algorithm AllocateWhileLoops is an overview of
// the merging process. The first step is to assign a separate while-loop to
// each Reducer, then iterate through all the Reducer combinations in pairs,
// using the LCS(Longest Common Subsequence) algorithm to find the common path
// length of the two Reducers, and if this length is greater than a
// predetermined threshold, try to merge the loops in which the two Reducers are
// located. However, not all Reducers with a common path can be merged, and the
// merging process requires a lot of dependency analysis.

Status SplittablePathRecorder::AllocateWhileLoops() {
  std::string prefix = "[AllocateWhileLoops] ";
  // first init all starting_node with a unique while_loop
  LOG(INFO) << prefix << "Start AllocateWhileLoops";
  InitWhileLoops();
  RecordAllDescdantStartNodes();
  for (auto first_key : start_node_vector) {
    for (auto second_key : start_node_vector) {
      if (first_key == second_key || Merged(first_key, second_key) ||
          !SizeCanMerge(first_key, second_key)) {
        continue;
      }
      if (start_node_to_start_inst[first_key]->opcode() == HloOpcode::kSort ||
          start_node_to_start_inst[second_key]->opcode() == HloOpcode::kSort) {
        continue;
      }
      int64_t first_while_loop_num = start_node_to_while_loop_num[first_key];
      int64_t second_while_loop_num = start_node_to_while_loop_num[second_key];
      if (while_loop_num_to_unmegerable_while_loop_num.contains(
              first_while_loop_num) &&
          while_loop_num_to_unmegerable_while_loop_num[first_while_loop_num]
              .contains(second_while_loop_num)) {
        // donn't need to check
        // while_loop_num_to_unmegerable_while_loop_num[second_while_loop_num]
        // since they must be contain to each other's
        // while_loop_num_to_unmegerable_while_loop_num
        continue;
      }
      const auto& first_paths = start_node_to_splittable_paths[first_key];
      const auto& second_paths = start_node_to_splittable_paths[second_key];
      int64_t best_lcs_len = 0;
      std::vector<size_t> best_lcs_path_indices_first = {};
      std::vector<size_t> best_lcs_path_indices_second = {};
      for (auto i = 0; i < first_paths.size(); ++i) {
        for (auto j = 0; j < second_paths.size(); ++j) {
          int64_t cur_lcs_len = LCS(first_paths[i], second_paths[j]);
          if (cur_lcs_len == 0) continue;
          if (cur_lcs_len > best_lcs_len &&
              ShouldMerge(cur_lcs_len, first_paths[i].size(),
                          second_paths[j].size())) {
            best_lcs_len = cur_lcs_len;
            best_lcs_path_indices_first = {i};
            best_lcs_path_indices_second = {j};
          } else if (cur_lcs_len == best_lcs_len &&
                     ShouldMerge(cur_lcs_len, first_paths[i].size(),
                                 second_paths[j].size())) {
            // need to record all possible best_merge_paths, the compatiable
            // paths have the same index.
            best_lcs_path_indices_first.emplace_back(i);
            best_lcs_path_indices_second.emplace_back(j);
          }
        }
      }

      if (best_lcs_len == 0) continue;
      LOG(INFO) << prefix << "Try merge first_start_node_inst="
                << start_node_to_start_inst[first_key]->name()
                << " second_start_node_inst="
                << start_node_to_start_inst[second_key]->name();
      for (int i = 0; i < best_lcs_path_indices_first.size(); ++i) {
        std::stringstream msg;
        msg << start_node_to_start_inst[first_key]->name() << " # path_"
            << best_lcs_path_indices_first[i] << ": {";
        for (int j = 0; j < first_paths[best_lcs_path_indices_first[i]].size();
             ++j) {
          msg << std::get<0>(first_paths[best_lcs_path_indices_first[i]][j])
                     ->name()
              << " ";
        }
        msg << "}\n";
        msg << start_node_to_start_inst[second_key]->name() << " # path_"
            << best_lcs_path_indices_second[i] << ": {";
        for (int j = 0;
             j < second_paths[best_lcs_path_indices_second[i]].size(); ++j) {
          msg << std::get<0>(second_paths[best_lcs_path_indices_second[i]][j])
                     ->name()
              << " ";
        }
        msg << "}\n";
        LOG(INFO) << msg.str();
      }
      TryMergableRelativeWhileLoop(first_key, second_key,
                                   best_lcs_path_indices_first,
                                   best_lcs_path_indices_second);
    }
  }

  RecordInstructionsInWhileLoop();
  LOG(INFO) << prefix << "Finish AllocateWhileLoops";
  return OkStatus();
}

StatusOr<HloInstruction*>
TensorSplitterRewriteVisitor::Splitter::SplitInstruction(HloInstruction* inst,
                                                         int64_t split_dim,
                                                         int64_t split_size) {
  LOG(INFO) << "\n @@@ Enter SplitInstruction for '" << inst->name() << "'";
  auto visited_inst_key =
      MakeVisitedInstructionKey(inst, split_dim, split_size);
  if (visited_instructions_.contains(visited_inst_key)) {
    LOG(INFO) << "\n &&& Found a duplicate for " << inst->name() << ", "
              << split_dim << ", " << split_size << ">";
    return visited_instructions_[visited_inst_key];
  }
  if (absl::c_linear_search(leafs_, inst)) {
    LOG(INFO) << "\n> Found in leafs '" << inst->name() << "'";
    if (Match(inst, m::Dot())) {
      LOG(INFO) << "\n# Split 'Dot' instruction '" << inst->name() << "'";
      TF_ASSIGN_OR_RETURN(HloInstruction * new_inst,
                          SplitLeafDot(inst, split_dim, split_size));
      visited_instructions_[visited_inst_key] = new_inst;
      return new_inst;
    } else if (Match(inst, m::Broadcast())) {
      LOG(INFO) << "\n# Split 'Broadcast' instruction '" << inst->name() << "'";
      TF_ASSIGN_OR_RETURN(HloInstruction * new_inst,
                          SplitLeafBroadcast(inst, split_dim, split_size));
      visited_instructions_[visited_inst_key] = new_inst;
      return new_inst;
    } else if (Match(inst, m::Parameter())) {
      LOG(INFO) << "\n# Split 'Parameter' instruction '" << inst->name() << "'";
      TF_ASSIGN_OR_RETURN(HloInstruction * new_inst,
                          SplitLeafParameter(inst, split_dim, split_size));
      visited_instructions_[visited_inst_key] = new_inst;
      return new_inst;
    } else if (Match(inst, m::Iota())) {
      LOG(INFO) << "\n# Split 'Iota' instruction '" << inst->name() << "'";
      TF_ASSIGN_OR_RETURN(HloInstruction * new_inst,
                          SplitLeafIota(inst, split_dim, split_size));
      visited_instructions_[visited_inst_key] = new_inst;
      return new_inst;
    }
  } else {
    HloInstruction *operand, *lhs, *rhs;
    std::vector<HloInstruction*> operands;
    bool merge_rest = GetDebugOptionsFromFlags().xla_tensor_split_merge_rest();

    if (Match(inst, m::Transpose(m::Op(&operand)))) {
      // For a transpose, the transpose might change which dimension is
      // being split. So we obtain the new split dimension and then
      // recursively a new operand to make a clone.
      int64_t operand_split_dim = inst->dimensions(split_dim);
      LOG(INFO) << "\n# Split 'Transpose:Op' instruction '" << inst->name()
                << "'";
      TF_ASSIGN_OR_RETURN(
          HloInstruction * new_operand,
          SplitInstruction(operand, operand_split_dim, split_size));

      Shape new_shape = ShapeUtil::MakeShape(inst->shape().element_type(),
                                             inst->shape().dimensions());
      new_shape.set_dimensions(split_dim, split_size);
      HloInstruction* new_inst;
      int64_t inst_split_dim_size = inst->shape().dimensions(split_dim);
      int64_t main_split_size =
          int64_t(inst_split_dim_size / split_size) * split_size;
      if (merge_rest && main_split_size < inst_split_dim_size) {
        std::vector<bool> dynamic_dimensions(new_shape.dimensions_size(),
                                             false);
        dynamic_dimensions[split_dim] = true;
        Shape dynamic_new_shape =
            ShapeUtil::MakeShape(new_shape.element_type(),
                                 new_shape.dimensions(), dynamic_dimensions);
        new_inst = builder_.AddInstruction(
            inst->CloneWithNewOperands(dynamic_new_shape, {new_operand}));
      } else {
        new_inst = builder_.AddInstruction(
            inst->CloneWithNewOperands(new_shape, {new_operand}));
      }
      visited_instructions_[visited_inst_key] = new_inst;
      return new_inst;
    } else if (MatchSupportedNestedReduce(inst)) {
      // For a reduce, split the 0th and only operand
      // (the initializer a scalar, so all we need to do
      // is update the shape and clone the operand with new
      // inputs)

      LOG(INFO) << "\n# Split 'NestedReduce' instruction '" << inst->name()
                << "'";
      int64_t operand_split_dim = split_dim;  // split dim in operand
      if (inst->dimensions(0) <= split_dim) {
        operand_split_dim += 1;
      }

      TF_ASSIGN_OR_RETURN(HloInstruction * new_operand,
                          SplitInstruction(inst->mutable_operand(0),
                                           operand_split_dim, split_size));

      HloInstruction* init_operand = inst->mutable_operand(1);
      HloInstruction* new_init_operand;
      AddParameter(init_operand, &new_init_operand);

      Shape new_shape = ShapeUtil::MakeShape(inst->shape().element_type(),
                                             inst->shape().dimensions());
      new_shape.set_dimensions(split_dim, split_size);
      HloInstruction* new_inst;
      int64_t inst_split_dim_size = inst->shape().dimensions(split_dim);
      int64_t main_split_size =
          int64_t(inst_split_dim_size / split_size) * split_size;
      if (merge_rest && main_split_size < inst_split_dim_size) {
        std::vector<bool> dynamic_dimensions(new_shape.dimensions_size(),
                                             false);
        dynamic_dimensions[split_dim] = true;
        Shape dynamic_new_shape =
            ShapeUtil::MakeShape(new_shape.element_type(),
                                 new_shape.dimensions(), dynamic_dimensions);
        new_inst = builder_.AddInstruction(inst->CloneWithNewOperands(
            dynamic_new_shape, {new_operand, new_init_operand}));
      } else {
        new_inst = builder_.AddInstruction(inst->CloneWithNewOperands(
            new_shape, {new_operand, new_init_operand}));
      }
      visited_instructions_[visited_inst_key] = new_inst;
      return new_inst;
    } else if (inst->opcode() == HloOpcode::kTriangularSolve) {
      LOG(INFO) << "\n# Split 'TriangularSolve' instruction '" << inst->name()
                << "'";
      TF_ASSIGN_OR_RETURN(
          HloInstruction * new_operand,
          SplitInstruction(inst->mutable_operand(1), split_dim, split_size));
      HloInstruction* mat;
      AddParameter(inst->mutable_operand(0), &mat);
      HloInstruction* new_inst;
      int64_t inst_split_dim_size =
          inst->mutable_operand(1)->shape().dimensions(split_dim);
      int64_t main_split_size =
          int64_t(inst_split_dim_size / split_size) * split_size;
      if (merge_rest && main_split_size < inst_split_dim_size) {
        std::vector<bool> dynamic_dimensions(
            new_operand->shape().dimensions_size(), false);
        dynamic_dimensions[split_dim] = true;
        Shape dynamic_new_shape = ShapeUtil::MakeShape(
            new_operand->shape().element_type(),
            new_operand->shape().dimensions(), dynamic_dimensions);
        new_inst = builder_.AddInstruction(
            inst->CloneWithNewOperands(dynamic_new_shape, {mat, new_operand}));
      } else {
        new_inst = builder_.AddInstruction(inst->CloneWithNewOperands(
            new_operand->shape(), {mat, new_operand}));
      }
      visited_instructions_[visited_inst_key] = new_inst;
      return new_inst;
    } else if (Match(inst, m::Dot(m::Op(&lhs), m::Op(&rhs)))) {
      std::stringstream msg;
      msg << "\n# Split 'Dot(Op, Op)' instruction '" << inst->name() << "'";
      // For an intermediate dot, split the correct operand and assemble
      // a new dot.
      bool split_lhs = ShapeUtil::ElementsIn(lhs->shape()) >
                       ShapeUtil::ElementsIn(
                           rhs->shape());  // this works, since only one of
                                           // the operands needs to be split
      msg << ", split_dim=" << split_dim << ", split_size=" << split_size
          << ", is_lhs=" << (split_lhs ? "yes" : "no");
      LOG(INFO) << msg.str();

      if (split_lhs) {
        CHECK(split_dim < lhs->shape().dimensions_size() - 1);
        int64_t lhs_contr_dim =
            inst->dot_dimension_numbers().lhs_contracting_dimensions(0);
        int64_t lhs_split_dim =
            split_dim >= lhs_contr_dim ? split_dim + 1 : split_dim;

        TF_ASSIGN_OR_RETURN(HloInstruction * new_lhs,
                            SplitInstruction(lhs, lhs_split_dim, split_size));
        HloInstruction* param_rhs;
        AddParameter(rhs, &param_rhs);

        Shape new_shape = ShapeUtil::MakeShape(inst->shape().element_type(),
                                               inst->shape().dimensions());
        new_shape.set_dimensions(split_dim, split_size);
        HloInstruction* new_inst;
        int64_t inst_split_dim_size = inst->shape().dimensions(split_dim);
        int64_t main_split_size =
            int64_t(inst_split_dim_size / split_size) * split_size;
        if (merge_rest && main_split_size < inst_split_dim_size) {
          std::vector<bool> dynamic_dimensions(new_shape.dimensions_size(),
                                               false);
          dynamic_dimensions[split_dim] = true;
          Shape dynamic_new_shape =
              ShapeUtil::MakeShape(new_shape.element_type(),
                                   new_shape.dimensions(), dynamic_dimensions);
          new_inst = builder_.AddInstruction(inst->CloneWithNewOperands(
              dynamic_new_shape, {new_lhs, param_rhs}));
        } else {
          new_inst = builder_.AddInstruction(
              inst->CloneWithNewOperands(new_shape, {new_lhs, param_rhs}));
        }
        visited_instructions_[visited_inst_key] = new_inst;
        return new_inst;
      } else {
        int64_t rhs_start = lhs->shape().dimensions_size() - 1;
        CHECK(split_dim >= rhs_start);
        int64_t rhs_contr_dim =
            inst->dot_dimension_numbers().rhs_contracting_dimensions(0);
        int64_t rhs_split_dim = split_dim - rhs_start >= rhs_contr_dim
                                    ? split_dim + 1 - rhs_start
                                    : split_dim - rhs_start;

        TF_ASSIGN_OR_RETURN(HloInstruction * new_rhs,
                            SplitInstruction(rhs, rhs_split_dim, split_size));
        Shape new_shape = ShapeUtil::MakeShape(inst->shape().element_type(),
                                               inst->shape().dimensions());
        HloInstruction* param_lhs;
        AddParameter(lhs, &param_lhs);

        new_shape.set_dimensions(split_dim, split_size);
        HloInstruction* new_inst;
        int64_t inst_split_dim_size = inst->shape().dimensions(split_dim);
        int64_t main_split_size =
            int64_t(inst_split_dim_size / split_size) * split_size;
        if (merge_rest && main_split_size < inst_split_dim_size) {
          std::vector<bool> dynamic_dimensions(new_shape.dimensions_size(),
                                               false);
          dynamic_dimensions[split_dim] = true;
          Shape dynamic_new_shape =
              ShapeUtil::MakeShape(new_shape.element_type(),
                                   new_shape.dimensions(), dynamic_dimensions);
          new_inst = builder_.AddInstruction(inst->CloneWithNewOperands(
              dynamic_new_shape, {param_lhs, new_rhs}));
        } else {
          new_inst = builder_.AddInstruction(
              inst->CloneWithNewOperands(new_shape, {param_lhs, new_rhs}));
        }
        visited_instructions_[visited_inst_key] = new_inst;
        return new_inst;
      }
    } else if (MatchPointwiseNary(inst, &operands)) {
      // For a pointwise operation recursively obtain the new operands and
      // clone the operation.
      LOG(INFO) << "\n# Split 'PointwiseNary' instruction '" << inst->name()
                << "'";
      std::vector<HloInstruction*> ops;
      for (HloInstruction* operand : operands) {
        TF_ASSIGN_OR_RETURN(HloInstruction * new_operand,
                            SplitInstruction(operand, split_dim, split_size));
        ops.push_back(new_operand);
      }
      Shape new_shape = ShapeUtil::MakeShape(inst->shape().element_type(),
                                             inst->shape().dimensions());
      new_shape.set_dimensions(split_dim, split_size);
      HloInstruction* new_inst;
      int64_t inst_split_dim_size = inst->shape().dimensions(split_dim);
      int64_t main_split_size =
          int64_t(inst_split_dim_size / split_size) * split_size;
      if (merge_rest && main_split_size < inst_split_dim_size) {
        std::vector<bool> dynamic_dimensions(new_shape.dimensions_size(),
                                             false);
        dynamic_dimensions[split_dim] = true;
        Shape dynamic_new_shape =
            ShapeUtil::MakeShape(new_shape.element_type(),
                                 new_shape.dimensions(), dynamic_dimensions);
        new_inst = builder_.AddInstruction(
            inst->CloneWithNewOperands(dynamic_new_shape, absl::MakeSpan(ops)));
      } else {
        new_inst = builder_.AddInstruction(
            inst->CloneWithNewOperands(new_shape, absl::MakeSpan(ops)));
      }
      visited_instructions_[visited_inst_key] = new_inst;
      return new_inst;
    } else {
      // Invariant violation
      // TODO: Is there a more idiomatic way to return a bad status?
      LOG(ERROR) << "Trying to split invalid operation '" << inst->name()
                 << "'";
      CHECK(false);
    }
  }
}

StatusOr<HloInstruction*> TensorSplitterRewriteVisitor::Splitter::SplitLeafDot(
    HloInstruction* dot, int64_t split_dim, int64_t split_size) {
  HloInstruction *lhs, *rhs;
  CHECK(Match(dot, m::Dot(m::Op(&lhs), m::Op(&rhs))));

  // For the dot we identify the parameter to split and then
  // Generate the final dot operation, as well as the operand
  // vector.

  Shape dot_shape = ShapeUtil::MakeShape(dot->shape().element_type(),
                                         dot->shape().dimensions());
  int64_t dot_split_dim = split_dim;
  dot_shape.set_dimensions(dot_split_dim, split_size);

  auto& dnums = dot->dot_dimension_numbers();
  int64_t dims_lhs =
      lhs->shape().rank() - dnums.lhs_contracting_dimensions_size();

  HloInstruction *split_op, *join_op;
  bool split_is_lhs;
  if (split_dim < dims_lhs) {
    // We are splitting up the lhs
    split_is_lhs = true;
    split_op = lhs;
    join_op = rhs;
    // TODO: Check if this is robust for multiple indices ...
    for (int64_t i = 0; i < dnums.lhs_contracting_dimensions_size(); i++) {
      if (split_dim >= dnums.lhs_contracting_dimensions(i)) split_dim += 1;
    }
  } else {
    // We are splitting up the rhs
    split_is_lhs = false;
    split_dim -= dims_lhs;
    split_op = rhs;
    join_op = lhs;
    // TODO: Check if this is robust for multiple indices ...
    for (int64_t i = 0; i < dnums.rhs_contracting_dimensions_size(); i++) {
      if (split_dim >= dnums.rhs_contracting_dimensions(i)) split_dim += 1;
    }
  }

  LOG(INFO) << "<<< "
            << "Splitting leaf dot " << dot->ToString()
            << "; split_dim=" << dot_split_dim << "; split_size=" << split_size
            << "; split_lhs=" << (split_is_lhs ? "yes" : "no")
            << "; op_split_dim=" << split_dim;

  // add parameters
  HloInstruction* split_op_param;
  int64_t split_op_tuple_idx = AddParameter(split_op, &split_op_param);
  HloInstruction* join_op_param;
  int64_t join_op_tuple_idx = AddParameter(join_op, &join_op_param);
  bool merge_rest = GetDebugOptionsFromFlags().xla_tensor_split_merge_rest();
  int64_t split_dim_size = split_op_param->shape().dimensions(split_dim);
  int64_t main_split_size = int64_t(split_dim_size / split_size) * split_size;
  if (merge_rest && main_split_size < split_dim_size) {
    // need padding to (split_count+1)*split_size
    int64_t padded_split_dim_size = main_split_size + split_size;
    PaddingConfig padding;
    Shape padded_shape =
        ShapeUtil::MakeShape(split_op_param->shape().element_type(),
                             split_op_param->shape().dimensions());
    padded_shape.set_dimensions(split_dim, padded_split_dim_size);
    for (int dim = 0; dim < split_op_param->shape().dimensions_size(); ++dim) {
      PaddingConfig::PaddingConfigDimension* dimension =
          padding.add_dimensions();
      dimension->set_edge_padding_low(0);
      if (dim == split_dim) {
        dimension->set_edge_padding_high(padded_split_dim_size -
                                         split_dim_size);
      } else {
        dimension->set_edge_padding_high(0);
      }
      dimension->set_interior_padding(0);
    }
    LOG(INFO) << "[SplitLeafDot] "
              << "split_op_param=" << split_op_param->ToString()
              << " split_dim=" << split_dim
              << " split_dim_size=" << split_dim_size
              << " split_op_param.shpe=" << split_op_param->shape().ToString();
    HloInstruction* zero =
        builder_.AddInstruction(HloInstruction::CreateConstant(
            LiteralUtil::Zero(padded_shape.element_type())));
    split_op_param = builder_.AddInstruction(
        HloInstruction::CreatePad(padded_shape, split_op_param, zero, padding));
    LOG(INFO) << "[SplitLeafDot] "
              << "After Padding: split_op=" << split_op_param->ToString()
              << " split_dim=" << split_dim
              << " split_dim_size=" << split_dim_size
              << " split_op.shpe=" << split_op_param->shape().ToString();
  }

  // dynamic slice by index
  Shape split_shape =
      ShapeUtil::MakeShape(split_op_param->shape().element_type(),
                           split_op_param->shape().dimensions());
  split_shape.set_dimensions(split_dim, split_size);

  std::vector<HloInstruction*> start_indices;
  for (int64_t dim = 0; dim < split_shape.dimensions_size(); dim++) {
    if (dim == split_dim) {
      start_indices.push_back(offset_);
    } else {
      start_indices.push_back(
          builder_.AddInstruction(CREATE_CONSTANT_INT32(0)));
    }
  }
  HloInstruction* split_slice =
      builder_.AddInstruction(HloInstruction::CreateDynamicSlice(
          split_shape, split_op_param, absl::MakeSpan(start_indices),
          split_shape.dimensions()));

  if (merge_rest && main_split_size < split_dim_size) {
    HloInstruction* orig_split_dim_size =
        builder_.AddInstruction(HloInstruction::CreateConstant(
            LiteralUtil::CreateR0<int32_t>(split_dim_size)));
    HloInstruction* split_size_const =
        builder_.AddInstruction(HloInstruction::CreateConstant(
            LiteralUtil::CreateR0<int32_t>(split_size)));
    // offset starts form zero, so
    // dynamic_size=min(orig_split_dim_size-offset_,split_size_const)
    HloInstruction* tmp_split_size = builder_.AddInstruction(
        HloInstruction::CreateBinary(offset_->shape(), HloOpcode::kSubtract,
                                     orig_split_dim_size, offset_));
    HloInstruction* dynamic_size = builder_.AddInstruction(
        HloInstruction::CreateBinary(offset_->shape(), HloOpcode::kMinimum,
                                     tmp_split_size, split_size_const));
    std::vector<bool> dynamic_dimensions(split_shape.dimensions_size(), false);
    dynamic_dimensions[split_dim] = true;
    Shape dynamic_split_slice_shape = ShapeUtil::MakeShape(
        split_slice->shape().element_type(), split_slice->shape().dimensions(),
        dynamic_dimensions);
    split_slice =
        builder_.AddInstruction(HloInstruction::CreateSetDimensionSize(
            dynamic_split_slice_shape, split_slice, dynamic_size, split_dim));
  }

  LOG(INFO) << "[SplitLeafDot] "
            << "dot.name=" << dot->name() << " split_dim=" << split_dim
            << " offset_=" << offset_->ToString()
            << " Create split_slice: slice=" << split_slice->ToString()
            << " slice_shape=" << split_slice->shape().ToString();
  // build the final dot
  std::vector<HloInstruction*> ops;
  if (split_is_lhs) {
    ops = {split_slice, join_op_param};
  } else {
    ops = {join_op_param, split_slice};
  }
  std::vector<bool> dynamic_dimensions(dot_shape.dimensions_size(), false);
  dynamic_dimensions[dot_split_dim] = true;
  Shape dynamic_dot_shape = ShapeUtil::MakeShape(
      dot_shape.element_type(), dot_shape.dimensions(), dynamic_dimensions);
  HloInstruction* new_leaf_dot = builder_.AddInstruction(
      dot->CloneWithNewOperands(dynamic_dot_shape, absl::MakeSpan(ops)));
  LOG(INFO) << "[SplitLeafDot] "
            << "new_dot=" << new_leaf_dot->ToString();
  return new_leaf_dot;
}

StatusOr<HloInstruction*>
TensorSplitterRewriteVisitor::Splitter::SplitLeafBroadcast(
    HloInstruction* broadcast, int64_t split_dim, int64_t split_size) {
  HloInstruction* operand;
  CHECK(Match(broadcast, m::Broadcast(m::Op(&operand))));

  // For a broadcast, we identify if we can split it by
  // changeing the broadcast itself, of if we have to
  // create slices of the underlying operand tensor.

  bool split_on_original_dim =
      absl::c_linear_search(broadcast->dimensions(), split_dim);

  int64_t parameter_idx;
  Shape parameter_shape = ShapeUtil::MakeShape(operand->shape().element_type(),
                                               operand->shape().dimensions());

  std::stringstream msg;
  msg << "broadcast=" << broadcast->ToString();
  msg << " broadcast->dimentions[";
  for (auto d : broadcast->dimensions()) msg << d << ",";
  msg << "], broadcast->dimentions().size=" << broadcast->dimensions().size();
  msg << ", split_dim=" << split_dim << ", split_size=" << split_size;

  msg << ", split_on_original_dim=" << split_on_original_dim;
  msg << ", operand_shape=" << parameter_shape;
  LOG(INFO) << "\n> @@@ " << msg.str();

  HloInstruction* new_operand;
  bool merge_rest = GetDebugOptionsFromFlags().xla_tensor_split_merge_rest();
  if (split_on_original_dim) {
    // we need to slice the parameter ...
    int64_t operand_split_dim;
    for (int64_t i = 0; i < broadcast->dimensions().size(); i++) {
      if (broadcast->dimensions(i) == split_dim) {
        operand_split_dim = i;
        break;
      }
    }

    parameter_shape.set_dimensions(operand_split_dim, split_size);

    std::vector<HloInstruction*> start_indices;
    for (int64_t dim = 0; dim < operand->shape().dimensions_size(); dim++) {
      if (dim == operand_split_dim) {
        start_indices.push_back(offset_);
      } else {
        start_indices.push_back(
            builder_.AddInstruction(CREATE_CONSTANT_INT32(0)));
      }
    }

    HloInstruction* parameter;
    parameter_idx = AddParameter(operand, &parameter);

    int64_t operand_split_dim_size =
        parameter->shape().dimensions(operand_split_dim);
    int64_t main_split_size =
        int64_t(operand_split_dim_size / split_size) * split_size;
    if (merge_rest && main_split_size < operand_split_dim_size) {
      // need padding to (split_count+1)*split_size
      int64_t padded_split_dim_size = main_split_size + split_size;
      PaddingConfig padding;
      Shape padded_shape = ShapeUtil::MakeShape(
          parameter->shape().element_type(), parameter->shape().dimensions());
      padded_shape.set_dimensions(operand_split_dim, padded_split_dim_size);
      for (int dim = 0; dim < parameter->shape().dimensions_size(); ++dim) {
        PaddingConfig::PaddingConfigDimension* dimension =
            padding.add_dimensions();
        dimension->set_edge_padding_low(0);
        if (dim == operand_split_dim) {
          dimension->set_edge_padding_high(padded_split_dim_size -
                                           operand_split_dim_size);
        } else {
          dimension->set_edge_padding_high(0);
        }
        dimension->set_interior_padding(0);
      }
      LOG(INFO) << "[SplitLeafBroadcast] "
                << " broadcast.shape=" << broadcast->shape().ToString()
                << " parameter.name=" << parameter->name()
                << " operand_split_dim=" << operand_split_dim
                << " operand_split_dim_size=" << operand_split_dim_size
                << " split_op.shape=" << parameter->shape().ToString()
                << " main_split_size=" << main_split_size
                << " padded_split_dim_size=" << padded_split_dim_size;
      HloInstruction* zero =
          builder_.AddInstruction(HloInstruction::CreateConstant(
              LiteralUtil::Zero(padded_shape.element_type())));
      parameter = builder_.AddInstruction(
          HloInstruction::CreatePad(padded_shape, parameter, zero, padding));
      LOG(INFO) << "[SplitLeafBroadcast] "
                << "After Padding: parameter.name=" << parameter->name()
                << " operand_split_dim=" << operand_split_dim
                << " operand_split_dim_size=" << operand_split_dim_size
                << " parameter=" << parameter->ToString();
    }

    new_operand = builder_.AddInstruction(HloInstruction::CreateDynamicSlice(
        parameter_shape, parameter, absl::MakeSpan(start_indices),
        parameter_shape.dimensions()));
    if (merge_rest && main_split_size < operand_split_dim_size) {
      HloInstruction* orig_split_dim_size =
          builder_.AddInstruction(HloInstruction::CreateConstant(
              LiteralUtil::CreateR0<int32_t>(operand_split_dim_size)));
      HloInstruction* split_size_const =
          builder_.AddInstruction(HloInstruction::CreateConstant(
              LiteralUtil::CreateR0<int32_t>(split_size)));
      // offset starts form zero, so
      // dynamic_size=min(orig_split_dim_size-offset_,split_size_const)
      HloInstruction* tmp_split_size = builder_.AddInstruction(
          HloInstruction::CreateBinary(offset_->shape(), HloOpcode::kSubtract,
                                       orig_split_dim_size, offset_));
      HloInstruction* dynamic_size = builder_.AddInstruction(
          HloInstruction::CreateBinary(offset_->shape(), HloOpcode::kMinimum,
                                       tmp_split_size, split_size_const));
      std::vector<bool> dynamic_dimensions(
          new_operand->shape().dimensions_size(), false);
      dynamic_dimensions[operand_split_dim] = true;
      Shape dynamic_new_operand_shape = ShapeUtil::MakeShape(
          new_operand->shape().element_type(),
          new_operand->shape().dimensions(), dynamic_dimensions);
      new_operand =
          builder_.AddInstruction(HloInstruction::CreateSetDimensionSize(
              dynamic_new_operand_shape, new_operand, dynamic_size,
              operand_split_dim));
      LOG(INFO) << "[SplitLeafBroadcast] "
                << "Create SetDimensionSize after new_operand="
                << new_operand->name()
                << " operand_split_dim=" << operand_split_dim
                << " operand_split_dim_size=" << operand_split_dim_size
                << " new_operand" << new_operand->ToString();
    }

  } else {
    // This will be a parameter and we just modify the broadcast ...
    parameter_idx = AddParameter(operand, &new_operand);
  }

  Shape broadcast_shape = ShapeUtil::MakeShape(
      broadcast->shape().element_type(), broadcast->shape().dimensions());
  broadcast_shape.set_dimensions(split_dim, split_size);
  std::vector<HloInstruction*> params = {new_operand};
  HloInstruction* new_boradcast_inst;
  int64_t split_dim_size = broadcast->shape().dimensions(split_dim);
  int64_t main_split_size = int64_t(split_dim_size / split_size) * split_size;
  if (merge_rest && main_split_size < split_dim_size) {
    std::vector<bool> dynamic_dimensions(broadcast_shape.dimensions_size(),
                                         false);
    dynamic_dimensions[split_dim] = true;
    Shape dynamic_broadcast_shape =
        ShapeUtil::MakeShape(broadcast_shape.element_type(),
                             broadcast_shape.dimensions(), dynamic_dimensions);
    new_boradcast_inst =
        builder_.AddInstruction(broadcast->CloneWithNewOperands(
            broadcast_shape, absl::MakeSpan(params)));
  } else {
    new_boradcast_inst =
        builder_.AddInstruction(broadcast->CloneWithNewOperands(
            broadcast_shape, absl::MakeSpan(params)));
  }
  LOG(INFO) << "[SplitLeafBroadcast] "
            << "new_broadcast=" << new_boradcast_inst->ToString();

  return new_boradcast_inst;
}

StatusOr<HloInstruction*>
TensorSplitterRewriteVisitor::Splitter::SplitLeafParameter(
    HloInstruction* parameter, int64_t split_dim, int64_t split_size) {
  CHECK(Match(parameter, m::Parameter()));
  const Shape& parameter_shape = parameter->shape();
  const auto& parameter_dims = parameter_shape.dimensions();
  const auto& element_type = parameter_shape.element_type();
  CHECK(parameter_shape.dimensions_size() > split_dim);

  HloInstruction* get_tuple_parameter;
  auto parameter_idx = AddParameter(parameter, &get_tuple_parameter);

  Shape slice_shape = ShapeUtil::MakeShape(element_type, parameter_dims);
  slice_shape.set_dimensions(split_dim, split_size);

  std::vector<HloInstruction*> start_indices;
  for (auto dim = 0; dim < parameter_shape.dimensions_size(); dim++) {
    if (dim == split_dim) {
      start_indices.push_back(offset_);
    } else {
      start_indices.push_back(
          builder_.AddInstruction(CREATE_CONSTANT_INT32(0)));
    }
  }

  std::stringstream msg;
  msg << "parameter '" << parameter->name() << "' dimensions[";
  for (auto d : parameter_dims) msg << d << ",";
  msg << "], parameter->dimentions().size=" << parameter_dims.size();
  msg << ", shape=" << parameter_shape;
  msg << ", split_dim=" << split_dim << ", split_size=" << split_size;
  msg << ", split_dim_for_param=" << parameter_shape.dimensions(split_dim);
  LOG(INFO) << "\n> @@@ " << msg.str();
  LOG(INFO) << "[SplitLeafParameter] "
            << "parameter.name=" << parameter->name()
            << "parameter.shape=" << parameter->shape().ToString()
            << " split_dim=" << split_dim << " offset_=" << offset_->name()
            << " Create split_slice: get_tuple_parameter.name="
            << get_tuple_parameter->name() << " get_tuple_parameter.shape="
            << get_tuple_parameter->shape().ToString()
            << " slice_shape=" << slice_shape.ToString();

  bool merge_rest = GetDebugOptionsFromFlags().xla_tensor_split_merge_rest();
  int64_t split_dim_size = get_tuple_parameter->shape().dimensions(split_dim);
  int64_t main_split_size = int64_t(split_dim_size / split_size) * split_size;
  if (merge_rest && main_split_size < split_dim_size) {
    // need padding to (split_count+1)*split_size
    int64_t padded_split_dim_size = main_split_size + split_size;
    PaddingConfig padding;
    Shape padded_shape =
        ShapeUtil::MakeShape(get_tuple_parameter->shape().element_type(),
                             get_tuple_parameter->shape().dimensions());
    padded_shape.set_dimensions(split_dim, padded_split_dim_size);
    for (int dim = 0; dim < get_tuple_parameter->shape().dimensions_size();
         ++dim) {
      PaddingConfig::PaddingConfigDimension* dimension =
          padding.add_dimensions();
      dimension->set_edge_padding_low(0);
      if (dim == split_dim) {
        dimension->set_edge_padding_high(padded_split_dim_size -
                                         split_dim_size);
      } else {
        dimension->set_edge_padding_high(0);
      }
      dimension->set_interior_padding(0);
    }
    LOG(INFO) << "[SplitParameter] "
              << "get_tuple_parameter.name=" << get_tuple_parameter->name()
              << " split_dim=" << split_dim
              << " split_dim_size=" << split_dim_size
              << " get_tuple_parameter.shpe="
              << get_tuple_parameter->shape().ToString();
    HloInstruction* zero =
        builder_.AddInstruction(HloInstruction::CreateConstant(
            LiteralUtil::Zero(padded_shape.element_type())));
    get_tuple_parameter = builder_.AddInstruction(HloInstruction::CreatePad(
        padded_shape, get_tuple_parameter, zero, padding));
    LOG(INFO) << "[SplitParameter] "
              << "After Padding: get_tuple_parameter.name="
              << get_tuple_parameter->name() << " split_dim=" << split_dim
              << " split_dim_size=" << split_dim_size
              << " get_tuple_parameter.shpe="
              << get_tuple_parameter->shape().ToString();
  }

  HloInstruction* split_slice =
      builder_.AddInstruction(HloInstruction::CreateDynamicSlice(
          slice_shape, get_tuple_parameter, absl::MakeSpan(start_indices),
          slice_shape.dimensions()));

  if (merge_rest && main_split_size < split_dim_size) {
    HloInstruction* orig_split_dim_size =
        builder_.AddInstruction(HloInstruction::CreateConstant(
            LiteralUtil::CreateR0<int32_t>(split_dim_size)));
    HloInstruction* split_size_const =
        builder_.AddInstruction(HloInstruction::CreateConstant(
            LiteralUtil::CreateR0<int32_t>(split_size)));
    // offset starts form zero, so
    // dynamic_size=min(orig_split_dim_size-offset_,split_size_const)
    HloInstruction* tmp_split_size = builder_.AddInstruction(
        HloInstruction::CreateBinary(offset_->shape(), HloOpcode::kSubtract,
                                     orig_split_dim_size, offset_));
    HloInstruction* dynamic_size = builder_.AddInstruction(
        HloInstruction::CreateBinary(offset_->shape(), HloOpcode::kMinimum,
                                     tmp_split_size, split_size_const));
    std::vector<bool> dynamic_dimensions(split_slice->shape().dimensions_size(),
                                         false);
    dynamic_dimensions[split_dim] = true;
    Shape dynamic_split_slice_shape = ShapeUtil::MakeShape(
        split_slice->shape().element_type(), split_slice->shape().dimensions(),
        dynamic_dimensions);
    split_slice =
        builder_.AddInstruction(HloInstruction::CreateSetDimensionSize(
            dynamic_split_slice_shape, split_slice, dynamic_size, split_dim));
  }
  LOG(INFO) << "[SplitParameter] "
            << "new_parameter=" << split_slice->ToString();
  return split_slice;
}

StatusOr<HloInstruction*> TensorSplitterRewriteVisitor::Splitter::SplitLeafIota(
    HloInstruction* iota, int64_t split_dim, int64_t split_size) {
  CHECK(Match(iota, m::Iota()));

  // For an iota, we simply produce smaller iota and add the
  // loop offset to each parameter
  LOG(INFO) << "[SplitLeafParameter] "
            << "iota=" << iota->ToString() << " split_dim=" << split_dim
            << " split_size=" << split_size;

  auto* iota_inst = DynCast<HloIotaInstruction>(iota);
  CHECK(iota_inst != nullptr);

  int64_t parameter_idx = 0;
  Shape iota_shape = ShapeUtil::MakeShape(iota->shape().element_type(),
                                          iota->shape().dimensions());
  iota_shape.set_dimensions(split_dim, split_size);
  bool merge_rest = GetDebugOptionsFromFlags().xla_tensor_split_merge_rest();
  int64_t split_dim_size = iota->shape().dimensions(split_dim);
  int64_t main_split_size = int64_t(split_dim_size / split_size) * split_size;
  HloInstruction* new_iota_inst;

  if (split_dim == iota_inst->iota_dimension()) {
    // The split is along the iota dimension, create offsets add
    // to a single internal iota
    HloInstruction* small_iota = builder_.AddInstruction(
        HloInstruction::CreateIota(iota_shape, iota_inst->iota_dimension()));

    HloInstruction* param;
    if (!ShapeUtil::SameElementType(offset_->shape(), small_iota->shape())) {
      Shape convert_shape = ShapeUtil::MakeShape(
          small_iota->shape().element_type(), offset_->shape().dimensions());
      param = builder_.AddInstruction(
          HloInstruction::CreateConvert(convert_shape, offset_));
    } else {
      param = offset_;
    }

    std::vector<int64_t> broadcast_dims = {};
    HloInstruction* broadcast =
        builder_.AddInstruction(HloInstruction::CreateBroadcast(
            iota_shape, param, absl::MakeSpan(broadcast_dims)));
    new_iota_inst = builder_.AddInstruction(HloInstruction::CreateBinary(
        iota_shape, HloOpcode::kAdd, small_iota, broadcast));
  } else {
    // The split is not along an iota dimension, simply
    // create a smaller iota and add that as parameters.
    new_iota_inst = builder_.AddInstruction(
        HloInstruction::CreateIota(iota_shape, iota_inst->iota_dimension()));
  }
  if (merge_rest && main_split_size < split_dim_size) {
    HloInstruction* orig_split_dim_size =
        builder_.AddInstruction(HloInstruction::CreateConstant(
            LiteralUtil::CreateR0<int32_t>(split_dim_size)));
    HloInstruction* split_size_const =
        builder_.AddInstruction(HloInstruction::CreateConstant(
            LiteralUtil::CreateR0<int32_t>(split_size)));
    // offset starts form zero, so
    // dynamic_size=min(orig_split_dim_size-offset_,split_size_const)
    HloInstruction* tmp_split_size = builder_.AddInstruction(
        HloInstruction::CreateBinary(offset_->shape(), HloOpcode::kSubtract,
                                     orig_split_dim_size, offset_));
    HloInstruction* dynamic_size = builder_.AddInstruction(
        HloInstruction::CreateBinary(offset_->shape(), HloOpcode::kMinimum,
                                     tmp_split_size, split_size_const));
    std::vector<bool> dynamic_dimensions(
        new_iota_inst->shape().dimensions_size(), false);
    dynamic_dimensions[split_dim] = true;
    Shape dynamic_parameter_shape = ShapeUtil::MakeShape(
        new_iota_inst->shape().element_type(),
        new_iota_inst->shape().dimensions(), dynamic_dimensions);
    new_iota_inst =
        builder_.AddInstruction(HloInstruction::CreateSetDimensionSize(
            dynamic_parameter_shape, new_iota_inst, dynamic_size, split_dim));
  }
  LOG(INFO) << "[SplitParameter] "
            << "new_iota_inst=" << new_iota_inst->ToString();
  return new_iota_inst;
}

int64_t TensorSplitterRewriteVisitor::Splitter::BuildRestOutputTuple(
    int64_t split_dim, int64_t split_size, HloInstruction* original,
    HloInstruction* part, bool combine_with_sum, bool combine_with_reduce) {
  HloInstruction* output;
  int64_t output_idx;
  if (combine_with_reduce) {
    CHECK(original->opcode() == HloOpcode::kReduce);
    CHECK(original->operand_count() == 2);
    // re-use reduce init for output init
    HloInstruction* output_init = original->mutable_operand(1);
    if (!ShapeUtil::IsScalar(original->shape())) {
      CHECK(ShapeUtil::IsScalar(output_init->shape()));
      output_init = original->parent()->AddInstruction(
          HloInstruction::CreateBroadcast(original->shape(), output_init, {}));
    }
    output_idx = AddParameter(output_init, &output);
  } else {
    // create the output init (broadcast off of 0)
    HloInstruction* output_init =
        original->parent()->AddInstruction(HloInstruction::CreateConstant(
            LiteralUtil::Zero(original->shape().element_type())));
    output_init = original->parent()->AddInstruction(
        HloInstruction::CreateBroadcast(original->shape(), output_init, {}));
    output_idx = AddParameter(output_init, &output);
  }

  HloInstruction* updated_output;
  if (combine_with_sum) {
    // we're splitting a dot on a dot dimension, this means
    // all that needs to be done is adding the part onto the
    // result (which is initialized as 0)
    updated_output = builder_.AddInstruction(HloInstruction::CreateBinary(
        output->shape(), HloOpcode::kAdd, output, part));
  } else if (combine_with_reduce) {
    // we're splitting on a reduced dimension
    CHECK(original->opcode() == HloOpcode::kReduce);
    CHECK(original->operand_count() == 2);
    HloComputation* reduce_fn = original->called_computations()[0];
    if (ShapeUtil::IsScalar(output->shape())) {
      // we can call the function directly
      updated_output = builder_.AddInstruction(HloInstruction::CreateCall(
          original->shape(), {output, part}, reduce_fn));
    } else {
      // we have to call the function through map
      updated_output = builder_.AddInstruction(HloInstruction::CreateMap(
          original->shape(), {output, part}, reduce_fn));
    }
  } else {
    // slice part onto output
    std::vector<HloInstruction*> start_indices;
    for (int64_t dim = 0; dim < output->shape().dimensions_size(); dim++) {
      if (dim == split_dim) {
        start_indices.push_back(offset_);
      } else {
        start_indices.push_back(
            builder_.AddInstruction(CREATE_CONSTANT_INT32(0)));
      }
    }
    updated_output =
        builder_.AddInstruction(HloInstruction::CreateDynamicUpdateSlice(
            output->shape(), output, part, start_indices));
  }
  LOG(INFO) << "[BuildRestOutputTuple] "
            << "original.name=" << original->name()
            << " offset_=" << offset_->name()
            << " Create updated_output=" << updated_output->name()
            << " shape=" << updated_output->shape().ToString();
  // use the updated_output directly in the remainder in order to
  // avoid cycles
  while_loop_info->rest_starting_node_inst_to_cloned_inst[original] =
      updated_output;

  // since this is for rest function call, doesn't have to have the same shape
  // with parameters, so just add the real output
  std::vector<HloInstruction*>& output_elements = while_loop_output_elements;
  output_elements.push_back(updated_output);
  return output_elements.size() - 1;
}

int64_t TensorSplitterRewriteVisitor::Splitter::BuildMergedLoopOutput(
    int64_t split_dim, int64_t split_size, HloInstruction* original,
    HloInstruction* part, bool combine_with_sum, bool combine_with_reduce) {
  HloInstruction* output;
  int64_t output_idx;
  bool merge_rest = GetDebugOptionsFromFlags().xla_tensor_split_merge_rest();
  if (combine_with_reduce) {
    CHECK(original->opcode() == HloOpcode::kReduce);
    CHECK(original->operand_count() == 2);
    // re-use reduce init for output init
    HloInstruction* output_init = original->mutable_operand(1);
    if (!ShapeUtil::IsScalar(original->shape())) {
      CHECK(ShapeUtil::IsScalar(output_init->shape()));
      output_init = original->parent()->AddInstruction(
          HloInstruction::CreateBroadcast(original->shape(), output_init, {}));
    }
    output_idx = AddParameter(output_init, &output);
  } else {
    if (combine_with_sum || (!merge_rest)) {
      // create the output init (broadcast off of 0)
      HloInstruction* output_init =
          original->parent()->AddInstruction(HloInstruction::CreateConstant(
              LiteralUtil::Zero(original->shape().element_type())));
      output_init = original->parent()->AddInstruction(
          HloInstruction::CreateBroadcast(original->shape(), output_init, {}));
      output_idx = AddParameter(output_init, &output);
    } else {
      // create the output init (broadcast off of 0)
      HloInstruction* output_init =
          original->parent()->AddInstruction(HloInstruction::CreateConstant(
              LiteralUtil::Zero(original->shape().element_type())));
      Shape padded_shape = ShapeUtil::MakeShape(
          original->shape().element_type(), original->shape().dimensions());
      int64_t main_split_size =
          int64_t(original->shape().dimensions(split_dim) / split_size) *
          split_size;
      if (main_split_size < original->shape().dimensions(split_dim)) {
        main_split_size += split_size;
      }
      padded_shape.set_dimensions(split_dim, main_split_size);
      output_init = original->parent()->AddInstruction(
          HloInstruction::CreateBroadcast(padded_shape, output_init, {}));
      output_idx = AddParameter(output_init, &output);
    }
  }

  HloInstruction* updated_output;
  if (combine_with_sum) {
    // we're splitting a dot on a dot dimension, this means
    // all that needs to be done is adding the part onto the
    // result (which is initialized as 0)
    updated_output = builder_.AddInstruction(HloInstruction::CreateBinary(
        output->shape(), HloOpcode::kAdd, output, part));
  } else if (combine_with_reduce) {
    // we're splitting on a reduced dimension
    CHECK(original->opcode() == HloOpcode::kReduce);
    CHECK(original->operand_count() == 2);
    HloComputation* reduce_fn = original->called_computations()[0];
    if (ShapeUtil::IsScalar(output->shape())) {
      // we can call the function directly
      updated_output = builder_.AddInstruction(HloInstruction::CreateCall(
          original->shape(), {output, part}, reduce_fn));
    } else {
      // we have to call the function through map
      updated_output = builder_.AddInstruction(HloInstruction::CreateMap(
          original->shape(), {output, part}, reduce_fn));
    }
  } else {
    // slice part onto output
    std::vector<HloInstruction*> start_indices;
    for (int64_t dim = 0; dim < output->shape().dimensions_size(); dim++) {
      if (dim == split_dim) {
        start_indices.push_back(offset_);
      } else {
        start_indices.push_back(
            builder_.AddInstruction(CREATE_CONSTANT_INT32(0)));
      }
    }
    updated_output =
        builder_.AddInstruction(HloInstruction::CreateDynamicUpdateSlice(
            output->shape(), output, part, start_indices));
  }
  LOG(INFO) << "[BuildMergedLoopOutput] "
            << "original.name=" << original->name()
            << " offset_=" << offset_->name()
            << " Create updated_output=" << updated_output->name()
            << " shape=" << updated_output->shape().ToString();
  // use the updated_output directly in the whilel_loop in order to
  // avoid cycles
  while_loop_info->starting_node_inst_to_cloned_inst[original] = updated_output;

  // add split size to index
  std::vector<HloInstruction*>& output_elements = while_loop_output_elements;
  int64_t output_index_offset = 0;
  if (param_start_index == 0) {
    HloInstruction* split_size_const =
        builder_.AddInstruction(CREATE_CONSTANT_INT32(split_size));
    HloInstruction* updated_index =
        builder_.AddInstruction(HloInstruction::CreateBinary(
            offset_->shape(), HloOpcode::kAdd, offset_, split_size_const));
    output_elements.push_back(updated_index);
    // need to skip current offset
    output_index_offset = 1;
  } else {
    output_index_offset = 0;
  }

  // collect idx, output and all parameters into a tuple ..

  for (int64_t i = param_start_index + output_index_offset;
       i < param_->shape().tuple_shapes_size(); i++) {
    if (i != output_idx) {
      HloInstruction* get_tuple =
          builder_.AddInstruction(HloInstruction::CreateGetTupleElement(
              param_->shape().tuple_shapes(i), param_, i));
      output_elements.push_back(get_tuple);
    } else {
      output_elements.push_back(updated_output);
    }
  }
  return output_elements.size() - 1;
}

HloComputation* TensorSplitterRewriteVisitor::CreateWhileSplitCondition(
    const std::string& name, const Shape& parameters_shape, int64_t stop_at) {
  HloComputation::Builder builder(name);
  HloInstruction* parameter = builder.AddInstruction(
      HloInstruction::CreateParameter(0, parameters_shape, "loop_param"));
  HloInstruction* iteration =
      builder.AddInstruction(HloInstruction::CreateGetTupleElement(
          parameters_shape.tuple_shapes(0), parameter, 0));
  HloInstruction* stop_iteration =
      builder.AddInstruction(CREATE_CONSTANT_INT32(stop_at));
  HloInstruction* compare = builder.AddInstruction(
      HloInstruction::CreateCompare(ShapeUtil::MakeShape(PRED, {}), iteration,
                                    stop_iteration, ComparisonDirection::kLt));
  return parent_module->AddEmbeddedComputation(builder.Build(compare));
}

std::vector<HloInstruction*>
TensorSplitterRewriteVisitor::CreateWhileSplitWithResults(
    HloComputation* parent_comp, HloComputation* condition,
    HloComputation* body, std::vector<HloInstruction*> parameters,
    std::vector<std::tuple<int64_t, Shape>> ids_and_shapes) {
  // Create init and loop
  HloInstruction* init =
      parent_comp->AddInstruction(HloInstruction::CreateTuple(parameters));
  HloInstruction* loop = parent_comp->AddInstruction(
      HloInstruction::CreateWhile(init->shape(), condition, body, init));

  // Extract While results for array and indices
  std::vector<HloInstruction*> results;
  for (auto id_shape : ids_and_shapes) {
    int64_t id;
    Shape shape;
    std::tie(id, shape) = id_shape;
    HloInstruction* result = parent_comp->AddInstruction(
        HloInstruction::CreateGetTupleElement(shape, loop, id));
    results.push_back(result);
  }
  return results;
}

bool TensorSplitterRewriteVisitor::CanFinishMergedWhileLoop(
    int64_t while_loop_num) {
  return while_loop_num_to_processed_count[while_loop_num] ==
         int64_t(while_loop_num_to_start_node[while_loop_num].size());
}

Status TensorSplitterRewriteVisitor::BuildFinalOutput(WhileLoopInfo& loop_info,
                                                      HloInstruction* loop) {
  std::string prefix = "[BuildFinalOutput] While_Loop_Num_" +
                       std::to_string(loop_info.while_loop_num);
  // use two for-loop
  // first for-loop to build sub_rest for all sort with rest
  // sort is special since it needs result from the final while-loop
  for (auto& sub_info : loop_info.final_sub_main_output_elements) {
    HloInstruction* orig_inst = sub_info.starting_node_inst;
    if (orig_inst->opcode() == HloOpcode::kSort) {
      if (sub_info.split_rest > 0) {
        int64_t sub_output_idx = sub_info.result_index;
        HloInstruction* result = orig_inst->parent()->AddInstruction(
            HloInstruction::CreateGetTupleElement(orig_inst->shape(), loop,
                                                  sub_output_idx));
        auto sort_comp = orig_inst->parent();
        HloInstruction* array = orig_inst->mutable_operand(0);
        HloInstruction* indices = orig_inst->mutable_operand(1);
        HloInstruction* get_array;
        HloInstruction* get_indices;
        get_array = orig_inst->users()[0];
        get_indices = orig_inst->users()[1];
        HloInstruction* array_slice = get_array->users()[0];
        HloInstruction* indices_slice = get_indices->users()[0];
        auto array_dims_size = array->shape().dimensions().size();
        auto last_dim = array_dims_size - 1;
        auto slice_k = array_slice->shape().dimensions(last_dim);
        Shape body_acc_array_shape = array_slice->shape();
        Shape body_acc_indices_shape = indices_slice->shape();
        Shape body_acc_shape = ShapeUtil::MakeTupleShape(
            {body_acc_array_shape, body_acc_indices_shape});

        // Extract While results for array and indices
        std::vector<HloInstruction*> sort_results;
        for (auto id_shape : sub_info.sort_ids_and_shapes) {
          int64_t id;
          Shape shape;
          std::tie(id, shape) = id_shape;
          HloInstruction* sort_result = sort_comp->AddInstruction(
              HloInstruction::CreateGetTupleElement(shape, result, id));
          sort_results.push_back(sort_result);
        }
        HloInstruction* while_sort_result = sort_results[0];

        HloInstruction* while_get_array =
            sort_comp->AddInstruction(HloInstruction::CreateGetTupleElement(
                body_acc_shape.tuple_shapes(0), while_sort_result, 0));
        HloInstruction* while_get_indices =
            sort_comp->AddInstruction(HloInstruction::CreateGetTupleElement(
                body_acc_shape.tuple_shapes(1), while_sort_result, 1));
        auto split_dim = sub_info.split_dim;
        CHECK(while_get_array->shape().dimensions(split_dim) == slice_k);
        CHECK(while_get_indices->shape().dimensions(split_dim) == slice_k);

        HloComputation::Builder& rest_builder = *loop_info.while_rest_builder;
        int64_t main_split_size = loop_info.split_count * loop_info.split_size;

        Splitter rest_splitter(
            false, &loop_info, loop_info.while_rest_parameters,
            loop_info.while_rest_output_elements, rest_builder, sort_comp,
            absl::MakeSpan(sub_info.sort_split_leafs),
            while_loop_num_to_visited_instructions[loop_info.while_loop_num],
            this->while_loop_num_to_instructions, main_split_size);

        TF_ASSIGN_OR_RETURN(HloInstruction * rest_array,
                            rest_splitter.SplitInstruction(
                                array, split_dim, sub_info.split_rest));

        TF_ASSIGN_OR_RETURN(HloInstruction * rest_indices,
                            rest_splitter.SplitInstruction(
                                indices, split_dim, sub_info.split_rest));

        HloInstruction* get_params_array = nullptr;
        HloInstruction* get_params_indices = nullptr;
        rest_splitter.AddParameter(while_get_array, &get_params_array);
        rest_splitter.AddParameter(while_get_indices, &get_params_indices);

        Shape merged_array_shape = while_get_array->shape();
        Shape merged_indices_shape = while_get_indices->shape();
        auto merged_split_dim_size = slice_k + sub_info.split_rest;
        merged_array_shape.set_dimensions(split_dim, merged_split_dim_size);
        merged_indices_shape.set_dimensions(split_dim, merged_split_dim_size);
        auto rest_merged_array =
            rest_builder.AddInstruction(HloInstruction::CreateConcatenate(
                merged_array_shape, {get_params_array, rest_array}, split_dim));
        auto rest_merged_indices =
            rest_builder.AddInstruction(HloInstruction::CreateConcatenate(
                merged_indices_shape, {get_params_indices, rest_indices},
                split_dim));

        Shape rest_sort_shape = ShapeUtil::MakeTupleShape(
            {rest_merged_array->shape(), rest_merged_indices->shape()});
        auto rest_sort =
            rest_builder.AddInstruction(orig_inst->CloneWithNewOperands(
                rest_sort_shape, {rest_merged_array, rest_merged_indices}));

        int64_t rest_idx = loop_info.while_rest_output_elements.size();
        loop_info.while_rest_output_elements.push_back(rest_sort);
        // store rest_result_index for the sort
        sub_info.result_rest_index = rest_idx;
      }
    }
  }

  bool merge_rest = GetDebugOptionsFromFlags().xla_tensor_split_merge_rest();
  HloInstruction* rest_result = nullptr;
  // TODO: may be need to check if there is a sort in the loop, need to build
  // rest
  if (loop_info.split_rest > 0 && (!merge_rest)) {
    // after the first loop, we can build the fianal rest computation
    HloInstruction* rest_output_tuple =
        loop_info.while_rest_builder->AddInstruction(
            HloInstruction::CreateTuple(loop_info.while_rest_output_elements));
    LOG(INFO) << prefix << "Final restoutput_tuple.shape="
              << rest_output_tuple->shape().ToString();
    HloComputation* rest_body = parent_module->AddEmbeddedComputation(
        loop_info.while_rest_builder->Build(rest_output_tuple));
    int64_t main_split_size = loop_info.split_count * loop_info.split_size;
    HloComputation* parent_comp =
        loop_info.final_sub_main_output_elements.front()
            .starting_node_inst->parent();
    loop_info.while_rest_parameters[0] =
        parent_comp->AddInstruction(CREATE_CONSTANT_INT32(main_split_size));
    HloInstruction* args = parent_comp->AddInstruction(
        HloInstruction::CreateTuple(loop_info.while_rest_parameters));
    rest_result = parent_comp->AddInstruction(HloInstruction::CreateCall(
        rest_output_tuple->shape(), {args}, rest_body));
  } else {
    LOG(INFO) << prefix << " has no split_rest";
  }

  // second for-loop to build final output
  for (auto& sub_info : loop_info.final_sub_main_output_elements) {
    HloInstruction* orig_inst = sub_info.starting_node_inst;
    int64_t sub_output_idx = sub_info.result_index;

    Shape result_shape =
        ShapeUtil::GetTupleElementShape(loop->shape(), sub_output_idx);
    HloInstruction* result = orig_inst->parent()->AddInstruction(
        HloInstruction::CreateGetTupleElement(result_shape, loop,
                                              sub_output_idx));

    LOG(INFO) << prefix << " result=" << result->ToString()
              << " sub_output_idx=" << std::to_string(sub_output_idx)
              << " parent.name()=" << orig_inst->parent()->name()
              << " shape=" << result->shape().ToString();
    if (loop_info.split_rest != sub_info.split_rest) {
      continue;
    }
    if (orig_inst->opcode() == HloOpcode::kDot) {
      if (sub_info.split_rest == 0 ||
          (sub_info.combine_parts_with_sum && merge_rest)) {
        LOG(INFO) << prefix << " Start Replace "
                  << "'" << orig_inst->name() << "' with"
                  << " '" << result->name() << "' ";
        TF_RETURN_IF_ERROR(ReplaceInstruction(orig_inst, result));
        LOG(INFO) << prefix << " Finish "
                  << "'" << orig_inst->name() << "' with"
                  << " '" << result->name() << "' ";
      } else if (merge_rest) {
        Shape slice_shape = ShapeUtil::MakeShape(
            orig_inst->shape().element_type(), orig_inst->shape().dimensions());
        std::vector<int64_t> starts;
        std::vector<int64_t> limits;
        std::vector<int64_t> strides;
        for (int64_t d = 0; d < orig_inst->shape().dimensions_size(); d++) {
          strides.push_back(1);
          starts.push_back(0);
          limits.push_back(orig_inst->shape().dimensions(d));
        }
        HloInstruction* full_result =
            orig_inst->parent()->AddInstruction(HloInstruction::CreateSlice(
                slice_shape, result, absl::MakeSpan(starts),
                absl::MakeSpan(limits), absl::MakeSpan(strides)));
        TF_RETURN_IF_ERROR(ReplaceInstruction(orig_inst, full_result));
        LOG(INFO) << prefix << " Replace "
                  << "'" << orig_inst->name() << "' with"
                  << " '" << full_result->name() << "' ";
      } else {
        // get rest_result for the dot
        sub_info.result_rest = orig_inst->parent()->AddInstruction(
            HloInstruction::CreateGetTupleElement(
                rest_result->shape().tuple_shapes(sub_info.result_rest_index),
                rest_result, sub_info.result_rest_index));
        if (sub_info.combine_parts_with_sum) {
          HloInstruction* full_result = orig_inst->parent()->AddInstruction(
              HloInstruction::CreateBinary(result->shape(), HloOpcode::kAdd,
                                           result, sub_info.result_rest));
          TF_RETURN_IF_ERROR(ReplaceInstruction(orig_inst, full_result));
          LOG(INFO) << prefix << " Replace "
                    << "'" << orig_inst->name() << "' with"
                    << " '" << full_result->name() << "' ";
        } else {
          int64_t dot_split_dim =
              sub_info.dot_split_dim;  // split dimension after dot occured
          Shape slice_shape = ShapeUtil::MakeShape(
              result->shape().element_type(), result->shape().dimensions());
          int64_t main_split_size =
              loop_info.split_count * loop_info.split_size;
          slice_shape.set_dimensions(dot_split_dim, main_split_size);
          std::vector<int64_t> starts;
          std::vector<int64_t> limits;
          std::vector<int64_t> strides;
          for (int64_t d = 0; d < orig_inst->shape().dimensions_size(); d++) {
            strides.push_back(1);
            starts.push_back(0);
            if (d == dot_split_dim) {
              limits.push_back(main_split_size);
            } else {
              limits.push_back(orig_inst->shape().dimensions(d));
            }
          }
          HloInstruction* result_slice =
              orig_inst->parent()->AddInstruction(HloInstruction::CreateSlice(
                  slice_shape, result, absl::MakeSpan(starts),
                  absl::MakeSpan(limits), absl::MakeSpan(strides)));
          HloInstruction* full_result = orig_inst->parent()->AddInstruction(
              HloInstruction::CreateConcatenate(
                  orig_inst->shape(), {result_slice, sub_info.result_rest},
                  dot_split_dim));

          TF_RETURN_IF_ERROR(ReplaceInstruction(orig_inst, full_result));
        }
      }
    } else if (orig_inst->opcode() == HloOpcode::kReduce ||
               orig_inst->opcode() == HloOpcode::kGetTupleElement) {
      if (sub_info.split_rest == 0 ||
          (sub_info.split_along_reduce_dim && merge_rest)) {
        std::stringstream msg;
        for (auto user : orig_inst->users()) {
          msg << user->name() << " ";
        }
        LOG(INFO) << prefix << " orig_inst: "
                  << "'" << orig_inst->name() << "' users: " << msg.str();
        TF_RETURN_IF_ERROR(ReplaceInstruction(orig_inst, result));
        LOG(INFO) << prefix << " Replace "
                  << "'" << orig_inst->name() << "' with"
                  << " '" << result->name() << "' ";
        msg.str("");
        for (auto user : result->users()) {
          msg << user->name() << " ";
        }
        LOG(INFO) << prefix << " new_inst: "
                  << "'" << result->name() << "' users: " << msg.str();
      } else if (merge_rest) {
        HloInstruction* old_output;
        if (orig_inst->shape().IsTuple()) {
          CHECK(orig_inst->user_count() == 1);
          old_output = orig_inst->users()[0];

        } else {
          old_output = orig_inst;
        }
        Shape slice_shape =
            ShapeUtil::MakeShape(old_output->shape().element_type(),
                                 old_output->shape().dimensions());
        std::vector<int64_t> starts;
        std::vector<int64_t> limits;
        std::vector<int64_t> strides;
        for (int64_t d = 0; d < old_output->shape().dimensions_size(); d++) {
          strides.push_back(1);
          starts.push_back(0);
          limits.push_back(old_output->shape().dimensions(d));
        }
        HloInstruction* full_result =
            orig_inst->parent()->AddInstruction(HloInstruction::CreateSlice(
                slice_shape, result, absl::MakeSpan(starts),
                absl::MakeSpan(limits), absl::MakeSpan(strides)));
        std::stringstream msg;
        for (auto user : orig_inst->users()) {
          msg << user->name() << " ";
        }
        LOG(INFO) << prefix << " orig_inst: "
                  << "'" << orig_inst->name() << "' users: " << msg.str();

        TF_RETURN_IF_ERROR(ReplaceInstruction(orig_inst, full_result));
        LOG(INFO) << prefix << " Replace "
                  << "'" << orig_inst->name() << "' with"
                  << " '" << full_result->name() << "' ";

        msg.str("");
        for (auto user : full_result->users()) {
          msg << user->name() << " ";
        }
        LOG(INFO) << prefix << " new_inst: "
                  << "'" << full_result->name() << "' users: " << msg.str();
      } else {
        // get rest_result for the reduce
        sub_info.result_rest = orig_inst->parent()->AddInstruction(
            HloInstruction::CreateGetTupleElement(
                rest_result->shape().tuple_shapes(sub_info.result_rest_index),
                rest_result, sub_info.result_rest_index));
        HloInstruction* full_result = result;
        if (sub_info.split_along_reduce_dim) {
          CHECK(orig_inst->opcode() == HloOpcode::kReduce);
          CHECK(orig_inst->operand_count() == 2);
          HloComputation* reduce_fn = orig_inst->called_computations()[0];
          if (ShapeUtil::IsScalar(sub_info.result_rest->shape())) {
            // we can call the function directly
            full_result =
                orig_inst->parent()->AddInstruction(HloInstruction::CreateCall(
                    orig_inst->shape(), {result, sub_info.result_rest},
                    reduce_fn));
          } else {
            // we have to call the function through map
            full_result =
                orig_inst->parent()->AddInstruction(HloInstruction::CreateMap(
                    orig_inst->shape(), {result, sub_info.result_rest},
                    reduce_fn));
          }
        } else {
          HloInstruction* old_output;
          if (orig_inst->shape().IsTuple()) {
            CHECK(orig_inst->user_count() == 1);
            old_output = orig_inst->users()[0];

          } else {
            old_output = orig_inst;
          }
          int64_t reduce_split_dim = sub_info.split_dim;
          for (int64_t r_dim : orig_inst->dimensions()) {
            if (r_dim < sub_info.split_dim) {
              reduce_split_dim--;
            }
          }
          int64_t main_split_size =
              loop_info.split_count * loop_info.split_size;
          auto dim_len = old_output->shape().dimensions_size();
          Shape slice_shape = ShapeUtil::MakeShape(
              result->shape().element_type(), result->shape().dimensions());
          slice_shape.set_dimensions(reduce_split_dim, main_split_size);
          std::vector<int64_t> starts;
          std::vector<int64_t> limits;
          std::vector<int64_t> strides;

          for (int64_t d = 0; d < dim_len; d++) {
            strides.push_back(1);
            starts.push_back(0);
            if (d == reduce_split_dim) {
              limits.push_back(main_split_size);
            } else {
              limits.push_back(old_output->shape().dimensions(d));
            }
          }

          HloInstruction* result_slice =
              orig_inst->parent()->AddInstruction(HloInstruction::CreateSlice(
                  slice_shape, result, absl::MakeSpan(starts),
                  absl::MakeSpan(limits), absl::MakeSpan(strides)));

          Shape slice_shape_rest =
              ShapeUtil::MakeShape(sub_info.result_rest->shape().element_type(),
                                   sub_info.result_rest->shape().dimensions());
          slice_shape_rest.set_dimensions(reduce_split_dim,
                                          sub_info.split_rest);
          std::vector<int64_t> starts_rest;
          std::vector<int64_t> limits_rest;
          std::vector<int64_t> strides_rest;

          for (int64_t d = 0; d < dim_len; d++) {
            strides_rest.push_back(1);
            auto full_size = old_output->shape().dimensions(d);
            limits_rest.push_back(full_size);
            if (d == reduce_split_dim) {
              starts_rest.push_back(main_split_size);
            } else {
              starts_rest.push_back(0);
            }
          }

          HloInstruction* result_rest_slice =
              orig_inst->parent()->AddInstruction(HloInstruction::CreateSlice(
                  slice_shape_rest, sub_info.result_rest,
                  absl::MakeSpan(starts_rest), absl::MakeSpan(limits_rest),
                  absl::MakeSpan(strides_rest)));
          full_result = orig_inst->parent()->AddInstruction(
              HloInstruction::CreateConcatenate(
                  orig_inst->shape(), {result_slice, result_rest_slice},
                  reduce_split_dim));
        }

        TF_RETURN_IF_ERROR(ReplaceInstruction(orig_inst, full_result));
        LOG(INFO) << prefix << " Replace "
                  << "'" << orig_inst->name() << "' with"
                  << " '" << full_result->name() << "' ";
      }
    } else if (orig_inst->opcode() == HloOpcode::kSort) {
      auto sort_comp = orig_inst->parent();
      HloInstruction* array = orig_inst->mutable_operand(0);
      HloInstruction* indices = orig_inst->mutable_operand(1);
      HloInstruction* get_array;
      HloInstruction* get_indices;
      get_array = orig_inst->users()[0];
      get_indices = orig_inst->users()[1];
      HloInstruction* array_slice = get_array->users()[0];
      HloInstruction* indices_slice = get_indices->users()[0];
      auto array_dims_size = array->shape().dimensions().size();
      auto last_dim = array_dims_size - 1;
      auto slice_k = array_slice->shape().dimensions(last_dim);
      Shape body_acc_array_shape = array_slice->shape();
      Shape body_acc_indices_shape = indices_slice->shape();
      Shape body_acc_shape = ShapeUtil::MakeTupleShape(
          {body_acc_array_shape, body_acc_indices_shape});

      // Extract While results for array and indices
      std::vector<HloInstruction*> sort_results;
      for (auto id_shape : sub_info.sort_ids_and_shapes) {
        int64_t id;
        Shape shape;
        std::tie(id, shape) = id_shape;
        HloInstruction* sort_result = sort_comp->AddInstruction(
            HloInstruction::CreateGetTupleElement(shape, result, id));
        sort_results.push_back(sort_result);
      }
      HloInstruction* while_sort_result = sort_results[0];

      HloInstruction* while_get_array =
          sort_comp->AddInstruction(HloInstruction::CreateGetTupleElement(
              body_acc_shape.tuple_shapes(0), while_sort_result, 0));
      HloInstruction* while_get_indices =
          sort_comp->AddInstruction(HloInstruction::CreateGetTupleElement(
              body_acc_shape.tuple_shapes(1), while_sort_result, 1));
      auto split_dim = sub_info.split_dim;
      CHECK(while_get_array->shape().dimensions(split_dim) == slice_k);
      CHECK(while_get_indices->shape().dimensions(split_dim) == slice_k);

      if (sub_info.split_rest == 0) {
        TF_RETURN_IF_ERROR(sort_comp->ReplaceInstructionWithDifferentShape(
            array_slice, while_get_array));
        TF_RETURN_IF_ERROR(sort_comp->ReplaceInstructionWithDifferentShape(
            indices_slice, while_get_indices));
      } else {
        // get rest_result for the sort
        sub_info.result_rest = orig_inst->parent()->AddInstruction(
            HloInstruction::CreateGetTupleElement(
                rest_result->shape().tuple_shapes(sub_info.result_rest_index),
                rest_result, sub_info.result_rest_index));
        auto rest_get_array =
            sort_comp->AddInstruction(HloInstruction::CreateGetTupleElement(
                sub_info.result_rest->shape().tuple_shapes(0),
                sub_info.result_rest, 0));
        auto rest_get_indices =
            sort_comp->AddInstruction(HloInstruction::CreateGetTupleElement(
                sub_info.result_rest->shape().tuple_shapes(1),
                sub_info.result_rest, 1));

        auto rest_array_slice =
            sort_comp->AddInstruction(array_slice->CloneWithNewOperands(
                array_slice->shape(), {rest_get_array}));
        auto rest_indices_slice =
            sort_comp->AddInstruction(indices_slice->CloneWithNewOperands(
                indices_slice->shape(), {rest_get_indices}));

        TF_RETURN_IF_ERROR(sort_comp->ReplaceInstructionWithDifferentShape(
            array_slice, rest_array_slice));
        TF_RETURN_IF_ERROR(sort_comp->ReplaceInstructionWithDifferentShape(
            indices_slice, rest_indices_slice));
      }
    } else {
      LOG(ERROR) << "Unimplemented OP" << sub_info.starting_node_inst->name();
      CHECK(false);
    }
  }
  LOG(INFO) << prefix << " Finish building "
            << "loop.parent = " << loop->parent()->name()
            << " instructions.size=" << loop->parent()->instruction_count();
  // LOG(INFO) << prefix << " Final computation:\n" <<
  // loop->parent()->ToString();

  return OkStatus();
}

Status TensorSplitterRewriteVisitor::FinalizeMergedWhileLoop(
    int64_t while_loop_num) {
  CHECK(while_loop_num_to_processed_count[while_loop_num] ==
        int64_t(while_loop_num_to_start_node[while_loop_num].size()));
  std::string prefix = "[FinalizeMergedWhileLoop] MergedWhileLoop_" +
                       std::to_string(while_loop_num) + " ";
  WhileLoopInfo& loop_info = while_loop_num_to_info[while_loop_num];

  HloComputation::Builder& body_builder = *loop_info.while_builder;
  HloComputation* while_parent_comp =
      loop_info.final_sub_main_output_elements.front()
          .starting_node_inst->parent();

  HloInstruction* output_tuple = body_builder.AddInstruction(
      HloInstruction::CreateTuple(loop_info.while_loop_output_elements));
  HloComputation* body =
      parent_module->AddEmbeddedComputation(body_builder.Build(output_tuple));
  LOG(INFO) << prefix << "Create body body first param.name="
            << body->parameter_instruction(0)->name()
            << " body first param.shape="
            << body->parameter_instruction(0)->shape().ToString()
            << " body.root_instruction.name" << body->root_instruction()->name()
            << " body.root_instruction.shape"
            << body->root_instruction()->shape().ToString();
  HloComputation::Builder cond_builder(
      "merged_while_loop_" + std::to_string(while_loop_num) + "_cond");
  HloInstruction* cond_param =
      cond_builder.AddInstruction(HloInstruction::CreateParameter(
          0, output_tuple->shape(), "loop_cond_param"));

  LOG(INFO) << prefix << " Create a cond_param=" << cond_param->name()
            << " shape=" << cond_param->shape().ToString();
  // use the first element of the first sub_output as the cond_offset
  HloInstruction* cond_offset =
      cond_builder.AddInstruction(HloInstruction::CreateGetTupleElement(
          output_tuple->shape().tuple_shapes(0), cond_param, 0));
  LOG(INFO) << prefix << " cond_offset.name()=" << cond_offset->name()
            << " shape=" << cond_offset->shape().ToString();

  int64_t main_split_size = loop_info.split_count * loop_info.split_size;
  bool merge_rest = GetDebugOptionsFromFlags().xla_tensor_split_merge_rest();
  if (merge_rest && loop_info.split_rest > 0) {
    main_split_size += loop_info.split_size;
  }
  LOG(INFO) << prefix << " main_split_size=" << main_split_size;
  HloInstruction* offset_less_than =
      cond_builder.AddInstruction(CREATE_CONSTANT_INT32(main_split_size));
  HloInstruction* compare =
      cond_builder.AddInstruction(HloInstruction::CreateCompare(
          ShapeUtil::MakeShape(PRED, {}), cond_offset, offset_less_than,
          ComparisonDirection::kLt));
  HloComputation* cond =
      parent_module->AddEmbeddedComputation(cond_builder.Build(compare));
  HloInstruction* init = while_parent_comp->AddInstruction(
      HloInstruction::CreateTuple(loop_info.while_loop_parameters));
  HloInstruction* loop = while_parent_comp->AddInstruction(
      HloInstruction::CreateWhile(output_tuple->shape(), cond, body, init));

  LOG(INFO) << prefix << "Create loop.name=" << loop->name()
            << " parent.name=" << loop->parent()->name()
            << " body instructions.size=" << body->instruction_count()
            << " init.name=" << init->name()
            << " init.shape=" << init->shape().ToString();
  return BuildFinalOutput(loop_info, loop);
}

Status TensorSplitterRewriteVisitor::AddDotToMergedWhileLoop(
    HloInstruction* dot, HloInstruction* lhs, HloInstruction* rhs) {
  std::stringstream ss;
  std::string prefix = "[AddDotToMergedWhileLoop] ";
  ss << prefix;
  auto& dnums = dot->dot_dimension_numbers();
  // TODO: Handle the case where both operands can be
  //       split in a better way.

  // Cases we handle:
  // 1. lhs can split
  // 2. rhs can split
  // 3. lhs = rhs + can split
  // 3.1 lhs + rhs can split on respective contracted dim

  bool rhs_is_lhs = lhs == rhs;

  std::vector<HloInstruction*> split_leafs_lhs;
  std::vector<HloInstruction*> split_leafs_rhs;
  std::vector<int64_t> exclude_dims_lhs;
  std::vector<int64_t> exclude_dims_rhs;

  auto lhs_dim_size = lhs->shape().dimensions_size();
  std::vector<int64_t> original_dims_lhs(lhs_dim_size);
  std::iota(original_dims_lhs.begin(), original_dims_lhs.end(), 0);

  bool can_split_lhs = OperandShouldBeSplit(lhs) &&
                       OperandCanBeSplit(lhs, &split_leafs_lhs,
                                         &original_dims_lhs, &exclude_dims_lhs);

  auto rhs_dim_size = rhs->shape().dimensions_size();
  std::vector<int64_t> original_dims_rhs(rhs_dim_size);
  std::iota(original_dims_rhs.begin(), original_dims_rhs.end(), 0);

  bool can_split_rhs = OperandShouldBeSplit(rhs) &&
                       OperandCanBeSplit(rhs, &split_leafs_rhs,
                                         &original_dims_rhs, &exclude_dims_rhs);

  auto node_key = MakeSplitNodeKey(dot);
  if (can_split_lhs && can_split_rhs && rhs_is_lhs) {
    //
    // Case :: Self dot
    //
    CHECK(dnums.lhs_contracting_dimensions().size() == 1);
    if (dnums.lhs_contracting_dimensions()[0] !=
        dnums.rhs_contracting_dimensions()[0]) {
      return OkStatus();
    }

    int64_t split_dim = dnums.lhs_contracting_dimensions()[0];
    // there must be only one splittable path for this case
    HloInstruction* recorded_start =
        std::get<0>(start_node_to_splittable_paths[node_key][0][1]);
    int64_t recorded_dim =
        std::get<1>(start_node_to_splittable_paths[node_key][0][1]);
    int64_t recorded_size =
        std::get<2>(start_node_to_splittable_paths[node_key][0][1]);
    int64_t split_size, split_count, split_rest;
    std::tie(split_size, split_count, split_rest) =
        DetermineSplitSize(lhs, split_dim);
    if (recorded_start->unique_id() != lhs->unique_id() ||
        recorded_dim != split_dim || recorded_size != split_size) {
      LOG(ERROR) << prefix << "Case 1 Starting_node '" << dot->name()
                 << "' lhs.name=" << lhs->name()
                 << " recorded_lhs.name=" << recorded_start->name()
                 << " recorded_dim=" << recorded_dim
                 << " split_dim=" << split_dim
                 << " recoded_size=" << recorded_size
                 << " split_size=" << split_size;
      CHECK(false);
    }

    int64_t while_loop_num = start_node_to_while_loop_num[node_key];
    if (!while_loop_num_to_info.contains(
            start_node_to_while_loop_num[node_key])) {
      std::unique_ptr<HloComputation::Builder> while_builder(
          new HloComputation::Builder("merged_while_loop_" +
                                      std::to_string(while_loop_num)));
      std::unique_ptr<HloComputation::Builder> while_rest_builder(
          new HloComputation::Builder("merged_rest_while_loop_" +
                                      std::to_string(while_loop_num)));
      while_loop_num_to_info[while_loop_num] = std::move(WhileLoopInfo(
          while_loop_num, split_size, split_count, split_rest,
          std::move(while_builder), std::move(while_rest_builder)));
      while_loop_num_to_processed_count[while_loop_num] = 0;
      while_loop_num_to_visited_instructions[while_loop_num] = {};
    }
    auto main_split_size = split_count * split_size;
    CHECK(main_split_size + split_rest == lhs->shape().dimensions(split_dim));

    ss << "<<< "
       << "Splitting self dot " << dot->name()
       << " operand will be split at dimension " << split_dim
       << " with split size " << split_size << " and rest size " << split_rest;

    HloComputation::Builder& body_builder =
        *while_loop_num_to_info[while_loop_num].while_builder;
    // use the same while_builder for the who merged_while_loop
    Splitter splitter(
        true, &while_loop_num_to_info[while_loop_num],
        while_loop_num_to_info[while_loop_num].while_loop_parameters,
        while_loop_num_to_info[while_loop_num].while_loop_output_elements,
        body_builder, dot->parent(), absl::MakeSpan(split_leafs_lhs),
        while_loop_num_to_visited_instructions[while_loop_num],
        this->while_loop_num_to_instructions);

    TF_ASSIGN_OR_RETURN(HloInstruction * split_lhs,
                        splitter.SplitInstruction(lhs, split_dim, split_size));

    HloInstruction* part = body_builder.AddInstruction(
        dot->CloneWithNewOperands(dot->shape(), {split_lhs, split_lhs}));

    int64_t output_index = splitter.BuildMergedLoopOutput(
        -1, split_size, dot, part, /*combine_with_sum =*/true);
    bool merge_rest = GetDebugOptionsFromFlags().xla_tensor_split_merge_rest();
    if (split_rest == 0 || merge_rest) {
      ss << "\n ----< Exit HandleDot for '" << dot->name() << "' SUCCESS";
      LOG(INFO) << ss.str();
      std::vector<std::tuple<int64_t, Shape>> empty_id_shapes;
      std::vector<HloInstruction*> empty_leafs;
      while_loop_num_to_info[while_loop_num].AddSubOutput(
          SubOutputInfo(output_index, dot, split_dim, empty_id_shapes,
                        empty_leafs, split_rest, -1, false, false));
      while_loop_num_to_processed_count[while_loop_num] += 1;
      if (CanFinishMergedWhileLoop(while_loop_num)) {
        return FinalizeMergedWhileLoop(while_loop_num);
      }
      return OkStatus();
    } else {
      // HloComputation::Builder rest_builder("tensor_splitter_dot_rest");
      HloComputation::Builder& rest_builder =
          *while_loop_num_to_info[while_loop_num].while_rest_builder;
      Splitter splitter(
          false, &while_loop_num_to_info[while_loop_num],
          while_loop_num_to_info[while_loop_num].while_rest_parameters,
          while_loop_num_to_info[while_loop_num].while_rest_output_elements,
          rest_builder, dot->parent(), absl::MakeSpan(split_leafs_lhs),
          while_loop_num_to_visited_instructions[while_loop_num],
          this->while_loop_num_to_instructions, main_split_size);

      TF_ASSIGN_OR_RETURN(
          HloInstruction * rest_lhs,
          splitter.SplitInstruction(lhs, split_dim, split_rest));

      HloInstruction* rest_part = rest_builder.AddInstruction(
          dot->CloneWithNewOperands(dot->shape(), {rest_lhs, rest_lhs}));
      int64_t rest_output_idx = while_loop_num_to_info[while_loop_num]
                                    .while_rest_output_elements.size();
      while_loop_num_to_info[while_loop_num]
          .while_rest_output_elements.push_back(rest_part);

      ss << "\n ----< Exit HandleDot for '" << dot->name() << "' SUCCESS";
      LOG(INFO) << ss.str();
      std::vector<std::tuple<int64_t, Shape>> empty_id_shapes;
      std::vector<HloInstruction*> empty_leafs;
      while_loop_num_to_info[while_loop_num].AddSubOutput(
          SubOutputInfo(output_index, dot, split_dim, empty_id_shapes,
                        empty_leafs, split_rest, rest_output_idx, true, false));
      while_loop_num_to_processed_count[while_loop_num] += 1;
      if (CanFinishMergedWhileLoop(while_loop_num)) {
        return FinalizeMergedWhileLoop(while_loop_num);
      }
      return OkStatus();
    }
  } else if (!rhs_is_lhs && can_split_lhs && can_split_rhs) {
    //
    // CASE :: both lhs and rhs need split
    //
    CHECK(dnums.lhs_contracting_dimensions().size() == 1);

    int64_t split_dim_lhs = dnums.lhs_contracting_dimensions()[0];
    int64_t split_dim_rhs = dnums.rhs_contracting_dimensions()[0];

    CHECK(lhs->shape().dimensions(split_dim_lhs) ==
          rhs->shape().dimensions(split_dim_rhs));

    int64_t split_size, split_count, split_rest;
    std::tie(split_size, split_count, split_rest) =
        DetermineSplitSize(lhs, split_dim_lhs);

    // since lhs and rhs path are recorded into a single path, thus only need
    // to check split_dim_lhs
    HloInstruction* recorded_start =
        std::get<0>(start_node_to_splittable_paths[node_key][0][1]);
    int64_t recorded_dim =
        std::get<1>(start_node_to_splittable_paths[node_key][0][1]);
    int64_t recorded_size =
        std::get<2>(start_node_to_splittable_paths[node_key][0][1]);

    if (recorded_start->unique_id() != lhs->unique_id() ||
        recorded_dim != split_dim_lhs || recorded_size != split_size) {
      LOG(ERROR) << prefix << "Case 2 Starting_node '" << dot->name()
                 << "' lhs.name=" << lhs->name()
                 << " recorded_lhs.name=" << recorded_start->name()
                 << " recorded_dim=" << recorded_dim
                 << " split_dim=" << split_dim_lhs
                 << " recoded_size=" << recorded_size
                 << " split_size=" << split_size;
      CHECK(false);
    }

    int64_t while_loop_num = start_node_to_while_loop_num[node_key];
    if (!while_loop_num_to_info.contains(
            start_node_to_while_loop_num[node_key])) {
      std::unique_ptr<HloComputation::Builder> while_builder(
          new HloComputation::Builder("merged_while_loop_" +
                                      std::to_string(while_loop_num)));
      std::unique_ptr<HloComputation::Builder> while_rest_builder(
          new HloComputation::Builder("merged_rest_while_loop_" +
                                      std::to_string(while_loop_num)));
      while_loop_num_to_info[while_loop_num] = std::move(WhileLoopInfo(
          while_loop_num, split_size, split_count, split_rest,
          std::move(while_builder), std::move(while_rest_builder)));
      while_loop_num_to_processed_count[while_loop_num] = 0;
      while_loop_num_to_visited_instructions[while_loop_num] = {};
    }

    auto main_split_size = split_count * split_size;
    CHECK(main_split_size + split_rest ==
          lhs->shape().dimensions(split_dim_lhs));

    ss << "<<< "
       << "Splitting dot " << dot->name()
       << " lhs and rhs will be split on contracted dimension with split "
          "size "
       << split_size << " and split rest " << split_rest;

    HloComputation::Builder& body_builder =
        *while_loop_num_to_info[while_loop_num].while_builder;
    for (HloInstruction* leaf : split_leafs_rhs) {
      split_leafs_lhs.push_back(leaf);
    }
    Splitter splitter(
        true, &while_loop_num_to_info[while_loop_num],
        while_loop_num_to_info[while_loop_num].while_loop_parameters,
        while_loop_num_to_info[while_loop_num].while_loop_output_elements,
        body_builder, dot->parent(), absl::MakeSpan(split_leafs_lhs),
        while_loop_num_to_visited_instructions[while_loop_num],
        this->while_loop_num_to_instructions);

    ss << "\n> Split LHS '" << lhs->name() << "'";
    TF_ASSIGN_OR_RETURN(
        HloInstruction * split_lhs,
        splitter.SplitInstruction(lhs, split_dim_lhs, split_size));

    ss << "\n> Split RHS '" << rhs->name() << "'";
    TF_ASSIGN_OR_RETURN(
        HloInstruction * split_rhs,
        splitter.SplitInstruction(rhs, split_dim_rhs, split_size));

    HloInstruction* part = body_builder.AddInstruction(
        dot->CloneWithNewOperands(dot->shape(), {split_lhs, split_rhs}));

    int64_t output_index = splitter.BuildMergedLoopOutput(
        -1, split_size, dot, part, /*combine_with_sum =*/true);
    bool merge_rest = GetDebugOptionsFromFlags().xla_tensor_split_merge_rest();
    if (split_rest == 0 || merge_rest) {
      ss << "\n ----< Exit HandleDot for '" << dot->name() << "' SUCCESS";
      LOG(INFO) << ss.str();
      std::vector<std::tuple<int64_t, Shape>> empty_id_shapes;
      std::vector<HloInstruction*> empty_leafs;
      // * slit_dim is usesless for this case, thus we just send split_dim_lhs
      while_loop_num_to_info[while_loop_num].AddSubOutput(
          SubOutputInfo(output_index, dot, split_dim_lhs, empty_id_shapes,
                        empty_leafs, split_rest, -1, false, false));
      while_loop_num_to_processed_count[while_loop_num] += 1;
      if (CanFinishMergedWhileLoop(while_loop_num)) {
        return FinalizeMergedWhileLoop(while_loop_num);
      }
      return OkStatus();
    } else {
      HloComputation::Builder& rest_builder =
          *while_loop_num_to_info[while_loop_num].while_rest_builder;
      std::vector<HloInstruction*> splitter_parameters;
      std::vector<HloInstruction*> splitter_output_elements;
      Splitter splitter(
          false, &while_loop_num_to_info[while_loop_num],
          while_loop_num_to_info[while_loop_num].while_rest_parameters,
          while_loop_num_to_info[while_loop_num].while_rest_output_elements,
          rest_builder, dot->parent(), absl::MakeSpan(split_leafs_lhs),
          while_loop_num_to_visited_instructions[while_loop_num],
          this->while_loop_num_to_instructions, main_split_size);

      TF_ASSIGN_OR_RETURN(
          HloInstruction * rest_lhs,
          splitter.SplitInstruction(lhs, split_dim_lhs, split_rest));
      TF_ASSIGN_OR_RETURN(
          HloInstruction * rest_rhs,
          splitter.SplitInstruction(rhs, split_dim_rhs, split_rest));

      HloInstruction* rest_part = rest_builder.AddInstruction(
          dot->CloneWithNewOperands(dot->shape(), {rest_lhs, rest_rhs}));
      int64_t rest_output_idx = while_loop_num_to_info[while_loop_num]
                                    .while_rest_output_elements.size();
      while_loop_num_to_info[while_loop_num]
          .while_rest_output_elements.push_back(rest_part);

      ss << "\n ----< Exit HandleDot for '" << dot->name() << "' SUCCESS";
      LOG(INFO) << ss.str();
      std::vector<std::tuple<int64_t, Shape>> empty_id_shapes;
      std::vector<HloInstruction*> empty_leafs;
      // * slit_dim is usesless for this case, thus we just send split_dim_lhs
      while_loop_num_to_info[while_loop_num].AddSubOutput(
          SubOutputInfo(output_index, dot, split_dim_lhs, empty_id_shapes,
                        empty_leafs, split_rest, rest_output_idx, true, false));
      while_loop_num_to_processed_count[while_loop_num] += 1;
      if (CanFinishMergedWhileLoop(while_loop_num)) {
        return FinalizeMergedWhileLoop(while_loop_num);
      }
      return OkStatus();
    }
  } else if ((can_split_lhs && !can_split_rhs) ||
             (!can_split_lhs && can_split_rhs)) {
    //
    // CASE :: one of lhs / rhs is split
    //
    bool split_is_lhs = can_split_lhs;
    HloInstruction* split_inst = split_is_lhs ? lhs : rhs;
    // * must use the recorded split_dim
    // all paths can be used for the while loop, just select the first one
    int64_t split_dim =
        std::get<1>(start_node_to_splittable_paths[node_key][0][1]);
    if (split_dim == -1) {
      // Bail, we can't split this tensor into equally sized parts.
      ss << "\n ----< Impossible Exit HandleDot for '" << dot->name()
         << "' FAILURE";
      LOG(INFO) << ss.str();
      CHECK(false);
    }

    bool combine_parts_with_sum =
        absl::c_linear_search(split_is_lhs ? dnums.lhs_contracting_dimensions()
                                           : dnums.rhs_contracting_dimensions(),
                              split_dim);

    int64_t split_size, split_count, split_rest;
    std::tie(split_size, split_count, split_rest) =
        DetermineSplitSize(split_inst, split_dim);

    HloInstruction* recorded_start =
        std::get<0>(start_node_to_splittable_paths[node_key][0][1]);
    int64_t recorded_size =
        std::get<2>(start_node_to_splittable_paths[node_key][0][1]);

    if (recorded_start->unique_id() != split_inst->unique_id() ||
        recorded_size != split_size) {
      LOG(ERROR) << prefix << "Case 3 Starting_node '" << dot->name()
                 << "' split_inst.name=" << split_inst->name()
                 << " recorded_start.name=" << recorded_start->name()
                 << " split_dim=" << split_dim
                 << " recoded_size=" << recorded_size
                 << " split_size=" << split_size;
      CHECK(false);
    }

    int64_t while_loop_num = start_node_to_while_loop_num[node_key];
    if (!while_loop_num_to_info.contains(
            start_node_to_while_loop_num[node_key])) {
      std::unique_ptr<HloComputation::Builder> while_builder(
          new HloComputation::Builder("merged_while_loop_" +
                                      std::to_string(while_loop_num)));
      std::unique_ptr<HloComputation::Builder> while_rest_builder(
          new HloComputation::Builder("merged_rest_while_loop_" +
                                      std::to_string(while_loop_num)));
      while_loop_num_to_info[while_loop_num] = std::move(WhileLoopInfo(
          while_loop_num, split_size, split_count, split_rest,
          std::move(while_builder), std::move(while_rest_builder)));
      while_loop_num_to_processed_count[while_loop_num] = 0;
      while_loop_num_to_visited_instructions[while_loop_num] = {};
    }

    auto main_split_size = split_count * split_size;
    CHECK(main_split_size + split_rest ==
          split_inst->shape().dimensions(split_dim));

    ss << "\n <<< "
       << "Splitting dot '" << dot->name() << "' "
       << (split_is_lhs ? "lhs" : "rhs") << " will be split on " << split_dim
       << " with split size " << split_size << " and rest size " << split_rest;

    HloComputation::Builder& body_builder =
        *while_loop_num_to_info[while_loop_num].while_builder;
    Splitter splitter(
        true, &while_loop_num_to_info[while_loop_num],
        while_loop_num_to_info[while_loop_num].while_loop_parameters,
        while_loop_num_to_info[while_loop_num].while_loop_output_elements,
        body_builder, dot->parent(),
        absl::MakeSpan(split_is_lhs ? split_leafs_lhs : split_leafs_rhs),
        while_loop_num_to_visited_instructions[while_loop_num],
        this->while_loop_num_to_instructions);

    TF_ASSIGN_OR_RETURN(
        HloInstruction * comp_root,
        splitter.SplitInstruction(split_inst, split_dim, split_size));

    // Add final dot inside of the computation
    HloInstruction* reduce_param;
    int64_t reduce_parameter_idx =
        splitter.AddParameter(split_is_lhs ? rhs : lhs, &reduce_param);

    Shape part_shape = ShapeUtil::MakeShape(dot->shape().element_type(),
                                            dot->shape().dimensions());
    int64_t dot_split_dim = split_dim;  // split dimension after dot occured
    if (split_is_lhs) {
      for (int64_t c_dim : dnums.lhs_contracting_dimensions()) {
        if (c_dim < dot_split_dim) dot_split_dim--;
      }
    } else {
      for (int64_t c_dim : dnums.rhs_contracting_dimensions()) {
        if (c_dim < dot_split_dim) dot_split_dim--;
      }
      dot_split_dim +=
          lhs->shape().rank() - dnums.lhs_contracting_dimensions_size();
    }
    if (!combine_parts_with_sum)
      part_shape.set_dimensions(dot_split_dim, split_size);
    bool merge_rest = GetDebugOptionsFromFlags().xla_tensor_split_merge_rest();
    if (combine_parts_with_sum && (split_rest == 0 || !merge_rest)) {
      Shape sliced_shape =
          ShapeUtil::MakeShape(reduce_param->shape().element_type(),
                               reduce_param->shape().dimensions());
      // FIXME: This assumes dots only contract once (which is currently
      // always true)
      int64_t param_split_dim = split_is_lhs
                                    ? dnums.rhs_contracting_dimensions()[0]
                                    : dnums.lhs_contracting_dimensions()[0];
      sliced_shape.set_dimensions(param_split_dim, split_size);
      std::vector<HloInstruction*> start_indices;
      for (int64_t dim = 0; dim < reduce_param->shape().dimensions_size();
           dim++) {
        if (dim == param_split_dim) {
          start_indices.push_back(splitter.offset());
        } else {
          start_indices.push_back(
              body_builder.AddInstruction(HloInstruction::CreateConstant(
                  LiteralUtil::CreateR0<int32_t>(0))));
        }
      }
      reduce_param =
          body_builder.AddInstruction(HloInstruction::CreateDynamicSlice(
              sliced_shape, reduce_param, absl::MakeSpan(start_indices),
              sliced_shape.dimensions()));
      LOG(INFO) << prefix << "combine_parts_with_sum: dot.name=" << dot->name()
                << " param_split_dim=" << param_split_dim
                << " Create reduce_param_slice: reduce_param_slice.name="
                << reduce_param->name()
                << " slice_shape=" << reduce_param->shape().ToString();
    } else if (combine_parts_with_sum) {
      // need to merge rest
      Shape sliced_shape =
          ShapeUtil::MakeShape(reduce_param->shape().element_type(),
                               reduce_param->shape().dimensions());
      // FIXME: This assumes dots only contract once (which is currently
      // always true)
      int64_t param_split_dim = split_is_lhs
                                    ? dnums.rhs_contracting_dimensions()[0]
                                    : dnums.lhs_contracting_dimensions()[0];
      sliced_shape.set_dimensions(param_split_dim, split_size);
      int64_t split_dim_size =
          reduce_param->shape().dimensions(param_split_dim);
      int64_t tmp_main_split_size =
          int64_t(split_dim_size / split_size) * split_size;

      int64_t padded_split_dim_size = tmp_main_split_size + split_size;
      PaddingConfig padding;
      Shape padded_shape =
          ShapeUtil::MakeShape(reduce_param->shape().element_type(),
                               reduce_param->shape().dimensions());
      padded_shape.set_dimensions(param_split_dim, padded_split_dim_size);
      for (int dim = 0; dim < reduce_param->shape().dimensions_size(); ++dim) {
        PaddingConfig::PaddingConfigDimension* dimension =
            padding.add_dimensions();
        dimension->set_edge_padding_low(0);
        if (dim == param_split_dim) {
          dimension->set_edge_padding_high(padded_split_dim_size -
                                           split_dim_size);
        } else {
          dimension->set_edge_padding_high(0);
        }
        dimension->set_interior_padding(0);
      }
      HloInstruction* zero =
          body_builder.AddInstruction(HloInstruction::CreateConstant(
              LiteralUtil::Zero(padded_shape.element_type())));
      reduce_param = body_builder.AddInstruction(
          HloInstruction::CreatePad(padded_shape, reduce_param, zero, padding));

      HloInstruction* orig_split_dim_size =
          body_builder.AddInstruction(HloInstruction::CreateConstant(
              LiteralUtil::CreateR0<int32_t>(split_dim_size)));

      std::vector<HloInstruction*> start_indices;
      for (int64_t dim = 0; dim < reduce_param->shape().dimensions_size();
           dim++) {
        if (dim == param_split_dim) {
          start_indices.push_back(splitter.offset());
        } else {
          start_indices.push_back(
              body_builder.AddInstruction(HloInstruction::CreateConstant(
                  LiteralUtil::CreateR0<int32_t>(0))));
        }
      }
      reduce_param =
          body_builder.AddInstruction(HloInstruction::CreateDynamicSlice(
              sliced_shape, reduce_param, absl::MakeSpan(start_indices),
              sliced_shape.dimensions()));
      LOG(INFO) << prefix << "combine_parts_with_sum: dot.name=" << dot->name()
                << " param_split_dim=" << param_split_dim
                << " Create reduce_param_slice: reduce_param_slice.name="
                << reduce_param->name()
                << " slice_shape=" << reduce_param->shape().ToString();
      HloInstruction* split_size_const =
          body_builder.AddInstruction(HloInstruction::CreateConstant(
              LiteralUtil::CreateR0<int32_t>(split_size)));
      // offset starts form zero, so
      // dynamic_size=min(orig_split_dim_size-offset_,split_size_const)
      HloInstruction* tmp_split_size =
          body_builder.AddInstruction(HloInstruction::CreateBinary(
              splitter.offset()->shape(), HloOpcode::kSubtract,
              orig_split_dim_size, splitter.offset()));
      HloInstruction* dynamic_size =
          body_builder.AddInstruction(HloInstruction::CreateBinary(
              splitter.offset()->shape(), HloOpcode::kMinimum, tmp_split_size,
              split_size_const));
      reduce_param =
          body_builder.AddInstruction(HloInstruction::CreateSetDimensionSize(
              sliced_shape, reduce_param, dynamic_size, param_split_dim));
    }

    std::vector<HloInstruction*> ops;
    if (split_is_lhs) {
      ops = {comp_root, reduce_param};
    } else {
      ops = {reduce_param, comp_root};
    }
    HloInstruction* part = body_builder.AddInstruction(
        dot->CloneWithNewOperands(part_shape, absl::MakeSpan(ops)));

    int64_t output_index = splitter.BuildMergedLoopOutput(
        dot_split_dim, split_size, dot, part, combine_parts_with_sum);

    if (split_rest == 0 || merge_rest) {
      ss << "\n ----< Exit HandleDot for '" << dot->name() << "' SUCCESS";
      LOG(INFO) << ss.str();
      std::vector<std::tuple<int64_t, Shape>> empty_id_shapes;
      std::vector<HloInstruction*> empty_leafs;
      while_loop_num_to_info[while_loop_num].AddSubOutput(
          SubOutputInfo(output_index, dot, split_dim, empty_id_shapes,
                        empty_leafs, split_rest, -1, false, false));
      while_loop_num_to_processed_count[while_loop_num] += 1;
      if (CanFinishMergedWhileLoop(while_loop_num)) {
        return FinalizeMergedWhileLoop(while_loop_num);
      }
      return OkStatus();

    } else {
      HloComputation::Builder& rest_builder =
          *while_loop_num_to_info[while_loop_num].while_rest_builder;
      std::vector<HloInstruction*> splitter_parameters;
      std::vector<HloInstruction*> splitter_output_elements;
      Splitter splitter(
          false, &while_loop_num_to_info[while_loop_num],
          while_loop_num_to_info[while_loop_num].while_rest_parameters,
          while_loop_num_to_info[while_loop_num].while_rest_output_elements,
          rest_builder, dot->parent(),
          absl::MakeSpan(split_is_lhs ? split_leafs_lhs : split_leafs_rhs),
          while_loop_num_to_visited_instructions[while_loop_num],
          this->while_loop_num_to_instructions, main_split_size);

      TF_ASSIGN_OR_RETURN(
          HloInstruction * comp_root,
          splitter.SplitInstruction(split_inst, split_dim, split_rest));

      // Add final dot inside of the computation
      HloInstruction* reduce_param;
      int64_t reduce_parameter_idx =
          splitter.AddParameter(split_is_lhs ? rhs : lhs, &reduce_param);

      Shape part_shape = ShapeUtil::MakeShape(dot->shape().element_type(),
                                              dot->shape().dimensions());
      int64_t dot_split_dim = split_dim;  // split dimension after dot occured
      if (split_is_lhs) {
        for (int64_t c_dim : dnums.lhs_contracting_dimensions()) {
          if (c_dim < dot_split_dim) dot_split_dim--;
        }
      } else {
        for (int64_t c_dim : dnums.rhs_contracting_dimensions()) {
          if (c_dim < dot_split_dim) dot_split_dim--;
        }
        dot_split_dim +=
            lhs->shape().rank() - dnums.lhs_contracting_dimensions_size();
      }
      if (!combine_parts_with_sum)
        part_shape.set_dimensions(dot_split_dim, split_rest);

      if (combine_parts_with_sum) {
        Shape sliced_shape =
            ShapeUtil::MakeShape(reduce_param->shape().element_type(),
                                 reduce_param->shape().dimensions());
        // FIXME: This assumes dots only contract once (which is currently
        // always true)
        int64_t param_split_dim = split_is_lhs
                                      ? dnums.rhs_contracting_dimensions()[0]
                                      : dnums.lhs_contracting_dimensions()[0];
        sliced_shape.set_dimensions(param_split_dim, split_rest);

        std::vector<HloInstruction*> start_indices;
        for (int64_t dim = 0; dim < reduce_param->shape().dimensions_size();
             dim++) {
          if (dim == param_split_dim) {
            start_indices.push_back(splitter.offset());
          } else {
            start_indices.push_back(
                rest_builder.AddInstruction(HloInstruction::CreateConstant(
                    LiteralUtil::CreateR0<int32_t>(0))));
          }
        }
        reduce_param =
            rest_builder.AddInstruction(HloInstruction::CreateDynamicSlice(
                sliced_shape, reduce_param, absl::MakeSpan(start_indices),
                sliced_shape.dimensions()));
        LOG(INFO) << prefix
                  << "combine_parts_with_sum: dot.name=" << dot->name()
                  << " param_split_dim=" << param_split_dim
                  << " Create rest reduce_param_slice: reduce_param_slice.name="
                  << reduce_param->name()
                  << " slice_shape=" << reduce_param->shape().ToString();
      }

      std::vector<HloInstruction*> ops;
      if (split_is_lhs) {
        ops = {comp_root, reduce_param};
      } else {
        ops = {reduce_param, comp_root};
      }
      HloInstruction* part = rest_builder.AddInstruction(
          dot->CloneWithNewOperands(part_shape, absl::MakeSpan(ops)));
      int64_t rest_output_idx = while_loop_num_to_info[while_loop_num]
                                    .while_rest_output_elements.size();
      while_loop_num_to_info[while_loop_num]
          .while_rest_output_elements.push_back(part);

      ss << "\n ----< Exit HandleDot for '" << dot->name() << "' SUCCESS";
      LOG(INFO) << ss.str();
      std::vector<std::tuple<int64_t, Shape>> empty_id_shapes;
      std::vector<HloInstruction*> empty_leafs;
      while_loop_num_to_info[while_loop_num].AddSubOutput(
          SubOutputInfo(output_index, dot, split_dim, empty_id_shapes,
                        empty_leafs, split_rest, rest_output_idx,
                        combine_parts_with_sum, false, dot_split_dim));
      while_loop_num_to_processed_count[while_loop_num] += 1;
      if (CanFinishMergedWhileLoop(while_loop_num)) {
        return FinalizeMergedWhileLoop(while_loop_num);
      }
      return OkStatus();
    }
  }

  ss << "\n ----< Exit HandleDot for '" << dot->name() << "' with no splitting";
  LOG(INFO) << ss.str();
  return OkStatus();
}
Status TensorSplitterRewriteVisitor::AddReduceToMergedWhileLoop(
    HloInstruction* reduce) {
  std::stringstream ss;
  std::string prefix = "[AddReduceToMergedWhileLoop] ";
  ss << prefix;

  // MatchSupportedReduce enforces that all initializers are
  // scalars, so we only need to split the operands to the
  // reduce itself.
  int64_t op_count = reduce->operand_count() / 2;
  std::vector<HloInstruction*> split_leafs;
  std::vector<int64_t> orig_dims;
  std::vector<int64_t> exclude_dims;
  for (int64_t i = 0; i < op_count; i++) {
    orig_dims.clear();
    for (int64_t j = 0; j < reduce->operand(i)->shape().dimensions_size();
         j++) {
      orig_dims.push_back(j);
    }

    if (!OperandCanBeSplit(reduce->mutable_operand(i), &split_leafs, &orig_dims,
                           &exclude_dims)) {
      ss << "\n<<<Impossible Again, looks like reduce '" << reduce->name()
         << "' cannot be split because of '"
         << reduce->mutable_operand(i)->name() << "'";
      ss << "\n <---- Exit HandleReduce for '" << reduce->name() << "' FAILURE";
      LOG(ERROR) << ss.str();
      CHECK(false);
      return OkStatus();
    }
  }

  if (reduce->shape().IsTuple()) {
    for (int64_t reduce_dim : reduce->dimensions()) {
      exclude_dims.push_back(reduce_dim);
    }
  }
  auto node_key = MakeSplitNodeKey(reduce);
  // * must use the recorded split_dim
  // all paths can be used for the while loop, just select the first one
  int64_t split_dim =
      std::get<1>(start_node_to_splittable_paths[node_key][0][1]);

  bool split_along_reduce_dim =
      absl::c_linear_search(reduce->dimensions(), split_dim);

  int64_t split_size, split_count, split_rest;
  std::tie(split_size, split_count, split_rest) =
      DetermineSplitSize(reduce->mutable_operand(0), split_dim);
  HloInstruction* recorded_start =
      std::get<0>(start_node_to_splittable_paths[node_key][0][1]);
  int64_t recorded_size =
      std::get<2>(start_node_to_splittable_paths[node_key][0][1]);

  if (recorded_start->unique_id() != reduce->mutable_operand(0)->unique_id() ||
      recorded_size != split_size) {
    LOG(ERROR) << prefix << " Starting_node '" << reduce->name()
               << "' start_operand.name=" << reduce->mutable_operand(0)->name()
               << " recorded_start.name=" << recorded_start->name()
               << " split_dim=" << split_dim
               << " recoded_size=" << recorded_size
               << " split_size=" << split_size;
    CHECK(false);
  }

  int64_t while_loop_num = start_node_to_while_loop_num[node_key];
  if (!while_loop_num_to_info.contains(
          start_node_to_while_loop_num[node_key])) {
    std::unique_ptr<HloComputation::Builder> while_builder(
        new HloComputation::Builder("merged_while_loop_" +
                                    std::to_string(while_loop_num)));
    std::unique_ptr<HloComputation::Builder> while_rest_builder(
        new HloComputation::Builder("merged_rest_while_loop_" +
                                    std::to_string(while_loop_num)));
    while_loop_num_to_info[while_loop_num] = std::move(
        WhileLoopInfo(while_loop_num, split_size, split_count, split_rest,
                      std::move(while_builder), std::move(while_rest_builder)));
    while_loop_num_to_processed_count[while_loop_num] = 0;
    while_loop_num_to_visited_instructions[while_loop_num] = {};
  }
  auto main_split_size = split_count * split_size;
  CHECK(main_split_size + split_rest ==
        reduce->mutable_operand(0)->shape().dimensions(split_dim));
  ss << "<<< "
     << "Splitting reduce " << reduce->name()
     << " operands will be split at dimension " << split_dim
     << " with split size " << split_size << " and rest size " << split_rest;

  HloComputation::Builder& body_builder =
      *while_loop_num_to_info[while_loop_num].while_builder;
  Splitter splitter(
      true, &while_loop_num_to_info[while_loop_num],
      while_loop_num_to_info[while_loop_num].while_loop_parameters,
      while_loop_num_to_info[while_loop_num].while_loop_output_elements,
      body_builder, reduce->parent(), absl::MakeSpan(split_leafs),
      while_loop_num_to_visited_instructions[while_loop_num],
      this->while_loop_num_to_instructions);

  std::vector<HloInstruction*> operands;
  for (int64_t i = 0; i < op_count; i++) {
    TF_ASSIGN_OR_RETURN(HloInstruction * split_op,
                        splitter.SplitInstruction(reduce->mutable_operand(i),
                                                  split_dim, split_size));
    operands.push_back(split_op);
  }

  // Add init parameters to computation
  for (int64_t i = 0; i < op_count; i++) {
    HloInstruction* init_op;
    splitter.AddParameter(reduce->mutable_operand(i + op_count), &init_op);
    operands.push_back(init_op);
  }

  // Since initializers are scalars and operands are
  // not, this means the computation already supports
  // broadcasting (i.e. has only pointwise operands with
  // no set shape). We can just copy it directly!

  // TODO: I believe that this is true, but should double
  //       check...

  int64_t reduce_split_dim = split_dim;  // split dim after reduce
  for (int64_t r_dim : reduce->dimensions()) {
    if (r_dim < split_dim) {
      reduce_split_dim--;
    }
  }

  Shape output_part_shape;
  HloInstruction *output_part, *old_output;
  if (reduce->shape().IsTuple()) {
    CHECK(reduce->user_count() == 1);
    old_output = reduce->users()[0];

    Shape new_reduce_shape = ShapeUtil::MakeTupleShape(
        absl::MakeSpan(reduce->shape().tuple_shapes()));
    if (!split_along_reduce_dim) {
      for (int64_t i = 0; i < new_reduce_shape.tuple_shapes_size(); i++) {
        new_reduce_shape.mutable_tuple_shapes(i)->set_dimensions(
            reduce_split_dim, split_size);
      }
    }
    HloInstruction* new_reduce = body_builder.AddInstruction(
        reduce->CloneWithNewOperands(new_reduce_shape, operands));

    output_part_shape = ShapeUtil::MakeShape(old_output->shape().element_type(),
                                             old_output->shape().dimensions());
    if (!split_along_reduce_dim) {
      output_part_shape.set_dimensions(reduce_split_dim, split_size);
    }
    output_part =
        body_builder.AddInstruction(HloInstruction::CreateGetTupleElement(
            output_part_shape, new_reduce, old_output->tuple_index()));
  } else {
    output_part_shape = ShapeUtil::MakeShape(reduce->shape().element_type(),
                                             reduce->shape().dimensions());
    if (!split_along_reduce_dim)
      output_part_shape.set_dimensions(reduce_split_dim, split_size);
    output_part = body_builder.AddInstruction(
        reduce->CloneWithNewOperands(output_part_shape, operands));
    old_output = reduce;
  }

  int64_t output_index = splitter.BuildMergedLoopOutput(
      reduce_split_dim, split_size, old_output, output_part, false,
      split_along_reduce_dim);
  bool merge_rest = GetDebugOptionsFromFlags().xla_tensor_split_merge_rest();
  if (split_rest > 0 && (!merge_rest)) {
    HloComputation::Builder& rest_builder =
        *while_loop_num_to_info[while_loop_num].while_rest_builder;
    Splitter rest_splitter(
        false, &while_loop_num_to_info[while_loop_num],
        while_loop_num_to_info[while_loop_num].while_rest_parameters,
        while_loop_num_to_info[while_loop_num].while_rest_output_elements,
        rest_builder, reduce->parent(), absl::MakeSpan(split_leafs),
        while_loop_num_to_visited_instructions[while_loop_num],
        this->while_loop_num_to_instructions, main_split_size);

    std::vector<HloInstruction*> operands;
    for (int64_t i = 0; i < op_count; i++) {
      TF_ASSIGN_OR_RETURN(
          HloInstruction * split_op,
          rest_splitter.SplitInstruction(reduce->mutable_operand(i), split_dim,
                                         split_rest));
      operands.push_back(split_op);
    }

    // Add init parameters to computation
    for (int64_t i = 0; i < op_count; i++) {
      HloInstruction* init_op;
      rest_splitter.AddParameter(reduce->mutable_operand(i + op_count),
                                 &init_op);
      operands.push_back(init_op);
    }

    int64_t reduce_split_dim = split_dim;
    for (int64_t r_dim : reduce->dimensions()) {
      if (r_dim < split_dim) {
        reduce_split_dim--;
      }
    }

    Shape output_part_shape;
    HloInstruction *output_part, *old_output;
    if (reduce->shape().IsTuple()) {
      CHECK(reduce->user_count() == 1);
      old_output = reduce->users()[0];

      Shape new_reduce_shape = ShapeUtil::MakeTupleShape(
          absl::MakeSpan(reduce->shape().tuple_shapes()));
      if (!split_along_reduce_dim) {
        for (int64_t i = 0; i < new_reduce_shape.tuple_shapes_size(); i++) {
          new_reduce_shape.mutable_tuple_shapes(i)->set_dimensions(
              reduce_split_dim, split_rest);
        }
      }
      HloInstruction* new_reduce = rest_builder.AddInstruction(
          reduce->CloneWithNewOperands(new_reduce_shape, operands));

      output_part_shape = ShapeUtil::MakeShape(
          old_output->shape().element_type(), old_output->shape().dimensions());
      if (!split_along_reduce_dim) {
        output_part_shape.set_dimensions(reduce_split_dim, split_rest);
      }
      output_part =
          rest_builder.AddInstruction(HloInstruction::CreateGetTupleElement(
              output_part_shape, new_reduce, old_output->tuple_index()));
    } else {
      output_part_shape = ShapeUtil::MakeShape(reduce->shape().element_type(),
                                               reduce->shape().dimensions());
      if (!split_along_reduce_dim) {
        output_part_shape.set_dimensions(reduce_split_dim, split_rest);
      }
      output_part = rest_builder.AddInstruction(
          reduce->CloneWithNewOperands(output_part_shape, operands));
      old_output = reduce;
    }

    int64_t rest_output_idx = rest_splitter.BuildRestOutputTuple(
        reduce_split_dim, split_rest, old_output, output_part, false,
        split_along_reduce_dim);
    std::vector<std::tuple<int64_t, Shape>> empty_id_shapes;
    std::vector<HloInstruction*> empty_leafs;

    while_loop_num_to_info[while_loop_num].AddSubOutput(SubOutputInfo(
        output_index, old_output, split_dim, empty_id_shapes, empty_leafs,
        split_rest, rest_output_idx, false, split_along_reduce_dim));
    while_loop_num_to_processed_count[while_loop_num] += 1;
    ss << "\n <---- Exit HandleReduce for '" << old_output->name()
       << "' SUCCESS";
    LOG(INFO) << ss.str();
    if (CanFinishMergedWhileLoop(while_loop_num)) {
      return FinalizeMergedWhileLoop(while_loop_num);
    }
    return OkStatus();
  }
  std::vector<std::tuple<int64_t, Shape>> empty_id_shapes;
  std::vector<HloInstruction*> empty_leafs;
  while_loop_num_to_info[while_loop_num].AddSubOutput(SubOutputInfo(
      output_index, old_output, split_dim, empty_id_shapes, empty_leafs,
      split_rest, -1, false, split_along_reduce_dim));
  while_loop_num_to_processed_count[while_loop_num] += 1;
  ss << "\n <---- Exit HandleReduce for '" << old_output->name() << "' SUCCESS";
  LOG(INFO) << ss.str();
  if (CanFinishMergedWhileLoop(while_loop_num)) {
    return FinalizeMergedWhileLoop(while_loop_num);
  }

  return OkStatus();
}

Status TensorSplitterRewriteVisitor::AddSortToMergedWhileLoop(
    HloInstruction* sort) {
  std::string prefix = "AddSortToMergedWhileLoop";
  HloInstruction* array = sort->mutable_operand(0);
  HloInstruction* indices = sort->mutable_operand(1);

  std::stringstream msg;

  HloInstruction* get_array;
  HloInstruction* get_indices;
  if (sort->shape().IsTuple() && sort->shape().tuple_shapes_size() == 2 &&
      sort->user_count() == 2) {
    get_array = sort->users()[0];
    get_indices = sort->users()[1];
  }

  HloInstruction* array_slice = get_array->users()[0];
  HloInstruction* indices_slice = get_indices->users()[0];
  auto left_is_slice =
      Match(array_slice, m::Slice(m::GetTupleElement(m::Sort())));
  auto right_is_slice =
      Match(indices_slice, m::Slice(m::GetTupleElement(m::Sort())));
  if (!(left_is_slice && right_is_slice)) {
    return OkStatus();
  }

  // Checks that the operation can be split
  auto array_dims_size = array->shape().dimensions().size();
  std::vector<int64_t> original_dims(array_dims_size);
  std::iota(original_dims.begin(), original_dims.end(), 0);

  std::vector<int64_t> exclude_dims;
  std::vector<HloInstruction*> split_leafs;
  std::vector<HloInstruction*> indices_split_leafs;

  bool can_split_array =
      OperandShouldBeSplit(array) &&
      OperandCanBeSplit(array, &split_leafs, &original_dims, &exclude_dims);

  bool can_split_indices = OperandCanBeSplit(indices, &indices_split_leafs,
                                             &original_dims, &exclude_dims);

  auto node_key = MakeSplitNodeKey(sort);
  // * must use the recorded split_dim
  // all paths can be used for the while loop, just select the first one
  int64_t split_dim =
      std::get<1>(start_node_to_splittable_paths[node_key][0][1]);
  if (split_dim == -1 || absl::c_linear_search(exclude_dims, split_dim)) {
    LOG(ERROR) << prefix
               << "Impossible Failed to find best split dimension for '"
               << sort->name() << "'";
    return OkStatus();
  }

  auto last_dim = array_dims_size - 1;
  if (split_dim != last_dim) {
    LOG(ERROR) << prefix << "Impossible Best split dimension for '"
               << sort->name() << "' can only be the last dimension. "
               << "Best found split dimension is " << split_dim;
    return OkStatus();
  }

  int64_t split_size, split_count, split_rest;
  std::tie(split_size, split_count, split_rest) =
      DetermineSplitSize(array, split_dim);

  auto slice_k = array_slice->shape().dimensions(last_dim);
  if (slice_k >= split_size) {
    LOG(ERROR) << prefix << "Impossible Splitting for '" << sort->name()
               << "' will not benefit user as the slicing dimension ("
               << slice_k << ") "
               << "is larger or equal to the split size (" << split_size << ")";
    return OkStatus();
  }

  HloInstruction* recorded_start =
      std::get<0>(start_node_to_splittable_paths[node_key][0][1]);
  int64_t recorded_size =
      std::get<2>(start_node_to_splittable_paths[node_key][0][1]);

  if (recorded_start->unique_id() != array->unique_id() ||
      recorded_size != split_size) {
    LOG(ERROR) << prefix << " Starting_node '" << sort->name()
               << "' start_operand.name=" << array->name()
               << " recorded_start.name=" << recorded_start->name()
               << " split_dim=" << split_dim
               << " recoded_size=" << recorded_size
               << " split_size=" << split_size;
    CHECK(false);
  }

  int64_t while_loop_num = start_node_to_while_loop_num[node_key];
  if (!while_loop_num_to_info.contains(
          start_node_to_while_loop_num[node_key])) {
    std::unique_ptr<HloComputation::Builder> while_builder(
        new HloComputation::Builder("merged_while_loop_" +
                                    std::to_string(while_loop_num)));
    std::unique_ptr<HloComputation::Builder> while_rest_builder(
        new HloComputation::Builder("merged_rest_while_loop_" +
                                    std::to_string(while_loop_num)));
    while_loop_num_to_info[while_loop_num] = std::move(
        WhileLoopInfo(while_loop_num, split_size, split_count, split_rest,
                      std::move(while_builder), std::move(while_rest_builder)));
    while_loop_num_to_processed_count[while_loop_num] = 0;
    while_loop_num_to_visited_instructions[while_loop_num] = {};
  }

  auto main_split_size = split_count * split_size;
  CHECK(main_split_size + split_rest == array->shape().dimensions(split_dim));

  HloComputation* sort_comp = sort->parent();

  // msg << "\n -> Split dim=" << split_dim << ", size=" << split_size
  //     << ", count=" << split_count << ", rest=" << split_rest
  //     << ", slice_k=" << slice_k;

  // Build While body

  absl::c_move(indices_split_leafs, std::back_inserter(split_leafs));

  HloComputation::Builder& body_builder =
      *while_loop_num_to_info[while_loop_num].while_builder;
  Splitter body_splitter(
      true, &while_loop_num_to_info[while_loop_num],
      while_loop_num_to_info[while_loop_num].while_loop_parameters,
      while_loop_num_to_info[while_loop_num].while_loop_output_elements,
      body_builder, sort_comp, absl::MakeSpan(split_leafs),
      while_loop_num_to_visited_instructions[while_loop_num],
      this->while_loop_num_to_instructions);

  TF_ASSIGN_OR_RETURN(
      HloInstruction * body_array,
      body_splitter.SplitInstruction(array, split_dim, split_size));

  TF_ASSIGN_OR_RETURN(
      HloInstruction * body_indices,
      body_splitter.SplitInstruction(indices, split_dim, split_size));

  HloInstruction* body_offset = body_splitter.offset();
  HloInstruction* body_split_size =
      body_builder.AddInstruction(CREATE_CONSTANT_INT32(split_size));

  // Create the body of while loop
  // The body sort operation acts on slices of the original tensor
  Shape body_sort_shape =
      ShapeUtil::MakeTupleShape({body_array->shape(), body_indices->shape()});
  HloInstruction* body_sort = body_builder.AddInstruction(
      sort->CloneWithNewOperands(body_sort_shape, {body_array, body_indices}));
  HloInstruction* body_get_array =
      body_builder.AddInstruction(HloInstruction::CreateGetTupleElement(
          body_sort_shape.tuple_shapes(0), body_sort, 0));
  HloInstruction* body_get_indices =
      body_builder.AddInstruction(HloInstruction::CreateGetTupleElement(
          body_sort_shape.tuple_shapes(1), body_sort, 1));

  HloInstruction* body_array_slice =
      body_builder.AddInstruction(array_slice->CloneWithNewOperands(
          array_slice->shape(), {body_get_array}));
  HloInstruction* body_indices_slice =
      body_builder.AddInstruction(indices_slice->CloneWithNewOperands(
          indices_slice->shape(), {body_get_indices}));

  // Initialize While sort input from outside of the loop
  // and place initial arrays tuple.
  // auto split_dim_size = slice_k * split_count; // TODO

  Shape body_acc_array_shape = array_slice->shape();
  Shape body_acc_indices_shape = indices_slice->shape();
  // body_acc_array_shape.set_dimensions(split_dim, split_dim_size); // TODO
  // body_acc_indices_shape.set_dimensions(split_dim, split_dim_size); // TODO
  Shape body_acc_shape =
      ShapeUtil::MakeTupleShape({body_acc_array_shape, body_acc_indices_shape});

  HloInstruction* body_init_acc_array_scalar =
      sort_comp->AddInstruction(HloInstruction::CreateConstant(
          LiteralUtil::Zero(body_acc_shape.tuple_shapes(0).element_type())));
  HloInstruction* body_init_acc_indices_scalar =
      sort_comp->AddInstruction(HloInstruction::CreateConstant(
          LiteralUtil::Zero(body_acc_shape.tuple_shapes(1).element_type())));

  HloInstruction* body_init_acc_array =
      sort_comp->AddInstruction(HloInstruction::CreateBroadcast(
          body_acc_shape.tuple_shapes(0), body_init_acc_array_scalar, {}));
  HloInstruction* body_init_acc_indices =
      sort_comp->AddInstruction(HloInstruction::CreateBroadcast(
          body_acc_shape.tuple_shapes(1), body_init_acc_indices_scalar, {}));

  HloInstruction* body_init_acc =
      sort_comp->AddInstruction(HloInstruction::CreateTuple(
          {body_init_acc_array, body_init_acc_indices}));

  HloInstruction* body_get_acc;
  auto body_acc_parameter_id =
      body_splitter.AddParameter(body_init_acc, &body_get_acc);

  // While operation updates
  // Update While accumulator for sort operation.

  // Continue building body. Sorting cached and new arrays.
  //

  HloInstruction* body_cond_params =
      body_builder.AddInstruction(HloInstruction::CreateTuple(
          {body_array_slice, body_indices_slice, body_get_acc}));

  const Shape& body_cond_params_shape = body_cond_params->shape();

  HloComputation::Builder false_builder("splitter_body_false_cond");
  {
    auto false_param =
        false_builder.AddInstruction(HloInstruction::CreateParameter(
            0, body_cond_params_shape, "body_sort_result"));

    auto false_array_slice_new =
        false_builder.AddInstruction(HloInstruction::CreateGetTupleElement(
            body_cond_params_shape.tuple_shapes(0), false_param, 0));
    auto false_indices_slice_new =
        false_builder.AddInstruction(HloInstruction::CreateGetTupleElement(
            body_cond_params_shape.tuple_shapes(1), false_param, 1));

    const Shape& false_acc_shape = body_cond_params_shape.tuple_shapes(2);
    auto false_acc = false_builder.AddInstruction(
        HloInstruction::CreateGetTupleElement(false_acc_shape, false_param, 2));

    auto false_array_acc =
        false_builder.AddInstruction(HloInstruction::CreateGetTupleElement(
            false_acc_shape.tuple_shapes(0), false_acc, 0));
    auto false_indices_acc =
        false_builder.AddInstruction(HloInstruction::CreateGetTupleElement(
            false_acc_shape.tuple_shapes(1), false_acc, 1));

    Shape false_array_shape = body_acc_array_shape;
    Shape false_indices_shape = body_acc_indices_shape;
    false_array_shape.set_dimensions(
        split_dim, 2 * body_acc_array_shape.dimensions(split_dim));
    false_indices_shape.set_dimensions(
        split_dim, 2 * body_acc_indices_shape.dimensions(split_dim));
    Shape false_sort_shape =
        ShapeUtil::MakeTupleShape({false_array_shape, false_indices_shape});

    auto false_array =
        false_builder.AddInstruction(HloInstruction::CreateConcatenate(
            false_sort_shape.tuple_shapes(0),
            {false_array_slice_new, false_array_acc}, split_dim));
    auto false_indices =
        false_builder.AddInstruction(HloInstruction::CreateConcatenate(
            false_sort_shape.tuple_shapes(1),
            {false_indices_slice_new, false_indices_acc}, split_dim));

    auto false_sort = false_builder.AddInstruction(sort->CloneWithNewOperands(
        false_sort_shape, {false_array, false_indices}));

    auto false_array_get =
        false_builder.AddInstruction(HloInstruction::CreateGetTupleElement(
            false_sort_shape.tuple_shapes(0), false_sort, 0));
    auto false_indices_get =
        false_builder.AddInstruction(HloInstruction::CreateGetTupleElement(
            false_sort_shape.tuple_shapes(1), false_sort, 1));

    auto false_array_slice =
        false_builder.AddInstruction(array_slice->CloneWithNewOperands(
            array_slice->shape(), {false_array_get}));
    auto false_indices_slice =
        false_builder.AddInstruction(indices_slice->CloneWithNewOperands(
            indices_slice->shape(), {false_indices_get}));

    false_builder.AddInstruction(
        HloInstruction::CreateTuple({false_array_slice, false_indices_slice}));
  }
  HloComputation* false_comp =
      parent_module->AddEmbeddedComputation(false_builder.Build());

  HloComputation::Builder true_builder("splitter_body_true_cond");
  {
    auto true_param =
        true_builder.AddInstruction(HloInstruction::CreateParameter(
            0, body_cond_params_shape, "body_sort_result"));

    auto true_array =
        true_builder.AddInstruction(HloInstruction::CreateGetTupleElement(
            body_cond_params_shape.tuple_shapes(0), true_param, 0));
    auto true_indices =
        true_builder.AddInstruction(HloInstruction::CreateGetTupleElement(
            body_cond_params_shape.tuple_shapes(1), true_param, 1));

    true_builder.AddInstruction(
        HloInstruction::CreateTuple({true_array, true_indices}));
  }
  HloComputation* true_comp =
      parent_module->AddEmbeddedComputation(true_builder.Build());

  HloInstruction* body_init_offset =
      body_builder.AddInstruction(CREATE_CONSTANT_INT32(0));
  HloInstruction* body_cond_pred =
      body_builder.AddInstruction(HloInstruction::CreateCompare(
          ShapeUtil::MakeShape(PRED, {}), body_offset, body_init_offset,
          ComparisonDirection::kEq));

  HloInstruction* body_update_acc =
      body_builder.AddInstruction(HloInstruction::CreateConditional(
          body_acc_shape, body_cond_pred, body_cond_params, true_comp,
          body_cond_params, false_comp));

  std::vector<HloInstruction*>& while_parameter_updates =
      body_splitter.while_loop_output_elements;
  // Update split slice iteration
  int64_t output_index_offset = 0;
  if (body_splitter.param_start_index == 0) {
    HloInstruction* body_update_offset = body_builder.AddInstruction(
        HloInstruction::CreateBinary(body_offset->shape(), HloOpcode::kAdd,
                                     body_offset, body_split_size));
    while_parameter_updates.push_back(body_update_offset);
    output_index_offset = 1;
  } else {
    output_index_offset = 0;
  }

  // Collect all updates to the input parameters into a tuple.
  HloInstruction* while_parameters = body_splitter.tuple_parameters();
  const Shape& while_parameters_shape = while_parameters->shape();

  for (auto id = body_splitter.param_start_index + output_index_offset;
       id < while_parameters_shape.tuple_shapes_size(); id++) {
    if (id == body_acc_parameter_id) {
      while_parameter_updates.push_back(body_update_acc);
    } else {
      HloInstruction* body_get_input =
          body_builder.AddInstruction(HloInstruction::CreateGetTupleElement(
              while_parameters_shape.tuple_shapes(id), while_parameters, id));
      while_parameter_updates.push_back(body_get_input);
    }
  }

  std::vector<std::tuple<int64_t, Shape>> ids_and_shapes = {
      std::make_tuple(body_acc_parameter_id, body_acc_shape)};

  if (split_rest == 0) {
    std::vector<HloInstruction*> empty_leafs;
    while_loop_num_to_info[while_loop_num].AddSubOutput(SubOutputInfo(
        while_parameter_updates.size() - 1, sort, split_dim, ids_and_shapes,
        empty_leafs, split_rest, -1, false, false));
    while_loop_num_to_processed_count[while_loop_num] += 1;
    LOG(INFO) << "\n <---- Exit HandleSort for '" << sort->name()
              << "' SUCCESS";
    if (CanFinishMergedWhileLoop(while_loop_num)) {
      return FinalizeMergedWhileLoop(while_loop_num);
    }

    return OkStatus();

  } else {
    // for sort, we can only build rest after the creation of while-loop
    while_loop_num_to_info[while_loop_num].AddSubOutput(
        SubOutputInfo(while_parameter_updates.size() - 1, sort, split_dim,
                      ids_and_shapes, split_leafs, split_rest, -1, false));
    while_loop_num_to_processed_count[while_loop_num] += 1;
    LOG(INFO) << "\n <---- Exit HandleSort for '" << sort->name()
              << "' SUCCESS";
    if (CanFinishMergedWhileLoop(while_loop_num)) {
      return FinalizeMergedWhileLoop(while_loop_num);
    }
    return OkStatus();
  }

  changed_ = true;
  return OkStatus();
}

Status TensorSplitterRewriteVisitor::HandleDot(HloInstruction* dot) {
  HloInstruction *lhs, *rhs;
  CHECK(Match(dot, m::Dot(m::Op(&lhs), m::Op(&rhs))));
  auto& dnums = dot->dot_dimension_numbers();

  std::stringstream ss;

  LOG(INFO) << "\n ---------------------------"
            << "\n ----> Enter HandleDot for '" << dot->name() << "'";

  if (OperandShouldBeSplit(dot)) {
    ss << "\n ----< Exit HandleDot for '" << dot->name() << "' FAILURE";
    LOG(INFO) << ss.str();
    return OkStatus();
  }
  LOG(INFO) << "\n ----< [TensorSplitterRewriteVisitor]  HandleDot for '"
            << dot->name() << " dot.shape=" << dot->shape().ToString()
            << "' dot.byte_size=" << ShapeUtil::ByteSizeOfElements(dot->shape())
            << " max_size_threshold= " << max_size_threshold;

  auto path_key = MakeSplitNodeKey(dot);
  if (!start_node_to_splittable_paths.contains(path_key)) {
    // this is not a splittable dot
    ss << "\n ----< Exit HandleDot for '" << dot->name() << "' FAILURE";
    LOG(INFO) << ss.str();
    return OkStatus();
  }

  if (while_loop_num_to_start_node[start_node_to_while_loop_num[path_key]]
          .size() == 1) {
    if (while_loop_num_to_start_node[start_node_to_while_loop_num[path_key]]
            .back() != path_key) {
      LOG(ERROR) << "starting_node '" << dot->name()
                 << "' path_key=" << path_key << " doesn't while_loop_num="
                 << start_node_to_while_loop_num[path_key] << "'s path_key="
                 << while_loop_num_to_start_node
                        [start_node_to_while_loop_num[path_key]]
                            .back();
      CHECK(false);
    }
    return AddDotToMergedWhileLoop(dot, lhs, rhs);
  } else {
    return AddDotToMergedWhileLoop(dot, lhs, rhs);
  }
}

Status TensorSplitterRewriteVisitor::HandleReduce(HloInstruction* reduce) {
  if (!MatchSupportedReduce(reduce)) {
    return OkStatus();
  }

  std::stringstream ss;

  LOG(INFO) << "\n =============================="
            << "\n ----> Enter HandleReduce for '" << reduce->name() << "'";

  // MatchSupportedReduce enforces that all inputs are of the
  // same shape, and that there is at least one operand!
  if (!OperandShouldBeSplit(reduce->mutable_operand(0))) {
    ss << "\n<<< Reduce '" << reduce->name()
       << "' cannot be split. Something is not splittable on the way up.";
    ss << "\n <---- Exit HandleReduce for '" << reduce->name() << "' FAILURE";
    LOG(INFO) << ss.str();
    return OkStatus();
  }

  // TODO: This is a hack, I need to more seriously rethink the
  //       two pass system, to mark elements in a first pass and combine
  //       sections properly ...
  if (OperandShouldBeSplit(reduce)) {
    ss << "\n<<< Looks like reduce '" << reduce->name()
       << "' cannot be split after all";
    ss << "\n <---- Exit HandleReduce for '" << reduce->name() << "' FAILURE";
    LOG(INFO) << ss.str();
    return OkStatus();
  }

  // If this is a multi-argument reduce, check if only one
  // result is used.
  if (reduce->shape().IsTuple() && reduce->user_count() > 1) {
    ss << "\n<<< Nah, looks like reduce '" << reduce->name()
       << "' cannot be split after all";
    ss << "\n <---- Exit HandleReduce for '" << reduce->name() << "' FAILURE";
    LOG(INFO) << ss.str();
    return OkStatus();
  }

  auto path_key = MakeSplitNodeKey(reduce);
  if (!start_node_to_splittable_paths.contains(path_key)) {
    // this is not a splittable reduce
    ss << "\n<<< Not recorded reduce '" << reduce->name();
    ss << "\n <---- Exit HandleReduce for '" << reduce->name() << "' FAILURE";
    LOG(INFO) << ss.str();
    return OkStatus();
  }

  // need to split
  if (while_loop_num_to_start_node[start_node_to_while_loop_num[path_key]]
          .size() == 1) {
    if (while_loop_num_to_start_node[start_node_to_while_loop_num[path_key]]
            .back() != path_key) {
      LOG(ERROR) << "starting_node '" << reduce->name()
                 << "' path_key=" << path_key << " doesn't while_loop_num="
                 << start_node_to_while_loop_num[path_key] << "'s path_key="
                 << while_loop_num_to_start_node
                        [start_node_to_while_loop_num[path_key]]
                            .back();
      CHECK(false);
    }
    return AddReduceToMergedWhileLoop(reduce);

  } else {
    return AddReduceToMergedWhileLoop(reduce);
  }
}

Status TensorSplitterRewriteVisitor::HandleSort(HloInstruction* sort) {
  CHECK(Match(sort, m::Sort()));
  std::stringstream msg;
  auto path_key = MakeSplitNodeKey(sort);
  if (!start_node_to_splittable_paths.contains(path_key)) {
    // this is not a splittable sort
    msg << "\n ----< Exit HandleSort for '" << sort->name() << "' FAILURE";
    LOG(INFO) << msg.str();
    return OkStatus();
  }

  // this must be a sort which needs to be split
  if (while_loop_num_to_start_node[start_node_to_while_loop_num[path_key]]
          .size() == 1) {
    if (while_loop_num_to_start_node[start_node_to_while_loop_num[path_key]]
            .back() != path_key) {
      LOG(ERROR) << "starting_node '" << sort->name()
                 << "' path_key=" << path_key << " doesn't while_loop_num="
                 << start_node_to_while_loop_num[path_key] << "'s path_key="
                 << while_loop_num_to_start_node
                        [start_node_to_while_loop_num[path_key]]
                            .back();
      CHECK(false);
    }
    return AddSortToMergedWhileLoop(sort);
  } else {
    return AddSortToMergedWhileLoop(sort);
  }
}

bool TensorSplitter::endsWith(const std::string& str, std::string pattern) {
  if (pattern.size() > str.size()) return false;
  for (int i = 1; i <= pattern.size(); i++) {
    if (pattern[pattern.size() - i] != str[str.size() - i]) return false;
  }
  return true;
}

std::tuple<int64_t, int64_t> TensorSplitter::SplitSettings() {
  auto tensor_size_threshold =
      GetDebugOptionsFromFlags().xla_tensor_size_threshold();
  auto tensor_split_size = GetDebugOptionsFromFlags().xla_tensor_split_size();
  LOG(INFO) << "[TensorSplitter::SplitSettings] tensor_size_threshold="
            << tensor_size_threshold.c_str()
            << " tensor_split_size=" << tensor_split_size;
  auto size_threshold = TensorBytes(tensor_size_threshold);
  auto split_size = TensorBytes(tensor_split_size);
  if (split_size == 0) {
    split_size = size_threshold;
  }
  return std::make_tuple(size_threshold, split_size);
}

int64_t TensorSplitter::TensorBytes(const std::string& option) {
  int64_t raw = (int64_t)atoi(option.c_str());
  LOG(INFO) << "[TensorSplitter::TensorBytes] option= '" << option.c_str()
            << "'";
  if (raw < 0) {
    LOG(INFO) << "[TensorSplitter::TensorBytes] raw=" << raw
              << " return default_value=" << 134217728;
    return 134217728;  // 1 GiB
  }

  if (endsWith(option, "GB") || endsWith(option, "gb"))
    return raw * 1000000000;  // 1e9
  else if (endsWith(option, "GiB"))
    return raw * 134217728;
  else if (endsWith(option, "MB") || endsWith(option, "mb"))
    return raw * 1000000;  // 1e6
  else if (endsWith(option, "MiB"))
    return raw * 1048576;
  else if (endsWith(option, "kB") || endsWith(option, "kb"))
    return raw * 1000;
  else if (endsWith(option, "kiB"))
    return raw * 1024;
  else
    return raw;  // interpret as bytes
}

StatusOr<bool> TensorSplitter::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  int64_t size_threshold;
  int64_t split_size;
  std::tie(size_threshold, split_size) = SplitSettings();
  SplittablePathRecorder recorder(size_threshold, split_size, module);
  recorder.RunOnModule(module, execution_threads);
  recorder.AllocateWhileLoops();
  TensorSplitterRewriteVisitor rewriter(
      size_threshold, split_size, module,
      recorder.GetStartNodeToSplittablePaths(),
      recorder.GetStartNodeToWhileLoopNum(),
      recorder.GetWhileLoopNumToStartNode(),
      recorder.GetWhileLoopNumToInstructions());
  LOG(INFO) << "[TensorSplitter::Run] Running tensor splitter for '"
            << module->name() << "'";
  LOG(INFO) << "[TensorSplitter::Run] split_size_threshold=" << size_threshold
            << " target_split_size=" << split_size;
  return rewriter.RunOnModule(module, execution_threads);
}
}  // namespace xla
