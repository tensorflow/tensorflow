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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_IR_EMITTER_UNNESTED_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_IR_EMITTER_UNNESTED_H_

#include <array>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_computation.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emitter.h"
#include "tensorflow/compiler/xla/service/gpu/kernel_mapping_scheme.h"
#include "tensorflow/compiler/xla/service/gpu/nccl_all_reduce_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/reusable_kernel_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/thunk.h"
#include "tensorflow/compiler/xla/service/llvm_ir/ir_array.h"
#include "tensorflow/compiler/xla/service/llvm_ir/llvm_util.h"
#include "tensorflow/tsl/platform/statusor.h"
#include "tensorflow/tsl/protobuf/autotuning.pb.h"

namespace xla {
namespace gpu {

struct BufferSlice {
  // The root buffer to look at.
  BufferAllocation::Slice buffer_slice;

  // The global constant name of the buffer, if it's a constant.
  std::string constant_name;

  // The buffer is modified by the kernel.
  bool written = false;

  Shape shape;
};

// Emits LLVM IR for an "unnested computation".
//
// An unnested computation is an HloComputation which you run by executing one
// or more kernels for each HloInstruction it contains.  Examples of unnested
// computations:
//
//  - An HloModule's root computation,
//  - The body of an HLO while loop,
//  - The true/false computation of an HLO conditional.
//
// Note the opportunity for confusion -- the while loop's computation is nested
// within the root computation, but it's emitted using IrEmitterUnnested!  Don't
// think about it too hard.
//
// Examples of things that are not unnested computations:
//
//  - The body of a fusion node.  IrEmitterUnnested emits the relevant code
//    within a kernel function using FusedIrEmitter.  (FusedIrEmitter is not
//    really an IrEmitter, but is more an "IR generator generator".)
//
class IrEmitterUnnested : public IrEmitter {
 public:
  // Contains threading information. Note that for performance we might apply
  // thread id "scaling" where the physical thread id (to achieve good SM
  // occupancy) will differ from logical thread id. This struct contains
  // logical thread ids, along with meta-information about the scaling applied.
  struct ThreadIdInfo {
    ThreadIdInfo(llvm::Value* thread_id, llvm::Value* thread_id_x,
                 llvm::Value* thread_id_y, llvm::Value* lane_id,
                 llvm::Value* block_id, llvm::Value* scaling)
        : thread_id(thread_id),
          thread_id_x(thread_id_x),
          thread_id_y(thread_id_y),
          lane_id(lane_id),
          block_id(block_id),
          scaling(scaling) {}

    llvm::Value* thread_id;

    // X-coordinate calculated from thread id: `thread_id % num_threads_x`
    llvm::Value* thread_id_x;

    // Y-coordinate calculated from thread id: `thread_id / num_threads_x`
    llvm::Value* thread_id_y;

    // Lane id: `thread_id % WarpSize`
    llvm::Value* lane_id;

    // Block id.
    llvm::Value* block_id;

    // Emits GEP into a shared memory, taking virtual thread scaling into
    // account. Automatically inserts the first zero required by LLVM GEP.
    // Defined on ThreadIdInfo to keep `scaling` private.
    //
    // Same semantics as CreateInBoundsGEP.
    llvm::Value* GEPIntoSharedMemory(
        llvm::IRBuilder<>* b, llvm::GlobalVariable* shared,
        absl::Span<llvm::Value* const> idx_major_to_minor,
        const llvm::Twine& name = "") const;

    // Calculuate the pointee type of the llvm::Value returned by
    // GEPIntoSharedMemory
    llvm::Type* GEPIntoSharedMemoryType(
        llvm::GlobalVariable* shared,
        absl::Span<llvm::Value* const> idx_major_to_minor) const;

   private:
    llvm::Value* scaling;
  };

  absl::string_view platform_name() const {
    return ir_emitter_context_->platform_name();
  }

  using ValueVector3 = std::array<llvm::Value*, 3>;
  using ValueVector2 = std::array<llvm::Value*, 2>;

  // A function object to generate code to process one element in a tile.
  //
  // index: the index for the first output element of the current thread.
  // y_loc: The y coordinate within a tile.
  // x_loc: The x coordinate within a tile.
  using EmitElementFunction = std::function<void(
      const ThreadIdInfo& thread_id_info, const llvm_ir::IrArray::Index& index,
      llvm::Value* y_loc, llvm::Value* x_loc)>;

  using ConstantGenerator = std::function<llvm::Value*(int64_t)>;

  // A function to generate the code to emit the entire tile.
  //
  // index: Absolute coordinate of the start of the tile in input.
  // tile_dimensions: Size of the tile
  using TileElementGenerator = std::function<void(
      const ThreadIdInfo& thread_id_info, const llvm_ir::IrArray::Index& index,
      ValueVector2 tile_dimensions)>;

  // Fusion root -> array of indexes, one per reduction output.
  using ReductionOutputMap =
      ConstHloInstructionMap<absl::Span<llvm_ir::IrArray const>>;

  using ExtraOutputGensMap = ConstHloInstructionMap<llvm_ir::ElementGenerator>;

  IrEmitterUnnested(const IrEmitterUnnested&) = delete;
  IrEmitterUnnested& operator=(const IrEmitterUnnested&) = delete;

  static StatusOr<std::unique_ptr<IrEmitterUnnested>> Create(
      const HloModuleConfig& hlo_module_config,
      IrEmitterContext* ir_emitter_context);

  // Transfers the ownship of thunk_sequence_ out.
  std::unique_ptr<ThunkSequence> ConsumeThunkSequence() {
    return std::make_unique<ThunkSequence>(std::move(thunk_sequence_));
  }

  // Emits code for the given LMHLO region.
  //
  // Also populates related information to 'ir_emitter_context_' for
  // large-constant initializations. Large constants don't get initializers in
  // the generated code and so must be initialized by XLA. The value of these
  // constants will be stored in 'content'. Constants with initializers in the
  // generated code will have empty 'content'.
  Status EmitLmhloRegion(mlir::Region* region);

  static void GetDependentDialects(mlir::DialectRegistry& registry);

 private:
  IrEmitterUnnested(const HloModuleConfig& hlo_module_config,
                    IrEmitterContext* ir_emitter_context);

  Status EmitUnreachable(mlir::Operation* op, std::string error_message);

  // IrEmitterUnnested handles the following instructions differently from
  // IrEmitter. It also mixes in some special handling for custom kernels
  // via the ThunkEmitter.
  Status EmitConstant(mlir::Operation* op);

  Status EmitCopy(mlir::Operation* op);

  Status EmitConditional(mlir::Operation* op);
  Status EmitConvolutionThunk(mlir::Operation* op);
  Status EmitGemmThunk(mlir::Operation* op);
#if GOOGLE_CUDA
  Status EmitCublasLtMatmulThunk(mlir::Operation* op);
  Status EmitCublasLtMatmulThunkF8(mlir::Operation* op);
  Status EmitConvolutionReorderThunk(mlir::Operation* op);
  Status EmitTritonFusion(mlir::Operation* op,
                          tensorflow::AutotuneResult::TritonGemmKey& config);
#endif  // GOOGLE_CUDA
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  Status EmitCholeskyThunk(mlir::Operation* op);
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  Status EmitCustomCallThunk(mlir::Operation* op);
  Status EmitFftThunk(mlir::Operation* op);
  Status EmitFusion(mlir::Operation* op);
  Status EmitLaunchFunc(mlir::Operation* op);
  Status EmitLoopFusion(mlir::Operation* op);
  Status EmitReduce(mlir::Operation* op);
  Status EmitSelectAndScatter(mlir::Operation* op);
  Status EmitWhile(mlir::Operation* op);
  Status EmitInfeed(mlir::Operation* op);
  Status EmitOutfeed(mlir::Operation* op);
  Status EmitRngGetAndUpdateState(mlir::Operation* op);
  Status EmitScatter(mlir::Operation* op);
  Status EmitSort(mlir::Operation* op);
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  Status EmitTriangularSolveCustomCall(mlir::Operation* op);
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

  template <typename NcclThunkType, typename OpT>
  Status EmitNcclThunk(mlir::Operation* op);
  template <typename NcclThunkType, typename OpT>
  Status EmitNcclAsyncDone(mlir::Operation* op);

  template <typename ThunkType, typename OpT>
  Status EmitReplicaOrPartitionId(mlir::Operation* op);

  template <typename NcclThunkType, typename OpT>
  Status EmitCollectivePermute(mlir::Operation* op);

  Status EmitOp(mlir::Operation* op);

  static Thunk::ThunkInfo GetThunkInfo(mlir::Operation* op);

  Status EmitTargetElementLoop(
      const HloInstruction& hlo,
      const llvm_ir::ElementGenerator& body_emitter) override;

  // Add a owning Thunk object to the thunk sequence.
  void AddThunkToThunkSequence(std::unique_ptr<Thunk> thunk) {
    thunk_sequence_.emplace_back(std::move(thunk));
  }

  // Load data from potentially unaligned address. If address is offset by
  // `alignment_bytes`, data is read in the unit of `alignment_bytes` to avoid
  // memory read misalignment in CUDA; otherwise, the entire data are loaded
  // from the given memory address.
  //
  //   address: the memory address to load data from.
  //   data_type: the type of data to load.
  //   alignment_bytes: the number of bytes required to align. The number of
  //     bytes of the data_type must be divisible by alignment_bytes.
  llvm::Value* CreateLoad(llvm::Value* address, llvm::Type* data_type,
                          int alignment_bytes);

  // Store data at a potentially unaligned address. If the address is offset by
  // `alignment_bytes`, data is stored in the unit of `alignment_bytes` to avoid
  // memory write misalignment in CUDA; otherwise, the entire data is stored at
  // the given memory address.
  //
  //   data: the data to be stored.
  //   address: the memory address to store data.
  //   alignment_bytes: the number of bytes required to align. The number of
  //     bytes of the data_type must be divisible by alignment_bytes.
  void CreateStore(llvm::Value* data, llvm::Value* address,
                   int alignment_bytes);

  // Input = {static array, dynamic_dim0, dynamic_dim1}
  // Output = {dynamic array(with dynamic dimension meta data at the end)}
  // For a tensor with static dimension [2][<=5] and dynamic dimension [2][3]
  // (`_` stands for padding)
  // Input = {{1,2,3,_,_,4,5,6_,_}, 2, 3}
  // Output = {{1,2,3,4,5,6,_,_,_,_,2,3}}

  // pseudo code for padToStatic on a 2d array
  //   ```
  // void padToStatic(int** input, int** output, int threads_per_block,
  //                  int meta_data_offset, int max_num_element,
  //                  int static_dim0_size, int static_dim1_size) {
  //   int* source_array = input[0];
  //   int* dest_array = output[0];

  //   // extract the dynamic dimension from the source array's metadata
  //   int* dyn_dim0_size = source_array + meta_data_offset;
  //   int* dyn_dim1_size = source_array + meta_data_offset + sizeof(int);

  //   // only one thread need to store the dynamic index
  //   int thread_id = GetThreadId();
  //   int block_id = GetBlockId();
  //   if (thread_id == 0 && block_id == 0) {
  //     *output[1] = *dyn_dim0_size;
  //     *output[2] = *dyn_dim1_size;
  //   }

  //   int dyn_element_total = 1;
  //   dyn_element_total *= *dyn_dim0_size;
  //   dyn_element_total *= *dyn_dim1_size;
  //   linear_index = block_id * threads_per_block + thread_id;
  //   if (linear_index < max_num_element) {
  //     Index static_index =
  //         delinerized(linerized_index, static_dim0_size, static_dim1_size);
  //     if (linerized_index < dyn_element_total) {
  //       Index dyn_index =
  //           delinerized(linerized_index, *dyn_dim0_size, *dyn_dim1_size);
  //       dest_array[dyn_index.dim0][dyn_index.dim1] =
  //           source_array[static_index.dim0][static_index.dim1];
  //     }
  //   }
  //   return;
  // }
  //   ```
  Status EmitPadToStatic(mlir::Operation* op);

  // Input = {dynamic array(with dynamic dimension meta data at the end)}
  // Output = {static array, dynamic_dim0, dynamic_dim1}
  // For a tensor with static dimension [2][<=5] and dynamic dimension [2][3]
  // (`_` stands for padding)
  // Input = {{1,2,3,4,5,6,_,_,_,_,2,3}}
  // Output = {{1,2,3,_,_,4,5,6_,_}, 2, 3}

  // pseudo code for sliceToDynamic on a 2d array
  //   ```
  // void sliceToDynamic(int** input, int** output, int threads_per_block,
  //                  int meta_data_offset, int max_num_element,
  //                  int static_dim0_size, int static_dim1_size) {
  //   int* source_array = input[0];
  //   int* dest_array = output[0];

  //   // calculate the location where metadata needs to be inserted
  //   int* dyn_dim0_size = dest_array + meta_data_offset;
  //   int* dyn_dim1_size = dest_array + meta_data_offset + sizeof(int);

  //   // only one thread need to store the dynamic index
  //   int thread_id = GetThreadId();
  //   int block_id = GetBlockId();
  //   if (thread_id == 0 && block_id == 0) {
  //     *dyn_dim0_size = *output[1];
  //     *dyn_dim1_size = *output[2];
  //   }

  //   int dyn_element_total = 1;
  //   dyn_element_total *= *dyn_dim0_size;
  //   dyn_element_total *= *dyn_dim1_size;
  //   linear_index = block_id * threads_per_block + thread_id;
  //   if (linear_index < max_num_element) {
  //     Index static_index =
  //         delinerized(linerized_index, static_dim0_size, static_dim1_size);
  //     if (linerized_index < dyn_element_total) {
  //       Index dyn_index =
  //           delinerized(linerized_index, *dyn_dim0_size, *dyn_dim1_size);
  //       dest_array[static_index.dim0][static_index.dim1] =
  //           source_array[dyn_index.dim0][dyn_index.dim1];
  //     }
  //   }
  //   return;
  // }
  //   ```
  Status EmitSliceToDynamic(mlir::Operation* op);

  StatusOr<BufferAllocation::Slice> GetAllocationSlice(
      mlir::Value v, std::string* constant_name = nullptr);

  int64_t ByteSizeOf(const Shape& shape) const {
    return llvm_ir::ByteSizeOf(
        shape, ir_emitter_context_->llvm_module()->getDataLayout());
  }

  // Builds the prototype of the IR kernel for `inst` and adds it to the module.
  // This kernel takes as arguments pointers to the given buffer allocations.
  llvm::Function* BuildKernelPrototype(
      absl::string_view name, absl::Span<const BufferAllocation* const> args);

  // An argument descriptor for "reusable kernels".
  struct ReusableKernelArgument {
    mlir::Value value;
    Shape shape;
    BufferAllocation::Slice slice;
    bool aliased = true;
    int64_t alignment = 1;
    bool written = true;
    // Holds the index of the first argument which has the same slice as this,
    // if this is not the first such argument.
    std::optional<int> first_with_same_slice;
  };

  // The return type of BuildReusableKernelPrototype.
  struct KernelAndIrArrays {
    llvm::Function* kernel = nullptr;
    std::vector<llvm_ir::IrArray> ir_arrays;
  };

  KernelAndIrArrays BuildReusableKernelPrototype(
      absl::string_view suggested_name,
      absl::Span<const ReusableKernelArgument> arguments,
      const LaunchDimensions& launch_dimensions);

  // Helper for writing extra outputs from inside a reduce kernel.
  Status EmitExtraOutputsForReduce(const Shape& reduction_operand_shape,
                                   const ReductionOutputMap& result_ir_arrays,
                                   const llvm_ir::IrArray::Index& index,
                                   const ReductionCodegenInfo& reduction_info,
                                   const ExtraOutputGensMap& extra_output_gens);

  // Generates code for reduction to contiguous dimensions.
  //
  // Row reduction uses the following algorithm described in CUDA-like
  // pseudocode:
  //
  // ```
  //  __global__ void reduce(int num_rows, float *in, float out) {
  //    __shared__ float[32] cache;
  //    int offset = blockDim.x * blockIdx.x + threadIdx.x;
  //    if (offset >= num_rows) return;
  //    int tile_bound = std::min(offset + kTileSizeX, num_rows);
  //    float accum = 0;
  //    for (int i=offset; i<num_rows; i+= blockDim.x) {
  //      accum += in[i];
  //    }
  //    accum = warp_reduce(accum);
  //    if (threadIdx.x % WarpSize == 0) {
  //      cache[threadIdx.x / WarpSize] = accum;
  //    }
  //    __syncthreads();
  //    if (threadIdx.x / WarpSize == 0) {
  //      bool warp_exists = threadIdx.x < (blockDim.x / WarpSize);
  //      float block_accum = warp_exists ? cache[threadIdx.x % WarpSize] : 0;
  //      block_accum = warp_reduce(accum);
  //      if (threadIdx.x == 0) {
  //        out += block_accum;
  //      }
  //    }
  //  }
  // ```
  //
  // Column reduction uses the following algorithm:
  //
  // ```
  // void reduce(float** in, float* out) {
  //   __shared__ float[32][33] cache;
  //   int thread_id = GetThreadId();
  //   int block_id = GetBlockId();
  //   int tile_size = 128;
  //
  //   float accum = 0;
  //   for (int i=0; i<tile_size; i++) {
  //     accum += in[thread_id.y * tile_size + i][block_id * 32 + thread_id.x];
  //   }
  //   cache[thread_id.x][thread_id.y] = accum;
  //
  //   __syncthreads();
  //   accum = cache[thread_id.y][thread_id.x];
  //   accum = warp_reduce(accum); // Sum all the values of `accum` in the same
  //                               // warp.
  //
  //   if (thread_id.y % 32 == 0) {
  //     out[block_id * 32 + thread_id.x] = accum;
  //   }
  // }
  // ```
  //
  // Moreover, a heuristic is implemented to divide the reduce instructions
  // into groups for parallelization (see `DivideOutputInstructionsIntoGroups`
  // for details about the heuristic.) Reduce instructions in the same group
  // will run sequentially while different groups will run in parallel.
  //
  // we use raw block_id_y to select the reduce groups for execution without
  // complicating the index calculation in the code generation of the reduce
  // instructions. In other words, a block_id_y is assigned to a group and so
  // different groups can be run in parallel.
  Status EmitUnnestedReduction(mlir::lmhlo::FusionOp fusion,
                               HloComputation* fused_computation);

  // Emits a kernel for the given hlo instruction using a tiled 0-2-1 transpose
  // algorithm to improve the memory access patterns for the input parameters
  // with a shape that is a 0-2-1 transpose of the output tensor shape. The
  // caller is responsible for making sure that it is safe to apply the shared
  // memory transpose on the input parameters.
  //
  //
  // For the purpose of tiling, the output tensors have a logical shape of three
  // components 0-2-1 while the relevant input parameters have a logical shape
  // of three components 0-1-2 in the order major to minor. The x- and y-
  // dimensions of the tensors are tiled in square tiles with an edge length
  // `kTileSize`. Each thread block of `kTileSize` x `kNumRows` threads
  // transposes one tile: each thread copies kTileSize/kNumRows elements from
  // the input to a shared memory tile, then the otherwise "regular HLO kernel"
  // reads from the shared memory instead of the original input.
  //
  // This is similar to the following CUDA algorithm in TensorFlow:
  // https://goo.gl/MStRV6.
  //
  // `kTileSize` should usually be same as warp size. We currently choose 32 for
  // `kTileSize` and 4 for `kNumRows`. The CUDA algorithm uses 8 for `kNumRows`.
  //
  // TODO(b/33320379): Here each block transposes 1 tile. It may be more
  // efficient to launch fewer blocks so each transposes many tiles.
  Status EmitUnnestedTranspose(mlir::lmhlo::FusionOp fusion,
                               HloComputation* fused_computation);

  // Computes the KernelMappingScheme for the reduce HLO and indicates whether
  // the reduction is a row reduction. For an un-fused reduce op, unnested_hlo
  // and first_reduce are the same instruction. For a kInput fusion,
  // unnested_hlo is the fusion instruction while first_reduce is the first
  // reduce op.
  StatusOr<ReductionCodegenInfo> ComputeReductionCodegenInfo(
      mlir::lmhlo::FusionOp fusion, HloComputation* fused_computation,
      HloInstruction* first_reduce,
      const std::vector<std::vector<HloInstruction*>>& instr_index_groups);

  // Generates code for input-fusible slices.
  //
  // Prerequisite: ROOT is either a slice or a tuple of slices. The input shapes
  // of all ROOT slices need to be the same while their output shapes can be
  // different. On the other hand, the input ranges of slices can be
  // overlapping. Further generalization/specialization when the needs are seen
  // in the future.
  Status EmitInputFusibleNonStridedSlices(mlir::Operation* op);

  Status EmitElementForInputFusibleSlices(
      const HloComputation* fused_computation,
      absl::Span<const llvm_ir::IrArray> ir_arrays,
      const llvm_ir::IrArray::Index& index);

  // Emits code for an in-place scatter, modifying `thunk`s launch dimensions in
  // the process. Scatter indices are taken from `scatter_indices_gen`, updates
  // from `updates_gen`. The output buffer is expected to have the operand
  // values in it already. If unique_indices is false, we will use an atomic
  // update. Using true for unique_indices behaves properly only when it is
  // guaranteed that the indices to be updated do not overlap. The caller is
  // responsible for ensuring this is the case.
  Status EmitScatter(mlir::lmhlo::ScatterOp scatter,
                     const LaunchDimensions& launch_dimensions,
                     const llvm_ir::IrArray& output,
                     const llvm_ir::ElementGenerator& scatter_indices_gen,
                     const llvm_ir::ElementGenerator& updates_gen,
                     std::function<llvm::Type*(int64_t)> get_index_type);

  // Structure describing a scatter operation for IR emission.
  // TODO(jurahul): Migrate element generators to use MLIR.
  //                Migrate update_computation to be an MLIR Region.
  struct ScatterDescriptor {
    std::string name;
    Shape operand_shape;
    Shape scatter_indices_shape;
    Shape updates_shape;
    mlir::mhlo::ScatterDimensionNumbersAttr dim_numbers;
    bool unique_indices;
    const HloComputation* update_computation;
    llvm_ir::IrArray output;
    llvm_ir::ElementGenerator scatter_indices_gen;
    llvm_ir::ElementGenerator updates_gen;
    std::function<llvm::Type*(int64_t)> get_index_type;
  };

  // Emits code for an in-place scatter using the provided scatter operation
  // description.
  Status EmitScatter(const ScatterDescriptor& desc,
                     const LaunchDimensions& launch_dimensions);

  Status EmitTransposeTile(mlir::lmhlo::FusionOp fusion,
                           HloComputation* fusion_hlo,
                           absl::Span<const llvm_ir::IrArray> operand_arrays,
                           absl::Span<const llvm_ir::IrArray> output_arrays,
                           const TilingScheme& tiling_scheme,
                           const LaunchDimensions& launch_dimensions);

  Status EmitScatter(mlir::lmhlo::FusionOp fusion_op,
                     const HloComputation* fused_computation);

  Status EmitDynamicUpdateSlice(mlir::lmhlo::FusionOp fusion_op,
                                const HloComputation* fused_computation);

  struct TilingKernelInfo {
    // Tiling bounds.
    ValueVector2 output_tile_bounds;

    // Starting tile, as calculated from block id only.
    llvm_ir::IrArray::Index tile_origin;

    // Thread meta-info.
    ThreadIdInfo thread_id_info;
  };

  // Emits a kernel for the hlo instruction using the given kernel mapping
  // scheme.
  StatusOr<TilingKernelInfo> EmitTilingKernel(
      const TilingScheme& tiling_scheme, llvm::Type* index_ty,
      const TileElementGenerator& tile_element_generator);

  // Emits code to iterate through a 2-dimensional tile with a given tile
  // dimensions and given strides, and call the callback at each iteration.,
  //
  // thread_id_y` and `thread_id_x` are the intra-tile coordinates for
  // the first element to process, and `index` is the index for the origin of
  // the tile. Emits bounds check to ensure that each processed element
  // is within the boundary defined by `tile_dimensions`.
  //
  // Rough pseudocode:
  //
  // Given: tile_dimensions, x_offset, y_offset
  //
  // for (y = 0; y < tile_dimensions[0]; y += num_threads_y) {
  //   for (x = 0; x < tile_dimensions[1]; x++) {
  //
  //     y_pos = y_offset + y
  //     x_pos = x_offset + x * stride
  //
  //     if (x_loc < tile_width) {
  //       emit_elem_function(y_offset + y, x_loc);
  //     }
  //   }
  // }
  //
  void EmitTile(const TilingScheme& tiling_scheme,
                const llvm_ir::IrArray::Index& tile_origin_index,
                const ThreadIdInfo& thread_id_info,
                ValueVector2 tile_dimensions,
                const EmitElementFunction& emit_elem_function);

  // Creates accumulator alloca's, populates them with initial values, generates
  // __shared__ caches and returns the populated object.
  ReductionCodegenState GenerateReductionCodegenState(
      mlir::lmhlo::FusionOp fusion, const ReductionCodegenInfo& reduction_info,
      absl::Span<const HloReduceInstruction* const> reduce_instr_index_group,
      FusedIrEmitter& fused_emitter);

  // Wraps up the code generation for a tile block of a reduction kernel:
  // write the calculated output into the output tensor.
  void EmitReductionOutput(
      llvm::Type* index_ty, mlir::lmhlo::FusionOp fusion,
      absl::Span<const HloReduceInstruction* const> reduce_instr_index_group,
      const ReductionOutputMap& result_ir_arrays,
      const ReductionCodegenState& reduction_codegen_state,
      const TilingKernelInfo& tiling_kernel_info);

  // Returns the address to write the reduction output to.
  llvm::Value* GetOutputAddressForReduction(
      int partial_result_idx, llvm::Type* index_ty,
      const ReductionCodegenState& reduction_codegen_state,
      const TilingKernelInfo& tiling_kernel_info,
      const ReductionOutputMap& output_arrays,
      const HloReduceInstruction* reduction, int output_idx);

  // Performs the actual write of the reduction result.
  using TypedPointer = std::pair<llvm::Value* const, llvm::Type* const>;
  void WriteReductionOutput(
      llvm::Type* index_ty,
      const ReductionCodegenState& reduction_codegen_state,
      const TilingKernelInfo& tiling_kernel_info,
      const ReductionOutputMap& output_arrays,
      const HloReduceInstruction* reduction, int partial_result_idx,
      const absl::Span<TypedPointer const> values);

  // `current_output`: the value the tile has calculated.
  // `output_address`: address where the output value has to be written.
  void EmitReductionOutputForRowReduction(
      const TilingKernelInfo& tiling_kernel_info,
      const ReductionCodegenState& reduction_codegen_state,
      llvm::Type* index_ty, const ReductionOutputMap& output_arrays,
      const HloReduceInstruction* reduction, int partial_result_idx);

  // Same arguments as EmitReductionOutputForRowReduction.
  void EmitReductionOutputForColumnReduction(
      const TilingKernelInfo& tiling_kernel_info,
      const ReductionCodegenState& reduction_codegen_state,
      llvm::Type* index_ty, const ReductionOutputMap& output_arrays,
      const HloReduceInstruction* reduction, int partial_result_idx);

  // Emits code for reductions in the output_instructions.
  Status EmitIRForReduction(mlir::lmhlo::FusionOp fusion,
                            absl::Span<HloInstruction* const> instr_index_group,
                            FusedIrEmitter& fused_emitter,
                            const ReductionOutputMap& result_ir_arrays,
                            const ReductionCodegenInfo& reduction_info,
                            const Shape& input_shape);

  // Generate a single element of the tile (update the accumulator state) for a
  // given reducer of index `i`.
  void GenerateElementForReducer(
      const HloReduceInstruction* reduction, llvm::Value* partial_result_index,
      const ReductionCodegenState& codegen_state,
      const llvm_ir::IrArray::Index& index_without_linear,
      const llvm_ir::IrArray::Index& input_index, int num_partial_results,
      const ReductionOutputMap& result_ir_arrays);

  // Emits shuffle-down reduction for the `partial_result_address` using the
  // reduction computation `reducer`, writes output into
  // `partial_result_address`.
  //
  // Multiple partial_result_address inputs happen when doing variadic
  // reduction: each one should get the output value.
  void EmitFullWarpShuffleDownLoopForReduce(
      const HloComputation* reducer,
      absl::Span<TypedPointer const> partial_result_addresses,
      int threads_per_block, int num_results_per_warp = 1);

  // Allocates a shared tile of given dimensions, applying scaling specified in
  // tilng_scheme as a major-most dimension to avoid collisions.
  llvm::GlobalVariable* AllocateShared(
      const TilingScheme& tiling_scheme, llvm::Type* element_type,
      absl::Span<int64_t const> dimensions_major_to_minor,
      absl::string_view buffer_name = "");

  struct KernelArgument {
    int order;
    mlir::Value value;
    BufferSlice slice;
  };

  StatusOr<KernelArgument> ValueToKernelArgument(mlir::Value operand, int order,
                                                 bool is_written);

  // Build a kernel thunk, add it to list of thunks, and return IrArrays backing
  // kernel arguments.
  StatusOr<std::vector<llvm_ir::IrArray>> BuildKernelThunkImpl(
      absl::string_view name, Thunk::ThunkInfo thunk_info,
      std::vector<KernelArgument> kernel_arguments,
      const LaunchDimensions& launch_dimensions);

  StatusOr<std::vector<llvm_ir::IrArray>> BuildKernelThunk(
      mlir::Operation* op, mlir::ValueRange operands,
      const LaunchDimensions& launch_dimensions);

  StatusOr<std::vector<llvm_ir::IrArray>> BuildKernelThunk(
      mlir::Operation* op, const LaunchDimensions& launch_dimensions);

  // Generates the argument descriptors for a "reusable kernel".
  StatusOr<std::vector<ReusableKernelArgument>> GetReusableKernelArguments(
      mlir::lmhlo::FusionOp fusion_op);

  // Calculates a fingerprint of the kernel arguments, which can be used for
  // checking reusability.
  //
  // For example 2 arguments that are aligned to 16 bytes, aliased and also
  // written by the kernel will be represented as "16aw,16aw".
  //
  // Overlapping arguments are only marked aliased, if at least one of them is
  // written and their buffers are not exactly the same. If 2 arguments' buffers
  // are exactly the same, then they are not marked aliased, but marked as
  // duplicates, for example like this: "16,=0,16w,=2". The example means that
  // the 1st argument is the same as the 0th and the 3rd is the same as the 2nd.
  // These duplicated parameters are passed to the kernel only once.
  static std::string GetArgumentFingerprint(
      absl::Span<const ReusableKernelArgument> kernel_arguments);

  // Calculates the fingerprint of a (fused_computation, kernel_arguments,
  // discriminator) tuple.
  //
  // If a given fusion is implemented using multiple kernels, then for each
  // kernel we should provide a discriminator, such as "init" and "impl".
  //
  // If the same fingerprint is returned twice, then we can reuse the kernel
  // generated for the first computation.
  static std::string GetFingerprint(
      const HloComputation* fused_computation,
      absl::Span<const ReusableKernelArgument> kernel_arguments,
      absl::string_view discriminator = "");

  // Removes some unneeded defining operations from the calculation of `value`,
  // before passing it to a ReusableKernelThunk.
  static StatusOr<mlir::Value> RemoveTransformingOperations(mlir::Value value);

  // Creates a ReusableKernelThunk.
  StatusOr<ReusableKernelThunk*> BuildReusableKernelThunkImpl(
      absl::string_view kernel_name,
      absl::Span<const ReusableKernelArgument> kernel_arguments,
      Thunk::ThunkInfo thunk_info, const LaunchDimensions& launch_dimensions);

  // Builds a thunk that calls a new or a reused "reusable kernel".
  //
  // The caller must specify the same launch dimensions for fusions which have
  // the same computation.
  //
  // If a given fusion is implemented using multiple kernels, then for each
  // kernel we should provide a discriminator, such as "init" and "impl".
  //
  // This returns an std::nullopt if the kernel was
  // reused. In that case, the caller should not emit the code again for the
  // implementation of the kernel.
  //
  // This is the typical usage pattern of this method:
  //
  // ```
  // TF_ASSIGN_OR_RETURN(
  //   std::optional<std::vector<llvm_ir::IrArray>> opt_ir_arrays,
  //   BuildReusableKernelThunk(fusion_op, launch_dimensions));
  // if (!opt_ir_arrays.has_value()) {
  //   // The kernel was reused, no need to emit code.
  //   return OkStatus();
  // }
  // std::vector<llvm_ir::IrArray>& ir_arrays = opt_ir_arrays.value();
  //
  // EmitYourSpecificKernelCode(ir_arrays);
  // ```
  //
  // TODO(tdanyluk): Consider also using reusable kernels for kernel generating
  // operations which are not fusions.
  StatusOr<std::optional<std::vector<llvm_ir::IrArray>>>
  BuildReusableKernelThunk(mlir::lmhlo::FusionOp fusion_op,
                           const LaunchDimensions& launch_dimensions,
                           absl::string_view discriminator = "");

  // Returns a thunk that, given a reduce or select-and-scatter op,
  // initializes its memory to the appropriate initial value.
  std::unique_ptr<Thunk> BuildConstantInitializerThunk(
      mlir::Operation* op, absl::Span<const uint8_t> init_value,
      mlir::Value dest, const BufferAllocation::Slice& dest_slice,
      const Shape& output_shape);

  StatusOr<std::unique_ptr<Thunk>> TryBuildConstantInitializerThunk(
      mlir::Operation* op, mlir::Value init_value, mlir::Value dest);

  Status BuildInitializerThunk(mlir::Operation* op, mlir::Value init_value,
                               mlir::Value dest);
  Status BuildFusedInitializerThunk(mlir::lmhlo::FusionOp fusion,
                                    int output_index);

  // Returns a WhileThunk that invokes thunk sequences for 'condition' and
  // 'body' sub-computations of while instruction 'hlo'.
  StatusOr<std::unique_ptr<Thunk>> BuildWhileThunk(
      mlir::lmhlo::WhileOp while_op, const Thunk::ThunkInfo& thunk_info);

  // Returns a ForThunk which executes 'loop_limit' invocations of a thunk
  // sequence from the 'body' sub-computation of the while instruction 'hlo'.
  StatusOr<std::unique_ptr<Thunk>> BuildForThunk(
      mlir::lmhlo::WhileOp while_op, const Thunk::ThunkInfo& thunk_info,
      const int64_t loop_limit);

  // Returns a ConditionalThunk which executes the thunk sequence for the
  // 'branch_computation' corresponding to the predicate/branch_index of the
  // given conditional instruction.
  StatusOr<std::unique_ptr<Thunk>> BuildConditionalThunk(
      const HloInstruction* conditional);

  // Emits the LLVM values for thread_id, thread_id.x, thread_id.y and lane
  // id.
  //
  // Returns a struct containting these values.
  //
  // In the presence of thread scaling in tiling scheme may return early if the
  // combination of thread_id/block_id does not correspond to a real block.
  // Assumes the current function returns void.
  StatusOr<ThreadIdInfo> EmitThreadIdInfo(const TilingScheme& tiling_scheme,
                                          llvm::Type* index_ty);
  // Emit __syncthreads(), synchronization barrier for all threads in a block.
  llvm::CallInst* EmitSyncThreads();

  // Emits current thread id with the given type.
  //
  // Sets the return value range to [0, threads_per_block).
  llvm::Value* EmitThreadId(int64_t threads_per_block, llvm::Type* index_ty);

  // Emits current block id.
  llvm::Value* EmitBlockId(int32_t num_blocks, llvm::Type* index_ty);

  // Prints a given format string with the given arguments, prefixed with
  // thread id and block id, and postfixed with a newline.
  //
  // `thread_id_filter` and `block_id_filter`: if provided, restrict printing
  // to only given thread and/or block id.
  void EmitPrintfWithThreadId(
      absl::string_view fmt, absl::Span<llvm::Value* const> arguments,
      std::optional<int64_t> thread_id_filter = std::nullopt,
      std::optional<int64_t> block_id_filter = std::nullopt);

  // Prints the given index.
  void EmitPrintfForIndex(
      absl::string_view fmt, const llvm_ir::IrArray::Index& index,
      std::optional<int64_t> thread_id_filter = std::nullopt,
      std::optional<int64_t> block_id_filter = std::nullopt);

  StatusOr<HloComputation*> GetOrCreateSubComputationFromRegion(
      mlir::Region* region, bool is_fusion);

  Status AssertNonDeterminismIsOkay(const std::string& op_name);

  // The thunk sequence this IrEmitter generates for the input computation.
  ThunkSequence thunk_sequence_;

  // Maps async start ops to their executors so done can access the thunk.
  // Executor may be null if the start op is degenerate (so not emitted).
  absl::flat_hash_map<mlir::Operation*, NcclCollectiveThunk::AsyncExecutor*>
      async_executors_;

  // Begin optional members for XLA HLO -> LMHLO:
  absl::flat_hash_map<const mlir::Region*, std::unique_ptr<HloModule>>
      scratch_nested_computations_;
  // End optional members for XLA HLO -> LMHLO.

  // __shared__ memory uses a different address space, so we cast it to
  // global address space before writing or reading.
  llvm::Value* CastSharedToGlobal(llvm::Value* input, llvm::Type* element_type,
                                  llvm::Twine name = "");

  // Returns the ShapedSlices for the given operands.
  StatusOr<std::vector<ShapedSlice>> GetShapedSlices(
      mlir::Operation::operand_range operands);

  // Returns the buffer allocation Slice for the given operands.
  StatusOr<std::vector<BufferAllocation::Slice>> GetSlices(
      mlir::Operation::operand_range operands);

  GpuElementalIrEmitter elemental_emitter_;

  // Maps computation fingerprints generated by GetFingerprint() to the first
  // ReusableKernelThunk generated for them.
  absl::flat_hash_map<std::string, ReusableKernelThunk*> kernel_reuse_cache_;
};

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_IR_EMITTER_UNNESTED_H_
