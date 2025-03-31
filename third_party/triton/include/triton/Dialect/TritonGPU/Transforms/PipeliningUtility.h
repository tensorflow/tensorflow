#ifndef TRITON_TRITONGPU_TRANSFORMS_PIPELINER_PIPELINING_UTILITY_H_
#define TRITON_TRITONGPU_TRANSFORMS_PIPELINER_PIPELINING_UTILITY_H_

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include <optional>
#include <utility>
#include <vector>

namespace mlir {
class ImplicitLocOpBuilder;
namespace triton {

static const char *kNumStagesAttrName = "tt.num_stages";
static const char *kDisallowAccMultiBufferAttrName =
    "tt.disallow_acc_multi_buffer";
static const char *kLoopStageAttrName = "loop.stage";
static const char *kLoopClusterAttrName = "loop.cluster";
static const char *kScheduledMaxStageAttrName = "tt.scheduled_max_stage";
static const char *kLatencyAttrName = "tt.latency";

bool loopHasDistGreaterThanOne(scf::ForOp forOp);
bool isOuterLoop(scf::ForOp forOp);

Value getPredMask(RewriterBase &rewriter, Type typeLike, Value currentMask,
                  Value pred);

/// Function to mask operations during scheduling.
Operation *predicateOp(RewriterBase &rewriter, Operation *op, Value pred);

/// Replace all uses of `oldUse` with `val` and propagate the type if needed.
/// This is useful when we need to change a memory descriptor from immutable to
/// mutable.
void replaceUsesAndPropagateType(OpBuilder &builder, Operation *oldUse,
                                 Value val);

// Return true if the given ForOp has the attribute
// `tt.disallow_acc_multi_buffer` set to true.
bool getDisallowAccMultiBuffer(scf::ForOp forOp);

/// Visit the operands of `op` and the operands of any nested ops defined
/// outside of `op`.
void visitNestedOperands(Operation *op,
                         function_ref<void(OpOperand &)> visitor);
/// Visit the operands of `op` and the operands of any nested ops defined
/// outside of `op`.
void visitNestedOperands(Operation *op, function_ref<void(Value)> visitor);
/// Get the operands of `op` and the operands of any nested ops defined outside
/// of `op`.
SetVector<Value> getNestedOperands(Operation *op);

// Return the definition of the given value. If the value is a loop-carried
// dependency, return the definition and the distance to it.
std::pair<OpResult, int64_t> getDefinitionAndDistance(scf::ForOp forOp,
                                                      Value value);
// Return the defining op of the given value, if the Value is an argument of the
// loop return the associated defining op in the loop and its distance to the
// Value.
std::pair<Operation *, int64_t> getDefiningOpAndDistance(scf::ForOp forOp,
                                                         Value value);

// Return maxumum length of the vectorized copy between registers and shared
// memory for the given tensor type and shared encoding.
int getCopyVecBytes(RankedTensorType registerTy,
                    gpu::SharedEncodingTrait sharedEnc);

// Serialize the latencies of the operations in the loops into the latency
// attribute.
void serializeLatencies(ModuleOp module, DenseMap<Operation *, int> &opLatency);

// Deserialize the latencies of the operations in the loops from the attribute.
DenseMap<Operation *, int> deserializeLatencies(Operation *op);

// Given a result of MemDescSubview, or Alloca, create a MemDescSubview with a
// single buffer slice (leading dimension equal to 1), at the given index.
Value createSingleBufferView(OpBuilder &builder, Value alloc, Value idx);
Value createSingleBufferView(OpBuilder &builder, Value alloc, int idx);

// Create an allocation for multibuffered scalars.
Value createScalarAlloc(ImplicitLocOpBuilder &rewriter, Type type,
                        unsigned numBuffers);
// Create an allocation and init the mbarriers.
Value createBarrierAlloc(scf::ForOp forOp, int numBarriers);
// Create an allocation that can hold distance number of tensor shapes.
Value createAlloc(scf::ForOp forOp, RankedTensorType ty, Location loc,
                  gpu::SharedEncodingTrait sharedEnc, unsigned distance);

// Determine if the operation is a TMA load.
bool isTMALoad(Operation *op);

// Get the type of the view of a multi-buffered tensor value.
gpu::MemDescType getBufferViewType(gpu::MemDescType allocTy);
// Get a generic shared encoding for a tensor.
gpu::SharedEncodingTrait getSharedEncoding(RankedTensorType ty);
// Get a shared encoding for a tensor based on its uses.
gpu::SharedEncodingTrait getSharedEncoding(Operation *loadOp);

// Erase the given loop carried values from the loop, where `loop` is replaced
// with a new loop.
void eraseLoopCarriedValues(scf::ForOp &loop, llvm::BitVector indices);

// Get the number of stages to pipeline the loop with, if it is explicitly
// specified.
int getNumStagesOrDefault(scf::ForOp forOp, int defaultNumStages);

} // namespace triton
} // namespace mlir

#endif // TRITON_TRITONGPU_TRANSFORMS_PIPELINER_PIPELINING_UTILITY_H_
