<!--ts-->

*   [Online Memory Allocation Overview in TensorFlow Lite Micro](#online-memory-allocation-overview-in-tensorflow-lite-micro)
    *   [Arena](#arena)
    *   [Existing buffers in the flatbuffer](#existing-buffers-in-the-flatbuffer)
    *   [Model Init Phase](#model-init-phase)
    *   [Model Prepare Phase](#model-prepare-phase)
    *   [Finish Model Allocation Phase](#finish-model-allocation-phase)

<!-- Added by: kreeger, at: Wed Apr 28 10:52:04 CDT 2021 -->

<!--te-->

# Online Memory Allocation Overview in TensorFlow Lite Micro

This document outlines how "online" memory is managed in TensorFlow Lite Micro
(TFLM).

## Arena

Online memory planning strategically places allocations in a single `uint8_t`
buffer array. The buffer is split into two main sections: the “head” and the
“tail”. Generally, non-persistent allocations are placed in the “head” and
persistent allocations are placed in the “tail”. More details about the arena
can be [found here](memory_management.md#tensor-arena).

## Existing buffers in the flatbuffer

The TFLite flatbuffer model contains a variety of information required to run a
model in TFLite or TFLM. The TFLM online memory planner will walk the main
subgraph and find all tensors required for the model (represented as
`TfLiteTensor` and `TfLiteEvalTensor` C structs at runtime). Persistent tensors
in the flatbuffer (e.g. weight tensors) will point at a buffer inlined in the
flatbuffer. These buffers are reused during online memory planning. The
corresponding C structures will point back at the buffer packed into the
flatbuffer.

## Model Init Phase

Either through the first call of `MicroInterpreter::Invoke()` or an explicit
call to `MicroInterpreter::AllocateTensors()` the online model allocation will
begin. The `MicroInterpreter` instance will invoke
`MicroAllocator::StartModelAllocation()`. This function will begin pulling data
out of the serialized flatbuffer and begin walking through the main subgraph.

The method `MicroAllocator::StartModelAllocation()` begins allocation in the
following order: * Initializes internal state for scratch buffer allocations *
Allocates a list of `TfLiteEvalTensor` C structs based on the number of tensors
in the subgraph. * Allocations are persistent and stored in the tail section. *
Tensors that reference buffers in the flatbuffer are assigned at this point. *
Allocates a list of `TfLiteRegistration` and `TfLiteNode` C structs for every
operator in the model subgraph * Allocations are persistent and stored in the
tail section. * Walks back through the list of subgraph operators and assigns
all C structs with relevant information from the flatbuffer.

At the conclusion of this phase, the operator kernel implementations are ready
for calls to the `TfLiteRegistration::init()` function. The `MicroInterpreter`
walks through the operator list and invokes all operator implementations that
have this function. Typically, operator implementations return the object to
store in the `user_data` field of a `TfLiteNode` struct.

## Model Prepare Phase

After the interpreter has initialized all operator kernels, another pass through
the subgraph is done. This time, each operator implementations that provides a
`TfLiteRegistration::prepare()` function is called. This phase in TFLM is used
for kernels to verify capabilities from model information, validate shapes,
allocate any scratch buffers requested (through
`TfLiteContext::GetScratchBuffer()`), and calculate quantization runtime data.

At this time, operator implementation will request tensor data through the
`TfLiteTensor` C struct. This struct is heavier and contains more information
that operators will need during this phase of initialization. Internally, TFLM
will allocate these instances per request in the temp section. The temp section
is the space between the head and the tail in the arena. During the prepare
phase, nothing is yet been placed in the head section. This extra space between
the head and tail is used to allocate buffers that are available until
`MicroAllocator::ResetTempAllocations()` is called. Additional information
[available here](memory_management.md#temporary-section).

NOTE: The `TfLiteTensor` struct is only available in TFLM during
`TfLiteRegistration::prepare()`, after this allocation phase tensor data can
only be accessed via a `TfLiteEvalTensor` struct.

Additionally, at this time each operator implementation may request scratch
buffer requests through `TfLiteContext::RequestScratchBufferInArena()`. These
requests are limited to `kMaxScratchBuffersPerOp` and are stored in an instance
variable for each operator prepare block. All requests are eventually moved to
the head section when the interpreter moves to the next operator.

After each call to `TfLiteRegistration::prepare()` the `MicroInterpreter` calls
`MicroAllocator::FinishPrepareNodeAllocations()`. This method resets temp
allocations and begins to store all scratch buffer requests inside the head
section of the arena.

After all operators have been prepared, the `MicroInterpreter` calls
`MicroAllocator::FinishModelAllocation()` to begin finalizing the online memory
plan.

## Finish Model Allocation Phase

The last phase of online memory planning is handled in
`MicroAllocator::FinishModelAllocation()`. This function performs the following
tasks

*   Allocates space in the tail for all persistent buffer requests that are
    currently in the head.
*   Commits Static Memory Plan
    *   Uses the `GreedyMemoryPlanner` to optimize the non-persistent space in
        the head.
    *   Optimizes for the operator that requires the largest byte-width buffer.
    *   Allocates pointers in the tail that provide pointers into shared space
        and offsets in the head.
    *   Sets the size of the head based on the result of
        `GreedyMemoryPlanner::GetMaxiumMemorySize()`.
*   Allocates variable tensor buffers in the tail section.

Once TFLM has finalized online model allocation, all buffers are prepared and
ready for optimal speed for inference. The system no longer enables operator
implementations to allocate scratch buffers after this point.
