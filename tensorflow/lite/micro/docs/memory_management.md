<!-- mdformat off(b/169948621#comment2) -->

<!--
Semi-automated TOC generation with instructions from
https://github.com/ekalinin/github-markdown-toc#auto-insert-and-update-toc
-->

<!--ts-->

*   [Memory Management in TensorFlow Lite Micro](#memory-management-in-tensorflow-lite-micro)
    *   [Tensor Arena](#tensor-arena)
        *   [Head Section](#head-section)
        *   [Temporary Section](#temporary-section)
        *   [Tail Section](#tail-section)
    *   [Recording Memory APIs](#recording-memory-apis)
        *   [Allocation Section Details](#allocation-section-details)

<!-- Added by: advaitjain, at: Mon 05 Oct 2020 02:21:02 PM PDT -->

<!--te-->

# Memory Management in TensorFlow Lite Micro

This document outlines how memory is managed internally by TensorFlow Lite Micro
(TFLM) today. It outlines the "online" allocation strategy used by the default
TFLM APIs for loading a model into a shared tensor arena.

## Tensor Arena

The main "working" space for TFLM allocations is inside a single `char` or
`int8_t` buffer. This buffer can be managed by passing it directly into a
`tflite::MicroInterpreter` constructor or through a `tflite::MicroAllocator`
instance that can be passed into a `tflite::MicroInterpreter` constructor.
Internally, the `tflite::MicroAllocator` classifies allocations into 3 different
sections:

*   **Head** - non-persistent allocations.
*   **Temporary** - short term "scoped" allocations.
*   **Tail** - persistent allocations.

The illustration below represents typical allocations in TFLM:

```
--------------------------------------------------------------------------------
|        |                     |                                               |
|  HEAD  |<--  TEMPORARY    -->|                    TAIL                       |
|        |                     |                                               |
--------------------------------------------------------------------------------
* Lowest Address                                               Highest Address *
```

### Head Section

This non-persistent section typically holds shared Tensor buffers. This section
does not allocate small iterative chunks, it can only be set by a specific
length for the entire section.

This allocation length of this section is managed by the
`tflite::GreedyMemoryPlanner`. That memory planner looks at the entire graph of
a model and tries to reuse as many buffers as possible to create the smallest
length for the head. The Tensor buffers for this section can be accessed via a
`TfLiteEvalTensor` or `TfLiteTensor` instance on the `tflite::MicroInterpreter`.

### Temporary Section

This section is used to allocate "scoped" or short-term, non-guaranteed buffers.
Allocations from this section start from the current end address of the head
section and grow towards the tail section. An allocation chain can be reset (and
must be reset before adjusting the head) and moves the current allocation start
address back to the end of the head section.

TFLM currently uses these allocations for a scope allocation of large C structs
or scratch memory that is expected to be valid for at least the lifetime of a
method call. This section.

### Tail Section

This section holds all persistent allocations used by TFLM. This section
contains many random sized allocations and grows towards the end of the head
section. Allocations in this section come from a variety of areas inside of
TFLM. TFLM provides a [recording API](#Recording-Memory-APIs) to assist with
auditing the contents of this section.

## Recording Memory APIs

TFLM provides simple APIs for auditing memory usage in the shared tensor arena.
These APIs are opt-in and require some additional memory overhead and a working
debug logging implementation
[(reference implementation)](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/micro/debug_log.cc).

A typical bare-bones TFLM interpreter setup looks as such:

```c++
// Buffer for the tensor arena:
size_t tensor_arena_size = 2048;
uint8_t tensor_arena[tensor_arena_size];

// Interpreter using the shared tensor arena above:
tflite::MicroInterpreter interpreter(
  tflite::GetModel(my_model_data), ops_resolver,
  tensor_arena, tensor_arena_size, error_reporter);

// Invoke one time which will allocate internals:
if (interpreter.Invoke() != kTfLiteOk) {
  TF_LITE_REPORT_ERROR(error_reporter, "Exception during invoke()!");
}
```

Recording API can simply be used by including the `RecordingMicroInterpreter`
class (`recording_micro_interpreter.h`) and replace `tflite::MicroInterpreter`
with `tflite::RecordingMicroInterpreter`. The same call to `invoke()` is
performed, but another call is made to `PrintAllocations()` which will output
detailed allocation logging:

```c++
// Add an include to the recording API:
#include "recording_micro_interpreter.h"

// Simply change the class name from 'MicroInterpreter' to 'RecordingMicroInterpreter':
tflite::RecordingMicroInterpreter interpreter(
  tflite::GetModel(my_model_data), ops_resolver,
  tensor_arena, tensor_arena_size, error_reporter);

// Invoke one time which will allocate internals:
if (interpreter.Invoke() != kTfLiteOk) {
  TF_LITE_REPORT_ERROR(error_reporter, "Exception during invoke()!");
}

// Print out detailed allocation information:
interpreter.PrintAllocations();
```

The output of this call will look something similar to this (output from the
[memory_arena_threshold_test](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/micro/memory_arena_threshold_test.cc#L205)):

```bash
[RecordingMicroAllocator] Arena allocation total 9568 bytes
[RecordingMicroAllocator] Arena allocation head 7744 bytes
[RecordingMicroAllocator] Arena allocation tail 1824 bytes
[RecordingMicroAllocator] 'TfLiteEvalTensor data' used 360 bytes with alignment overhead (requested 360 bytes for 15 allocations)
[RecordingMicroAllocator] 'Persistent TfLiteTensor data' used 0 bytes with alignment overhead (requested 0 bytes for 0 tensors)
[RecordingMicroAllocator] 'Persistent TfLiteTensor quantization data' used 0 bytes with alignment overhead (requested 0 bytes for 0 allocations)
[RecordingMicroAllocator] 'TfLiteTensor variable buffer data' used 0 bytes with alignment overhead (requested 0 bytes for 0 allocations)
[RecordingMicroAllocator] 'NodeAndRegistration struct' used 392 bytes with alignment overhead (requested 392 bytes for 7 NodeAndRegistration structs)
[RecordingMicroAllocator] 'Operator runtime data' used 136 bytes with alignment overhead (requested 136 bytes for 5 OpData structs)
```

### Allocation Section Details

More information about each recorded allocation section:

*   'TfLiteEvalTensor data'
    *   C struct that holds the data type, dimension, and a pointer to the
        buffer representing the Tensor.
*   'Persistent TfLiteTensor data'
    *   C struct that holds more information than a `TfLiteEvalTensor` struct in
        the graph.
    *   Allocations in this bucket will only show up when accessing tensors from
        the accessors on `tflite::MicroInterpreter`.
*   'Persistent TfLiteTensor quantization data'
    *   Length of persistent quantization data assigned to persistent
        `TfLiteTensor` structs.
    *   Allocations in this bucket will only show up when accessing tensors from
        the accessors on `tflite::MicroInterpreter`.
*   'TfLiteTensor variable buffer data'
    *   Length of buffer data from a variable tensor (retains data throughout
        calls to `invoke()`).
*   'NodeAndRegistration struct'
    *   C struct that holds a `TfLiteRegistration` and `TfLiteNode` struct
        instance.
    *   Each operator in a model will contain one `NodeAndRegistration` struct.
*   'Operator runtime data'
    *   Persistent allocations of data cached by TFLM kernels (e.g. quantization
        params, multipliers, etc).
