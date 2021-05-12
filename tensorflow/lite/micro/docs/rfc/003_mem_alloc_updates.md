# TensorFlow Lite for Microcontrollers Memory Allocations Update

| Status        | Proposed                                                    |
:-------------- |:----------------------------------------------------------- |
| **RFC #3**    |                                                             |
| **Author(s)** | Bhanu Prakash Bandaru Venkata (bhanup@cadence.com)          |
|               | Mayur Jagtap (mayurj@cadence.com)                           |
|               | Niranjan Yadla (nyadla@cadence.com)                         |
|               | Raj Pawate (pawateb@cadence.com)                            |
|               | Vijay Pawar (vpawar@cadence.com)                            |
| **Sponsor**   | Pete Warden (petewarden@google.com)                         |
| **Updated**   | 2021-05-12                                                  |

## Background

TensorFlow Lite for Microcotrollers (TFLM) makes deploying of neural networks on DSP very easy. For edge devices, DSP would generally run multiple neural networks simultaneously and would also run DSP workload for frontend decode of neural networks, music playback, speech encode - decode etc. To enable quick and efficient deployment of these mix workloads on DSP, some framework would be useful. We are planning to use Cadence's Xtensa Audio Framework (XAF) for this purpose.
The current memory allocation scheme in TFLM is not favorable for integration into XAF, so following updates are being proposed.

## Current TFLM Memory Allocations
In current TFLM implementation, one large tensor arena buffer is passed to MicroInterpreter. MicroInterpreter then allocates various tensors and buffers required for the network from this arena through init and prepare calls of operators. Scratch allocations are done from the head of the arena and persistent allocations are done from the tail of the arena.
Note, scratch memory is used by the network during one invoke call and can be reused outside of the invoke call whereas persistent memory is used by the network during its lifetime and cannot be reused for any other purpose.
While, the current allocation scheme is clean and easy to use, following issues can be observed:
- Input and output tensors of the network are within arena. Application / upper layer would need to copy data to and from these buffers on each invoke call.
- It may not be possible to reuse scratch memory for other DSP workload.
- The upfront memory requirement of network is not known.

## Proposed Updates in TFLM Memory Allocations
Following updates in TFLM memory allocation scheme would really make it efficient and more attractive for embedded frameworks.
- Provide a query API to get input, output, scratch and persistent memory requirements of the network at init time.
- Allow scratch and persistent buffers to be separate and respective buffer pointers would be passed to MicroInterpreter.
- Allow input and output buffers to be allocated by application / upper layer and respective buffer pointers would be passed to MicroInterpreter.

A query API can be something like below:
```
tflite::void GetMemRequirements(const Model* model, 
                                int32_t* inp_sz,
                                int32_t* out_sz,
                                int32_t* persist_sz,
                                int32_t* scratch_sz);
```
Application / upper layer would allocate buffers of sizes returned by above API.  The MicroInterpreter constructor should have a variant which accepts separate pointers for input, output, scratch and persist buffers as below.
```
MicroInterpreter(const Model* model, 
                 const MicroOpResolver& op_resolver,
                 uint8_t* p_inp, 
                 uint8_t* p_out,
                 uint8_t* p_persist, 
                 uint8_t* p_scratch,  
                 ErrorReporter* error_reporter,
                 tflite::Profiler* profiler = nullptr);
```

