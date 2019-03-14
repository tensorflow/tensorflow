# TensorFlow Lite 2019 Roadmap

**Updated: March 6th, 2019**

The following represents a high level overview of our 2019 plan. You should be
conscious that this roadmap may change at anytime relative to a range of factors
and the order below does not reflect any type of priority. As a matter of
principle, we typically prioritize issues that the majority of our users are
asking for and so this list fundamentally reflects that.

We break our roadmap into four key segments: usability, performance,
optimization and portability. We strongly encourage you to comment on our
roadmap and provide us feedback in the TF Lite discussion groups and forums.

## Usability

*   **More ops coverage**
    *   Prioritize many more ops based on user feedback
*   **Op versioning & signatures**
    *   Op kernels will get version numbers
    *   Op kernels will be identifiable by signature
*   **New Convertor**
    *   Implementing a new TensorFlow Lite convertor that will better handle
        graph conversion (i.e. control flow, conditionals etc) and replace TOCO
*   **Continue to improve TF Select Ops**
    *   Support more types of conversion utilizing TF Selects such as hash
        tables, strings etc.
    *   Support smaller binary size when using select TF ops via op stripping
*   **LSTM / RNN support**
    *   Add full support of conversion for LSTMs and RNNs
*   **Graph Visualization Tooling**
    *   Provide enhanced graph visualization tooling
*   **Pre-and-post processing support**
    *   Add more support for pre-and-post processing of inference
*   **Control Flow & Training on-device**
    *   Add support for control flow related ops
    *   Add support for training on-device
*   **New APIs**
    *   New C API as core for language bindings and most clients
    *   Objective-C API for iOS
    *   SWIFT API for iOS
    *   Updated Java API for Android
    *   C# Unity language bindings
*   **Add more Models**
    *   Add more models to the support section of the site

## Performance

*   **More hardware delegates**
    *   Add support for more hardware delegates
*   **Support NN API**
    *   Continually support and improve support for NN API
*   **Framework Extensibility**
    *   Enable simplistic overwriting of CPU kernels with customized optimized
        versions
*   **GPU Delegate**
    *   Continue to extend the total support ops for OpenGL and Metal ops
    *   Open-source
*   **Improve TFLite CPU performance**
    *   Optimizations for float and quantized models

## Optimization

*   **Model Optimization Toolkit**
    *   Post training quantization + hybrid kernels
    *   Post Training quantization + fixed-point kernels
    *   Training with quantization
*   **More support for more techniques**
    *   RNN Support
    *   Sparsity/Pruning
    *   Lower bit-width support

## Portability

*   **Microcontroller Support**
    *   Add support for a range of 8-bit, 16-bit and 32-bit MCU architecture use
        cases for Speech and Image Classification
