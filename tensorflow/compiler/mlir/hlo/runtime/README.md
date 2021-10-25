## DISC: DynamIc Shape Compiler

DISC is an mlir-hlo based e2e compiler including both compiler side and runtime
side. For runtime side, we have different targeting environments (e.g.
tensorflow, pytorch, or sometimes even a standalone binary). In order to
simplify the design of the compiler side, we design a Runtime Abstraction Layer
(RAL) to separate the compiler side and runtime side. Thus the compiler side
only need to target RAL itself and it is the responsibility of RAL to handle the
differences between different targeting environments.

Another function of RAL is to manage stateful resources. To this end, it
provides a context object, and hides all stateful operations behind this
context, thus the compiler side itself doesn't need to care about the resource
initialization. For example, a kernel must be loaded before it can be launched
on GPU. However, the loading operation should only be taken once during the
whole lifetime of the context in order to achieve the best performance. Based on
the initialization-free interfaces (e.g.
load_kernel_if_not_and_then_launch_kernel) provided by RAL, compiler side can
focus on its core optimization logic and lets the RAL to manage the resource
status.

This directory contains the part of code to implement RAL. We suppose the code
will be integrated to the AI framework (e.g. TF, PyTorch) plugin. Thus, it'd be
better to not introduce other dependency.
