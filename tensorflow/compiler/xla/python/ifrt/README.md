IFRT (the Interim Framework Runtime) is a high-level ML runtime API that is
designed to be used as the interface between a user-facing framework, such as
JAX, PyTorch, or TensorFlow, and the runtimes below. The purpose of the IFRT
is to make a framework portable across as wide a range of hardware
configurations as possible. IFRT allows the framework to more or less
declaratively express the work that needs to be done, and delegate policy
choices about how to efficiently execute that work to the runtime
implementations.

The status quo is that PjRt ("pretty much just another runtime"), which is a
low-level ML runtime API for XLA computations, doubles as a high-level ML
runtime API in several deployment scenarios. This low-level ML runtime API
was originally designed to abstract away implementation details of underlying
hardware attached to a single local host, for example masking minor
differences between generations of TPUs/GPUs and CPUs. However, PjRt nowadays
is being asked to embrace vastly different scenarios such as scaling to
distributed execution spanning thousands of accelerators on modern shared
infrastructure. We believe that we have reached a point where we should
bifurcate the current PjRt API, and allow the PjRt API and a new API, IFRT,
to deviate to better support their intended use cases. Initially, because we
are bifurcating the APIs rather than inventing a fundamentally new API, it is
a design goal that it will be easy to migrate from PjRt to IFRT. Over time as
the APIs deviate, such migration may become harder.

IFRT requires careful prototyping as it is a new portable layer that would
interact with multiple user-facing frameworks and low-level runtimes. Our
initial prototyping effort aims at demonstrating that an IFRT implementation
for PjRt enables existing PjRt users to use IFRT with little friction,
removing several complexities caused by direct interaction with low-level
runtime APIs. An IFRT prototype in the XLA source tree will accelerate this
prototyping effort by sharing the building and testing infrastructure with
XLA and will ensure coherent development of XLA and IFRT.

Once early IFRT prototyping is complete we will consult with stakeholders to
ensure both that all requirements are met, and also that there is a simple
migration path from PjRt to IFRT for teams who will benefit from migration.
