# Programmer's Guide

The documents in this unit dive into the details of writing TensorFlow
code.  This section begins with the following guides, each of which
explain a particular aspect of TensorFlow:

  * @{$variables$Variables: Creation, Initialization, Saving, Loading, and
     Sharing}, which details the mechanics of TensorFlow Variables.
  * @{$dims_types$Tensor Ranks, Shapes, and Types}, which explains Tensor
    rank (the number of dimensions), shape (the size of each dimension),
    and datatypes.
  * @{$threading_and_queues$Threading and Queues}, which explains TensorFlow's
    rich queuing system.
  * @{$reading_data$Reading Data}, which documents three different mechanisms
    for getting data into a TensorFlow program.

The following guide is helpful when training a complex model over multiple
days:

  * @{$supervisor$Supervisor: Training Helper for Days-Long Trainings}, which
    explains how to gracefully handle system crashes during a lengthy training
    session.

TensorFlow provides a debugger named `tfdbg`, which is documented in the
following guide:

  * @{$debugger$Debugging TensorFlow Programs},
    which walks you through the use of `tfdbg` within an application. It covers
    using `tfdbg` with both the low-level TensorFlow API and the Estimator API.

To learn about the TensorFlow versioning scheme consult:

  * @{$version_compat$The TensorFlow Version Compatibility Guide}, which explains
TensorFlow's versioning nomenclature and compatibility rules.

We conclude this section with a FAQ about TensorFlow programming:

  * @{$faq$Frequently Asked Questions}
