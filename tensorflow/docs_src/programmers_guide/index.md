# Programmer's Guide

The documents in this unit dive into the details of writing TensorFlow
code.  This section begins with the following guides, each of which
explain a particular aspect of TensorFlow:

  * @{$variables$Variables: Creation, Initialization, Saving, and Loading},
    which details the mechanics of TensorFlow Variables.
  * @{$dims_types$Tensor Ranks, Shapes, and Types}, which explains Tensor
    rank (the number of dimensions), shape (the size of each dimension),
    and datatypes.
  * @{$variable_scope$Sharing Variables}, which explains how to share and
    manage large sets of variables when building complex models.
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

A `MetaGraph` consists of both a computational graph and its associated
metadata.  A `MetaGraph` contains the information required to continue
training, perform evaluation, or run inference on a previously
trained graph.  The following guide details `MetaGraph` objects:

  * @{$meta_graph$Exporting and Importing a MetaGraph}.

`SavedModel` is the universal serialization format for Tensorflow models. TensorFlow provides SavedModel CLI (command-line interface) as a tool to inspect and execute a MetaGraph in a SavedModel. The detailed usages and examples are
documented in the following guide:

  * @{$saved_model_cli$SavedModel CLI (Command-Line Interface)}.

To learn about the TensorFlow versioning scheme, consult the following two
guides:

  * @{$version_semantics$TensorFlow Version Semantics}, which explains
    TensorFlow's versioning nomenclature and compatibility rules.
  * @{$data_versions$TensorFlow Data Versioning: GraphDefs and Checkpoints},
    which explains how TensorFlow adds versioning information to computational
    graphs and checkpoints in order to support compatibility across versions.

We conclude this section with a FAQ about TensorFlow programming:

  * @{$faq$Frequently Asked Questions}
