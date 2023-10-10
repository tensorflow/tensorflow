# Using TensorFlow Securely

This document discusses the TensorFlow security model. It describes the security
risks to consider when using models, checkpoints or input data for training or
serving. We also provide guidelines on what constitutes a vulnerability in
TensorFlow and how to report them.

This document applies to other repositories in the TensorFlow organization,
covering security practices for the entirety of the TensorFlow ecosystem.

## TensorFlow models are programs

TensorFlow
[**models**](https://developers.google.com/machine-learning/glossary/#model) (to
use a term commonly used by machine learning practitioners) are expressed as
programs that TensorFlow executes. TensorFlow programs are encoded as
computation
[**graphs**](https://developers.google.com/machine-learning/glossary/#graph).
Since models are practically programs that TensorFlow executes, using untrusted
models or graphs is equivalent to running untrusted code.

If you need to run untrusted models, execute them inside a
[**sandbox**](https://developers.google.com/code-sandboxing). Memory corruptions
in TensorFlow ops can be recognized as security issues only if they are
reachable and exploitable through production-grade, benign models.

### Compilation

Compiling models via the recommended entry points described in
[XLA](https://www.tensorflow.org/xla) and
[JAX](https://jax.readthedocs.io/en/latest/jax-101/02-jitting.html)
documentation should be safe, while some of the testing and debugging tools that
come with the compiler are not designed to be used with untrusted data and
should be used with caution when working with untrusted models.

### Saved graphs and checkpoints

When loading untrusted serialized computation graphs (in form of a `GraphDef`,
`SavedModel`, or equivalent on-disk format), the set of computation primitives
available to TensorFlow is powerful enough that you should assume that the
TensorFlow process effectively executes arbitrary code.

The risk of loading untrusted checkpoints depends on the code or graph that you
are working with. When loading untrusted checkpoints, the values of the traced
variables from your model are also going to be untrusted. That means that if
your code interacts with the filesystem, network, etc. and uses checkpointed
variables as part of those interactions (ex: using a string variable to build a
filesystem path), a maliciously created checkpoint might be able to change the
targets of those operations, which could result in arbitrary
read/write/executions.

### Running a TensorFlow server

TensorFlow is a platform for distributed computing, and as such there is a
TensorFlow server (`tf.train.Server`). The TensorFlow server is intended for
internal communication only. It is not built for use in untrusted environments
or networks.

For performance reasons, the default TensorFlow server does not include any
authorization protocol and sends messages unencrypted. It accepts connections
from anywhere, and executes the graphs it is sent without performing any checks.
Therefore, if you run a `tf.train.Server` in your network, anybody with access
to the network can execute arbitrary code with the privileges of the user
running the `tf.train.Server`.

## Untrusted inputs during training and prediction

TensorFlow supports a wide range of input data formats. For example it can
process images, audio, videos, and text. There are several modules specialized
in taking those formats, modifying them, and/or converting them to intermediate
formats that can be processed by TensorFlow.

These modifications and conversions are handled by a variety of libraries that
have different security properties and provide different levels of confidence
when dealing with untrusted data. Based on the security history of these
libraries we consider that it is safe to work with untrusted inputs for PNG,
BMP, GIF, WAV, RAW, RAW\_PADDED, CSV and PROTO formats. All other input formats,
including tensorflow-io should be sandboxed if used to process untrusted data.

For example, if an attacker were to upload a malicious video file, they could
potentially exploit a vulnerability in the TensorFlow code that handles videos,
which could allow them to execute arbitrary code on the system running
TensorFlow.

It is important to keep TensorFlow up to date with the latest security patches
and follow the sandboxing guideline above to protect against these types of
vulnerabilities.

## Security properties of execution modes

TensorFlow has several execution modes, with Eager-mode being the default in v2.
Eager mode lets users write imperative-style statements that can be easily
inspected and debugged and it is intended to be used during the development
phase.

As part of the differences that make Eager mode easier to debug, the [shape
inference
functions](https://www.tensorflow.org/guide/create_op#define_the_op_interface)
are skipped, and any checks implemented inside the shape inference code are not
executed.

The security impact of skipping those checks should be low, since the attack
scenario would require a malicious user to be able to control the model which as
stated above is already equivalent to code execution. In any case, the
recommendation is not to serve models using Eager mode since it also has
performance limitations.

## Multi-Tenant environments

It is possible to run multiple TensorFlow models in parallel. For example,
`ModelServer` collates all computation graphs exposed to it (from multiple
`SavedModel`) and executes them in parallel on available executors. Running
TensorFlow in a multitenant design mixes the risks described above with the
inherent ones from multitenant configurations. The primary areas of concern are
tenant isolation, resource allocation, model sharing and hardware attacks.

### Tenant isolation

Since any tenants or users providing models, graphs or checkpoints can execute
code in context of the TensorFlow service, it is important to design isolation
mechanisms that prevent unwanted access to the data from other tenants.

Network isolation between different models is also important not only to prevent
unauthorized access to data or models, but also to prevent malicious users or
tenants sending graphs to execute under another tenant’s identity.

The isolation mechanisms are the responsibility of the users to design and
implement, and therefore security issues deriving from their absence are not
considered a vulnerability in TensorFlow.

### Resource allocation

A denial of service caused by one model could bring down the entire server, but
we don't consider this as a vulnerability, given that models can exhaust
resources in many different ways and solutions exist to prevent this from
happening (e.g., rate limits, ACLs, monitors to restart broken servers).

### Model sharing

If the multitenant design allows sharing models, make sure that tenants and
users are aware of the security risks detailed here and that they are going to
be practically running code provided by other users. Currently there are no good
ways to detect malicious models/graphs/checkpoints, so the recommended way to
mitigate the risk in this scenario is to sandbox the model execution.

### Hardware attacks

Physical GPUs or TPUs can also be the target of attacks. [Published
research](https://scholar.google.com/scholar?q=gpu+side+channel) shows that it
might be possible to use side channel attacks on the GPU to leak data from other
running models or processes in the same system. GPUs can also have
implementation bugs that might allow attackers to leave malicious code running
and leak or tamper with applications from other users. Please report
vulnerabilities to the vendor of the affected hardware accelerator.

## Reporting vulnerabilities

### Vulnerabilities in TensorFlow

This document covers different use cases for TensorFlow together with comments
whether these uses were recommended or considered safe, or where we recommend
some form of isolation when dealing with untrusted data. As a result, this
document also outlines what issues we consider as TensorFlow security
vulnerabilities.

We recognize issues as vulnerabilities only when they occur in scenarios that we
outline as safe; issues that have a security impact only when TensorFlow is used
in a discouraged way (e.g. running untrusted models or checkpoints, data parsing
outside of the safe formats, etc.) are not treated as vulnerabilities..

### Reporting process

Please use [Google Bug Hunters reporting form](https://g.co/vulnz) to report
security vulnerabilities. Please include the following information along with
your report:

  - A descriptive title
  - Your name and affiliation (if any).
  - A description of the technical details of the vulnerabilities.
  - A minimal example of the vulnerability. It is very important to let us know
    how we can reproduce your findings. For memory corruption triggerable in
    TensorFlow models, please demonstrate an exploit against one of Alphabet's
    models in <https://tfhub.dev/>
  - An explanation of who can exploit this vulnerability, and what they gain
    when doing so. Write an attack scenario that demonstrates how your issue
    violates the use cases and security assumptions defined in the threat model.
    This will help us evaluate your report quickly, especially if the issue is
    complex.
  - Whether this vulnerability is public or known to third parties. If it is,
    please provide details.

We will try to fix the problems as soon as possible. Vulnerabilities will, in
general, be batched to be fixed at the same time as a quarterly release. We
credit reporters for identifying security issues, although we keep your name
confidential if you request it. Please see Google Bug Hunters program website
for more info.
