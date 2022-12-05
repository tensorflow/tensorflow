# Using TensorFlow Securely

This document discusses the TensorFlow security model. It describes how to
safely deal with untrusted programs (models or model parameters), and input
data. We also provide guidelines on what constitutes a vulnerability in
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
The model's parameters are often stored separately in **checkpoints**.

At runtime, TensorFlow executes the computation graph using the parameters
provided. Note that the behavior of the computation graph may change depending
on the parameters provided. **TensorFlow itself is not a sandbox**. When
executing the computation graph, TensorFlow may read and write files, send and
receive data over the network, and even spawn additional processes. All these
tasks are performed with the permission of the TensorFlow process. Allowing for
this flexibility makes for a powerful machine learning platform, but it has
security implications.

The computation graph may also accept **inputs**. Those inputs are the
data you supply to TensorFlow to train a model, or to use a model to run
inference on the data.

**TensorFlow models are programs, and need to be treated as such from a security
perspective.**

## Execution models of TensorFlow code

The TensorFlow library has a wide API which can be used in multiple scenarios.
The security requirements are also different depending on the usage.

The API usage with the least security concerns is doing iterative exploration
via the Python interpreter or small Python scripts. Here, only some parts of the
API are exercised and eager execution is the default, meaning that each
operation executes immediately. This mode is useful for testing, including
fuzzing. For direct access to the C++ kernels, users of TensorFlow can directly
call `tf.raw_ops.xxx` APIs. This gives control over all the parameters that
would be sent to the kernel. Passing invalid combinations of parameters can
allow insecure behavior (see definition of a vulnerability in a section below).
However, these won’t always translate to actual vulnerabilities in TensorFlow.
This would be similar to directly dereferencing a null pointer in a C++ program:
not a vulnerability by itself but a coding error.

The next 2 modes of using the TensorFlow API have the most security
implications. These relate to the actual building and use of machine learning
models. Both during training and inference, the TensorFlow runtime will build
and execute computation graphs from (usually Python) code written by a
practitioner (using compilation techniques to turn eager code into graph mode).
In both of these scenarios, a vulnerability can be exploited to cause
significant damage, hence the goal of the security team is to eliminate these
vulnerabilities or otherwise reduce their impact. This is essential, given that
both training and inference can run on accelerators (e.g. GPU, TPU) or in a
distributed manner.

Finally, the last mode of executing TensorFlow library code is as part of
additional tooling. For example, TensorFlow provides a `saved_model_cli` tool
which can be used to scan a `SavedModel` (the serialization format used by
TensorFlow for models) and describe it. These tools are usually run by a single
developer, on a single host, so the impact of a vulnerability in them is
somewhat reduced.

## Running untrusted models

As a general rule: **Always** execute untrusted models inside a sandbox (e.g.,
[nsjail](https://github.com/google/nsjail)).

There are several ways in which a model could become untrusted. Obviously, if an
untrusted party supplies TensorFlow kernels, arbitrary code may be executed.
The same is true if the untrusted party provides Python code, such as the Python
code that generates TensorFlow graphs.

Even if the untrusted party only supplies the serialized computation graph (in
form of a `GraphDef`, `SavedModel`, or equivalent on-disk format), the set of
computation primitives available to TensorFlow is powerful enough that you
should assume that the TensorFlow process effectively executes arbitrary code.
One common solution is to allow only a few safe Ops. While this is possible in
theory, we still recommend you sandbox the execution.

It depends on the computation graph whether a user provided checkpoint is safe.
It is easily possible to create computation graphs in which malicious
checkpoints can trigger unsafe behavior. For example, consider a graph that
contains a `tf.cond` operation depending on the value of a `tf.Variable`. One
branch of the `tf.cond` is harmless, but the other is unsafe. Since the
`tf.Variable` is stored in the checkpoint, whoever provides the checkpoint now
has the ability to trigger unsafe behavior, even though the graph is not under
their control.

In other words, graphs can contain vulnerabilities of their own. To allow users
to provide checkpoints to a model you run on their behalf (e.g., in order to
compare model quality for a fixed model architecture), you must carefully audit
your model, and we recommend you run the TensorFlow process in a sandbox.

Similar considerations should apply if the model uses **custom ops** (C++ code
written outside of the TensorFlow tree and loaded as plugins).

## Accepting untrusted inputs

It is possible to write models that are secure in the sense that they can safely
process untrusted inputs assuming there are no bugs. There are, however, two
main reasons to not rely on this: First, it is easy to write models which must
not be exposed to untrusted inputs, and second, there are bugs in any software
system of sufficient complexity. Letting users control inputs could allow them
to trigger bugs either in TensorFlow or in dependencies.

In general, it is good practice to isolate parts of any system which is exposed
to untrusted (e.g., user-provided) inputs in a sandbox.

A useful analogy to how any TensorFlow graph is executed is any interpreted
programming language, such as Python. While it is possible to write secure
Python code which can be exposed to user supplied inputs (by, e.g., carefully
quoting and sanitizing input strings, size-checking input blobs, etc.), it is
very easy to write Python programs which are insecure. Even secure Python code
could be rendered insecure by a bug in the Python interpreter, or in a bug in a
Python library used (e.g.,
[this one](https://www.cvedetails.com/cve/CVE-2017-12852/)).

## Running a TensorFlow server

TensorFlow is a platform for distributed computing, and as such there is a
TensorFlow server (`tf.train.Server`). **The TensorFlow server is meant for
internal communication only. It is not built for use in an untrusted network.**

For performance reasons, the default TensorFlow server does not include any
authorization protocol and sends messages unencrypted. It accepts connections
from anywhere, and executes the graphs it is sent without performing any checks.
Therefore, if you run a `tf.train.Server` in your network, anybody with
access to the network can execute what you should consider arbitrary code with
the privileges of the process running the `tf.train.Server`.

When running distributed TensorFlow, you must isolate the network in which the
cluster lives. Cloud providers provide instructions for setting up isolated
networks, which are sometimes branded as "virtual private cloud." Refer to the
instructions for
[GCP](https://cloud.google.com/compute/docs/networks-and-firewalls) and
[AWS](https://aws.amazon.com/vpc/)) for details.

Note that `tf.train.Server` is different from the server created by
`tensorflow/serving` (the default binary for which is called `ModelServer`).
By default, `ModelServer` also has no built-in mechanism for authentication.
Connecting it to an untrusted network allows anyone on this network to run the
graphs known to the `ModelServer`. This means that an attacker may run
graphs using untrusted inputs as described above, but they would not be able to
execute arbitrary graphs. It is possible to safely expose a `ModelServer`
directly to an untrusted network, **but only if the graphs it is configured to
use have been carefully audited to be safe**.

Similar to best practices for other servers, we recommend running any
`ModelServer` with appropriate privileges (i.e., using a separate user with
reduced permissions). In the spirit of defense in depth, we recommend
authenticating requests to any TensorFlow server connected to an untrusted
network, as well as sandboxing the server to minimize the adverse effects of
any breach.

## Multitenancy environments

It is possible to run multiple TensorFlow models in parallel. For example,
`ModelServer` collates all computation graphs exposed to it (from multiple
`SavedModel`) and executes them in parallel on available executors. A denial of
service caused by one model could bring down the entire server, but we don't
consider this as a high impact vulnerability, given that there exists solutions
to prevent this from happening (e.g., rate limits, ACLs, monitors to restart
broken servers).

However, it is a critical vulnerability if a model could be manipulated such
that it would output parameters of another model (or itself!) or data that
belongs to another model.

Models that also run on accelerators could be abused to do hardware damage or to
leak data that exists on the accelerators from previous executions, if not
cleared.

## Vulnerabilities in TensorFlow

TensorFlow is a large and complex system. It also depends on a large set of
third party libraries (e.g., `numpy`, `libjpeg-turbo`, PNG parsers, `protobuf`).
It is possible that TensorFlow or its dependencies may contain vulnerabilities
that would allow triggering unexpected or dangerous behavior with specially
crafted inputs.

Given TensorFlow's flexibility, it is possible to specify computation graphs
which exhibit unexpected or unwanted behavior. The fact that TensorFlow models
can perform arbitrary computations means that they may read and write files,
communicate via the network, produce deadlocks and infinite loops, or run out of
memory. It is only when these behaviors are outside the specifications of the
operations involved that such behavior is a vulnerability.

A `FileWriter` writing a file is not unexpected behavior and therefore is not a
vulnerability in TensorFlow. A `MatMul` allowing arbitrary binary code execution
**is** a vulnerability.

This is more subtle from a system perspective. For example, it is easy to cause
a TensorFlow process to try to allocate more memory than available by specifying
a computation graph containing an ill-considered `tf.tile` operation. TensorFlow
should exit cleanly in this case (it would raise an exception in Python, or
return an error `Status` in C++). However, if the surrounding system is not
expecting the possibility, such behavior could be used in a denial of service
attack (or worse). Because TensorFlow behaves correctly, this is not a
vulnerability in TensorFlow (although it would be a vulnerability of this
hypothetical system).

As a general rule, it is incorrect behavior for TensorFlow to access memory it
does not own, or to terminate in an unclean way. Bugs in TensorFlow that lead to
such behaviors constitute a vulnerability.

One of the most critical parts of any system is input handling. If malicious
input can trigger side effects or incorrect behavior, this is a bug, and likely
a vulnerability.

**Note**: Assertion failures used to be considered a vulnerability in
TensorFlow. If an assertion failure  only leads to program termination and no
other exploits, we will no longer consider assertion failures (e.g.,
`CHECK`-fails) as vulnerabilities. However, if the assertion failure occurs only
in debug mode (e.g., `DCHECK`) and in production-optimized mode the issue turns
into other code weakness(e.g., heap overflow, etc.), then we will consider
this to be a vulnerability. We recommend reporters to try to maximize the impact
of the vulnerability report (see also [the Google VRP
rules](https://bughunters.google.com/about/rules/6625378258649088/google-and-alphabet-vulnerability-reward-program-vrp-rules)
and [the Google OSS VRP
rules](https://bughunters.google.com/about/rules/6521337925468160/google-open-source-software-vulnerability-reward-program-rules)).

**Note**: Although the iterative exploration of TF API via fuzzing
`tf.raw_ops.xxx` symbols is the best way to uncover code weakeness, please bear
in mind that this is not a typical usecase that has security implications. It is
better to try to translate the vulnerability to something that can be exploited
during training or inference of a model (i.e., build a model that when given a
specific input would produce unwanted behavior). Alternatively, if the
TensorFlow API is only used in ancillary tooling, consider the environment where
the tool would run. For example, if `saved_model_cli` tool would crash on
parsing a `SavedModel` that is not considered a vulnerability but a bug (since
the user can use other ways to inspect the model if needed). However, it would
be a vulnerability if passing a `SavedModel` to `saved_model_cli` would result
in opening a new network connection, corrupting CPU state, or other forms of
unwanted behavior.

## Reporting vulnerabilities

Please fill out [this report form](https://forms.gle/mr12SgzXENhxQ7jD6) about
any security related issues you find.

Please use a descriptive title for your report.

In addition, please include the following information along with your report:

*   Your name and affiliation (if any).
*   A description of the technical details of the vulnerabilities. It is very
    important to let us know how we can reproduce your findings.
*   A minimal example of the vulnerabity.
*   An explanation of who can exploit this vulnerability, and what they gain
    when doing so -- write an attack scenario. This will help us evaluate your
    report quickly, especially if the issue is complex.
*   Whether this vulnerability is public or known to third parties. If it is,
    please provide details.

After the initial reply to your report, the security team will endeavor to keep
you informed of the progress being made towards a fix and announcement.
TensorFlow uses the following disclosure process:

* When a report is received, we confirm the issue and determine its severity.
  **Please try to maximize impact in the report**, going beyond just obtaining
  unwanted behavior in a fuzzer.
* If we know of specific third-party services or software based on TensorFlow
  that require mitigation before publication, those projects will be notified.
* An advisory is prepared (but not published) which details the problem and
  steps for mitigation.
* The vulnerability is fixed and potential workarounds are identified.
* We will publish a security advisory for all fixed vulnerabilities.

For each vulnerability, we try to ingress it as soon as possible, given the size
of the team and the number of reports. Vulnerabilities will, in general, be
batched to be fixed at the same time as a quarterly release. An exception to
this rule is for high impact vulnerabilities where exploitation of models used
for inference in products (i.e., not models created just to showcase a
vulnerability) is possible. In these cases, we will attempt to do patch releases
within an accelerated timeline, not waiting for the next quarterly release.

Past security advisories are listed
[here](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/security/README.md).
In the future, we might sunset this list and only use GitHub's Security Advisory
format, to simplify the post-vulnerability-fix process.  We credit reporters for
identifying security issues, although we keep your name confidential if you
request it.

**Note**: Since September 2022, you may also use [the Google OSS VRP
program](https://bughunters.google.com/about/rules/6521337925468160/google-open-source-software-vulnerability-reward-program-rules)
to submit vulnerability reports. All consideration in this section still apply.
