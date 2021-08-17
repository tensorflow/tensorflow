This directory contains shim header files that forward to the TF Lite
C API and to the key headers of the TF Lite C++ API.

The intent is that the shims in this directory could be modified to optionally
redirect to a different implementation of those APIs (for example,
one built into the underlying operating system platform).

These should be used as follows: #includes from .cc files that are
_implementing_ the shimmed TF Lite APIs should include the regular TF
Lite API headers.  #includes from files that are _using_ the shimmed
APIs should include the shimmed headers.
