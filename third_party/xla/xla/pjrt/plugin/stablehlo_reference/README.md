# StableHLO Reference Interpreter PJRT Plugin

This directory implements a basic PJRT C++ plugin which accepts StableHLO and
evaluates it with the StableHLO Reference Interpreter.

This plugin has minimal XLA dependencies, and can run the majority of the LAX
Metal Testsuite, with the remaining errors being identified as interpreter bugs
or areas where the interpreter doesn't guarantee numeric precision and tests
using too strict of tolerances.

`=== 16 failed, 1676 passed, 58 skipped in 1035.19s (0:17:15) ===`

## Build a plugin using the Reference Plugin as a scaffolding

Modify the `Compile` and `Execute` functions in `executable.cc` to bridge from
StableHLO to a separate compiler IR / executable.