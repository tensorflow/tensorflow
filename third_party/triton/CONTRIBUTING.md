# Governance Structure

Triton adopts the following hierarchical technical governance structure:
* A community of **contributors** who file issues and submit pull requests
* A group of **module maintainers** who own parts of Triton and drive their development
* A body of **core maintainers** who own Triton overall and drive its development
* A **lead core maintainer** who is the catch-all decision maker when consensus cannot be reached by core maintainers

All contributions are expected to follow Triton’s design principles, as enforced by module and core maintainers. While high-quality pull requests are appreciated and encouraged, all maintainers reserve the right to prioritize their own work over code reviews at-will, hence contributors should not expect their work to be reviewed promptly.

Contributors can maximize the chances of their work being accepted by maintainers by meeting a high quality bar before sending a PR to maintainers.  We encourage maintainers who contribute to Triton on behalf of a company to get reviews from senior developers within their company before sending to maintainers.
Module maintainers
We aim to make the Triton codebase as modular as possible, such that different components (e.g., subdirectories) can be improved in parallel under the supervision of different module maintainers.

What constitutes (or not) a module is up to the core maintainers. Core maintainers also reserve the right to decide whether the development of a module should happen – or keep happening – in-tree or not.

**List of in-tree modules (as of 05/12/2024, alphabetical order):**
* AMD backend (Lei Zhang)
* Interpreter (Keren Zhou)
* Profiler (Keren Zhou)

Note: Parts of Triton that are not listed above (e.g., Nvidia backend) are assumed to be owned by core maintainers.

Note: Some important parts of the Triton eco-system (e.g., Intel XPU backend) may be maintained out-of-tree and advertised in our repository. The governance rules described in this document do not carry over to these modules.

__List of out-of-tree modules (as of 05/12/2024, alphabetical order):__
* CPU backend (Bert Maher, Ilya Enkovich)
* Intel backend (Ettore Tiotto, Whitney Tsang)


## Core maintainers
The core maintainers drive the development of Triton at large and set the roadmap for the project. As such, they have the following responsibilities:
* Proposing, implementing and reviewing profound changes to user-facing APIs, IR specifications and/or pass infrastructures
* Enforcing code quality standards and adherence to core design principles
* Drawing module boundaries and resolving disputes between module maintainers


The core maintainers as a group have the power to veto any decision made at a Module maintainer level.

The core maintainers should publicly articulate their decision-making, and share the reasoning behind their decisions, vetoes, and dispute resolution.

__List of core maintainers (as of 01/30/2025, alphabetical order):__
* Jeff Niu
* Keren Zhou
* Mario Lezcano-Casado
* Pawel Szczerbuk
* Peter Bell
* Phil Tillet
* Thomas Raoux
* Zahi Moudallal

## Lead core maintainer
When core maintainers cannot come to a consensus, a publicly declared lead maintainer is expected to settle the debate and make executive decisions.

The Lead Core Maintainer should publicly articulate their decision-making, and give a clear reasoning for their decisions.

The Lead Core Maintainer is also responsible for confirming or removing core maintainers.

**Lead maintainer (as of 05/12/2024)**
* Phil Tillet

# Decision Making

## Uncontroversial Changes

We are committed to accepting functional bug fixes that meet our quality standards – and include minimized unit tests to avoid future regressions. Performance improvements generally fall under the same category, with the caveat that they may be rejected if the trade-off between usefulness and complexity is deemed unfavorable by core maintainers (e.g., complex swizzling logic to improve the performance of non-tensor-cores matrix multiplications). Design changes that neither fix known functional nor performance issues are automatically considered controversial.

## Controversial Changes

More controversial design changes (e.g., changes in our IRs/APIs/Passes) are evaluated on a case-by-case basis under the subjective judgment of core maintainers. While it is possible for contributors to propose and land deep design changes upstream (see https://github.com/triton-lang/triton/pull/1305), the community should expect such occurrences to be relatively rare.
