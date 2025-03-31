# Releasing Triton

Triton releases provide a stable snapshot of the code base encapsulated into a binary that can easily be consumed through PyPI. Additionally, releases represent points in time when we, as the development team, can signal to the community that certain new features are available, what improvements have been made, and any changes that are coming that may impact them (i.e. breaking changes).

## Release Compatibility Matrix

Following is the Release Compatibility Matrix for Triton releases:

| Triton version | Python version | Manylinux version |
| --- | --- | --- |
| 3.2.0 | >=3.9, <=3.13 | glibc 2.17+ x86-64 |
| 3.1.0 | >=3.8, <=3.12 | glibc 2.17+ x86-64 |
| 3.0.0 | >=3.8, <=3.12 | glibc 2.17+ x86-64 |
| 2.3.1 | >=3.7, <=3.12 | glibc 2.17+ x86-64 |
| 2.3.0 | >=3.7, <=3.12 | glibc 2.17+ x86-64 |
| 2.2.0 | >=3.7, <=3.12 | glibc 2.17+ x86-64 |
| 2.1.0 | >=3.7, <=3.11 | glibc 2.17+ x86-64 |
| 2.0.0 | >=3.6, <=3.11 | glibc 2.17+ x86-64 |
| 1.1.1 | >=3.6, <=3.9 | glibc 2.17+ x86-64 |
| 1.1.0 | >=3.6, <=3.9 | glibc 2.17+ x86-64 |
| 1.0.0 | >=3.6, <=3.9 | glibc 2.17+ x86-64 |

## Release Cadence

Following is the release cadence for year 2024/2025. All future release dates below are tentative. Please note: Patch Releases are optional.

| Minor Version | Release branch cut | Release date | Patch Release date |
| --- | --- | --- | --- |
| 3.5.0 | Sep 2025 | Oct 2025 | --- |
| 3.4.0 | Jun 2025 | Jul 2025 | --- |
| 3.3.0 | Feb/Mar 2025 | Apr 2025 | --- |
| 3.2.0 | Dec 2024 | Jan 2025 | --- |
| 3.1.0 | Jun 2024 | Oct 2024 | --- |
| 3.0.0 | Jun 2024 | Jul 2024 | --- |
| 2.3.0 | Dec 2023 | Apr 2024 | May 2024 |
| 2.2.0 | Dec 2023 | Jan 2024 | --- |

## Release Cherry-Pick Criteria

After branch cut, we approach finalizing the release branch with clear criteria on what cherry picks are allowed in. Note: a cherry pick is a process to land a PR in the release branch after branch cut. These are typically limited to ensure that the team has sufficient time to complete a thorough round of testing on a stable code base.

* Regression fixes - that address functional/performance regression against the most recent release (e.g. 3.2 for 3.3 release)
* Critical fixes - critical fixes for severe issue such as silent incorrectness, backwards compatibility, crashes, deadlocks, (large) memory leaks
* Fixes to new features introduced in the most recent release (e.g. 3.2 for 3.3 release)
* Documentation improvements
* Release branch specific changes (e.g. change version identifiers or CI fixes)

Please note: **No feature work allowed for cherry picks**. All PRs that are considered for cherry-picks need to be merged on trunk, the only exception are Release branch specific changes. An issue is for tracking cherry-picks to the release branch is created after the branch cut. **Only issues that have ‘cherry-picks’ in the issue tracker will be considered for the release.**
