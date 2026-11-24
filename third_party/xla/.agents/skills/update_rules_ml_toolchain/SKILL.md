---
name: update-rules-ml-toolchain
description: >-
  Updates the pinned reference to google-ml-infra/rules_ml_toolchain (the
  hermetic CUDA/ROCm/SYCL/clang toolchain repo) to a new commit, in both the
  Bzlmod and WORKSPACE dependency definitions, recomputing the integrity
  hashes with the correct tool for each file. Use when asked to update,
  upgrade, or bump rules_ml_toolchain to the latest commit or to a specific
  commit/SHA. Don't use for other third-party dependency bumps (BCR modules,
  other tf_http_archive entries) - only rules_ml_toolchain is in scope.
---

# Update `rules_ml_toolchain` Reference

`rules_ml_toolchain` (https://github.com/google-ml-infra/rules_ml_toolchain)
is not published to the Bazel Central Registry, so XLA pins it by commit SHA
in **two independent places** that must be kept in sync: the Bzlmod module
graph (`MODULE.bazel`) and the legacy WORKSPACE graph (`workspace3.bzl`).
Each place pins the SHA *and* a hash of the source archive, computed with a
different tool/encoding, plus a `urls` field pointing at the same archive -
six values in total (2 files x {urls, strip_prefix, hash}). Getting one
wrong in a single file breaks that build (Bzlmod rejects a bad integrity
hash; `tf_http_archive` requires `sha256`); getting the SHA/hash pair
updated in only one of the two files leaves Bzlmod and WORKSPACE silently
drifted out of sync, each building a different `rules_ml_toolchain` commit.

## Step 0: Determine the Target Commit

- If the user gave an explicit commit SHA, use it verbatim - but it must be
  the full 40-character lowercase hex commit SHA, not an abbreviated SHA or
  a branch/tag name. For an abbreviated SHA, GitHub's `/archive/<sha>.tar.gz`
  endpoint still extracts to `rules_ml_toolchain-<40-char-SHA>/`, so plugging
  an abbreviated SHA into `strip_prefix` breaks Bazel archive extraction. If
  the user gives a branch/tag, resolve it with `git ls-remote
  https://github.com/google-ml-infra/rules_ml_toolchain <ref>`; if they give
  an abbreviated SHA, resolve it via the GitHub API (`curl -fsSL -H
  "Accept: application/vnd.github.sha"
  "https://api.github.com/repos/google-ml-infra/rules_ml_toolchain/commits/<short-sha>"`)
  before proceeding.
- If the user asked for "latest", resolve it yourself - don't guess:
  `git ls-remote https://github.com/google-ml-infra/rules_ml_toolchain HEAD`
  (or check the repo's default branch on GitHub) and confirm the resolved
  SHA with the user before editing if there's any ambiguity about which
  branch/tag "latest" means.

## Step 1: Locate Every Reference and Check Whether an Update Is Needed

```sh
git grep -n "rules_ml_toolchain-" -- "*.bzl" "*.bazel" "WORKSPACE*"
```

Use `git grep`, not a plain recursive `grep -rn ... .` - in checkouts with
prior Bazel builds, an untracked-tree walk (`external/`, `.cache/`,
`bazel-*` symlinks) can turn up spurious hits from already-fetched
repositories. `git grep` only searches tracked files.

As of writing this resolves to exactly two hits, which are the two files
below. If the grep ever returns more (or fewer) than two hits, treat that as
a signal the pinning scheme changed and adjust this procedure accordingly -
don't silently edit only the two files from memory.

- `MODULE.bazel` - the `archive_override(module_name = "rules_ml_toolchain", ...)` block.
- `workspace3.bzl` - the `tf_http_archive(name = "rules_ml_toolchain", ...)` block.

Extract the SHA currently embedded in each `strip_prefix =
"rules_ml_toolchain-<SHA>"` and compare it to the target SHA from Step 0.

- If it differs, continue to Step 2 - a real edit is needed.
- If it already matches (e.g. a prior PR already did this bump), don't
  assume the existing hashes are correct just because the SHA matches: still
  run Steps 2-3 below to download the archive and recompute both hashes from
  scratch, then compare them against what's already in the two files. If
  they match, report that the reference is already up to date and correct
  and skip Step 4. If they don't match, fix the stale hash(es) in Step 4
  even though `strip_prefix` itself doesn't need to change.

## Step 2: Download the Archive Once

```sh
curl -fsSL -o /tmp/rules_ml_toolchain-<SHA>.tar.gz \
  "https://github.com/google-ml-infra/rules_ml_toolchain/archive/<SHA>.tar.gz"
tar -tzf /tmp/rules_ml_toolchain-<SHA>.tar.gz | head -n 1
```

Use an explicit `/tmp/...` path rather than `cd /tmp` - a `cd` does not
persist across separate shell tool calls in most coding agents, so later
steps would silently operate on the wrong directory. Use `-f` (not just
`-sL`): without it, `curl` exits `0` on an HTTP error like 404 and writes
the error page body (e.g. `"Not Found"`) to the output file, which Step 3
would then silently hash as if it were the real archive. The `tar -tzf`
spot check confirms the download is a real tarball (and prints the
`rules_ml_toolchain-<SHA>/` top-level directory, which should match
`<SHA>`) before you trust its hash.

Both hashes below are computed from this one file - download it once and
derive both from it so they are guaranteed to describe the same bytes.

## Step 3: Compute Both Hashes

- **`workspace3.bzl` (`tf_http_archive`, plain hex sha256):**

  ```sh
  openssl dgst -sha256 -r /tmp/rules_ml_toolchain-<SHA>.tar.gz
  ```

  (`-r` prints the plain `<hex> *<filename>` form; use `sha256sum` only if
  `openssl` isn't available - both must agree on the same archive.)

- **`MODULE.bazel` (`archive_override`, Bazel "integrity" attribute - base64,
  `sha256-` prefixed):** This is the SRI format Bazel's `archive_override`
  expects, not a plain hex digest - using `sha256sum`'s hex output here is
  wrong.

  ```sh
  openssl dgst -sha256 -binary /tmp/rules_ml_toolchain-<SHA>.tar.gz | openssl base64 -A
  # then prefix the output with "sha256-"
  ```

  (This matches the `# To calculate integrity:` comment already sitting
  above the `archive_override` block in `MODULE.bazel` - reuse it rather
  than inventing a new recipe.)

## Step 4: Edit Both Files

Skip this step if Step 1 already found the SHA, and both hashes, to be
correct - there's nothing to edit.

In **`MODULE.bazel`**, update the `archive_override` for `rules_ml_toolchain`:

```python
archive_override(
    module_name = "rules_ml_toolchain",
    integrity = "sha256-<base64 from Step 3>",
    strip_prefix = "rules_ml_toolchain-<SHA>",
    urls = ["https://github.com/google-ml-infra/rules_ml_toolchain/archive/<SHA>.tar.gz"],
)
```

In **`workspace3.bzl`**, update the matching `tf_http_archive`:

```python
tf_http_archive(
    name = "rules_ml_toolchain",
    sha256 = "<hex from Step 3>",
    strip_prefix = "rules_ml_toolchain-<SHA>",
    urls = tf_mirror_urls(
        "https://github.com/google-ml-infra/rules_ml_toolchain/archive/<SHA>.tar.gz",
    ),
)
```

Both `strip_prefix` and `urls` values must use the same `<SHA>`; both hashes
must come from the identical archive downloaded in Step 2 (just encoded
differently).

## Step 5: Bazel Spot Check

Before claiming the update is done, confirm Bzlmod actually accepts the new
`archive_override` - a typo'd URL, `strip_prefix`, or integrity hash will
surface here as a fetch/checksum error rather than at `git diff` time:

```sh
bazel mod deps --lockfile_mode=update
```

Run this from the repo root. A clean resolution (no download/checksum
errors for `rules_ml_toolchain`) confirms the new pin is fetchable and
correctly hashed. Do **not** hand-edit `MODULE.bazel.lock` - it's
gitignored and regenerated by Bazel itself; `--lockfile_mode=update` is what
lets this command refresh it instead of failing on a stale one.

## Step 6: Final Diff Check

Re-verify what actually changed:

```sh
git diff -- MODULE.bazel workspace3.bzl
```

If this is empty, Step 1 already determined the repo was pinned to the
requested commit with correct hashes - report that clearly instead of
fabricating a no-op commit. Otherwise this diff should contain exactly the
edits from Step 4.

## Step 7: Clean Up

Delete the scratch archive downloaded in Step 2
(`/tmp/rules_ml_toolchain-<SHA>.tar.gz`) once both hashes have been
extracted - it's not a repo artifact.
