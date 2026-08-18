"""Exposes the ROCm CI image's system pciutils (libpci) headers to Bazel.

pciutils ships in the ROCm build image (rocm/tensorflow-build) but not in the
hermetic sysroot, so under --config=rocm_ci the compiler cannot find
<pci/pci.h> (used by MORI's src/application/topology/pci.cpp). Rather than
vendor pciutils -- whose public headers include a generated pci/config.h that
is awkward to reproduce -- we reference the image's already-installed copy.
This keeps the headers as declared Bazel inputs (so the absolute-include check
passes) while still "using the one from the container".

The header directory is NOT the same across images: Debian/Ubuntu multiarch
puts them under /usr/include/x86_64-linux-gnu/pci, while others use the plain
/usr/include/pci. new_local_repository can't express "whichever exists", so we
use a custom repository rule that probes candidates at fetch time and symlinks
the headers into a pci/ subdir (exposed as <pci/pci.h> via includes = ["."]).

Override the search explicitly with the SYSTEM_LIBPCI_PATH env var / repo_env
if your image keeps the headers somewhere else.

libpci.so itself is linked from the image via @roc_mori//:libpci (-lpci),
resolved locally (rocm_clang_hermetic links with CppLink=local). Only
referenced by @roc_mori//:libpci, which is only built for ROCm targets.
"""

_BUILD_FILE = """\
load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # GPL-2.0-or-later (pciutils; headers only, linked dynamically)

# Headers are symlinked into ./pci by the repository rule; includes = ["."]
# makes them resolvable as <pci/pci.h>, matching how the sources include them.
cc_library(
    name = "pci_headers",
    hdrs = glob(["pci/*.h"]),
    includes = ["."],
    visibility = ["//visibility:public"],
)
"""

# Searched in order; first existing directory wins. SYSTEM_LIBPCI_PATH (if set)
# is tried first so a container with an unusual layout can force the right dir.
_DEFAULT_CANDIDATES = [
    "/usr/include/x86_64-linux-gnu/pci",  # Debian/Ubuntu multiarch (libpci-dev)
    "/usr/include/pci",  # classic / non-multiarch layout
]

def _system_libpci_impl(repository_ctx):
    candidates = []
    override = repository_ctx.os.environ.get("SYSTEM_LIBPCI_PATH", "").strip()
    if override:
        candidates.append(override)
    candidates.extend(_DEFAULT_CANDIDATES)

    chosen = None
    for c in candidates:
        if repository_ctx.path(c).exists:
            chosen = c
            break

    if chosen == None:
        fail(
            "system_libpci: could not find pciutils headers (<pci/pci.h>). " +
            "Tried: {}. Install libpci-dev, or set SYSTEM_LIBPCI_PATH to the ".format(candidates) +
            "directory that contains pci.h (e.g. via --repo_env=SYSTEM_LIBPCI_PATH=...).",
        )

    # Symlink each header into a pci/ subdir so glob() sees real files.
    res = repository_ctx.execute(["find", chosen, "-maxdepth", "1", "-name", "*.h"])
    if res.return_code != 0:
        fail("system_libpci: failed to list headers in {}: {}".format(chosen, res.stderr))

    found = False
    for line in res.stdout.splitlines():
        path = line.strip()
        if not path:
            continue
        name = path.rsplit("/", 1)[-1]
        repository_ctx.symlink(path, "pci/" + name)
        found = True

    if not found:
        fail("system_libpci: no *.h headers found in {}".format(chosen))

    repository_ctx.file("BUILD.bazel", _BUILD_FILE)

system_libpci_repository = repository_rule(
    implementation = _system_libpci_impl,
    # Re-fetch if the override changes so switching containers is picked up.
    environ = ["SYSTEM_LIBPCI_PATH"],
    local = True,
    configure = True,
)

def repo():
    """Registers @system_libpci, auto-detecting the image's pci header dir."""
    system_libpci_repository(name = "system_libpci")
