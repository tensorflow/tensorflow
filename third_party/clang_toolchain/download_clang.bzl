""" Helpers to download a recent clang release."""

def _get_platform_folder(os_name):
    os_name = os_name.lower()
    if os_name.startswith("windows"):
        return "Win"
    if os_name.startswith("mac os"):
        return "Mac"
    if not os_name.startswith("linux"):
        fail("Unknown platform")
    return "Linux_x64"

def _download_chromium_clang(
        repo_ctx,
        platform_folder,
        package_version,
        sha256,
        out_folder):
    cds_url = "https://commondatastorage.googleapis.com/chromium-browser-clang"
    cds_file = "clang-%s.tgz" % package_version
    cds_full_url = "{0}/{1}/{2}".format(cds_url, platform_folder, cds_file)
    repo_ctx.download_and_extract(cds_full_url, output = out_folder, sha256 = sha256)

def download_clang(repo_ctx, out_folder):
    """ Download a fresh clang release and put it into out_folder.

    Clang itself will be located in 'out_folder/bin/clang'.
    We currently download one of the latest releases of clang by the
    Chromium project (see
    https://chromium.googlesource.com/chromium/src/+/master/docs/clang.md).

    Args:
      repo_ctx: An instance of repository_context object.
      out_folder: A folder to extract the compiler into.
    """
    # TODO(ibiryukov): we currently download and extract some extra tools in the
    # clang release (e.g., sanitizers). We should probably remove the ones
    # we don't need and document the ones we want provide in addition to clang.

    # Latest CLANG_REVISION and CLANG_SUB_REVISION of the Chromiums's release
    # can be found in https://chromium.googlesource.com/chromium/src/tools/clang/+/master/scripts/update.py
    CLANG_REVISION = "340427"
    CLANG_SUB_REVISION = 1

    package_version = "%s-%s" % (CLANG_REVISION, CLANG_SUB_REVISION)

    checksums = {
        "Linux_x64": "8a8f21fb624fc7be7e91e439a13114847185375bb932db51ba590174ecaf764b",
        "Mac": "ba894536b7c8d37103a5ddba784f268d55e65bb2ea1200a2cf9f2ef1590eaacd",
        "Win": "c3f5bd977266dfd011411c94a13e00974b643b70fb0225a5fb030f7f703fa474",
    }

    platform_folder = _get_platform_folder(repo_ctx.os.name)
    _download_chromium_clang(
        repo_ctx,
        platform_folder,
        package_version,
        checksums[platform_folder],
        out_folder,
    )
