""" Helpers to download a recent clang release."""

def _get_platform_folder(os_name):
  os_name = os_name.lower()
  if os_name.startswith('windows'):
    return 'Win'
  if os_name.startswith('mac os'):
    return 'Mac'
  if not os_name.startswith('linux'):
    fail('Unknown platform')
  return 'Linux_x64'

def _download_chromium_clang(repo_ctx, platform_folder, package_version, sha256,
                             out_folder):
  cds_url = 'https://commondatastorage.googleapis.com/chromium-browser-clang'
  cds_file = 'clang-%s.tgz' % package_version
  cds_full_url = '{0}/{1}/{2}'.format(cds_url, platform_folder, cds_file)
  repo_ctx.download_and_extract(cds_full_url, output=out_folder, sha256=sha256)

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
  CLANG_REVISION = '336424'
  CLANG_SUB_REVISION = 1

  package_version = '%s-%s' % (CLANG_REVISION, CLANG_SUB_REVISION)

  checksums = {
      'Linux_x64':
          '2ea97e047470da648f5d078af008bce6891287592382cee3d53a1187d996da94',
      'Mac':
          'c6e28909cce63ee35e0d51284d9f0f6e8838f7fb8b7a0dc9536c2ea900552df0',
      'Win':
          '1299fda7c4378bfb81337f7e5f351c8a1f953f51e0744e2170454b8d722f3db7',
  }

  platform_folder = _get_platform_folder(repo_ctx.os.name)
  _download_chromium_clang(repo_ctx, platform_folder, package_version,
                           checksums[platform_folder], out_folder)
