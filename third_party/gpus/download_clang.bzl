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
  CLANG_REVISION = '318667'
  CLANG_SUB_REVISION = 1

  package_version = '%s-%s' % (CLANG_REVISION, CLANG_SUB_REVISION)

  checksums = {
      'Linux_x64':
          'e63e5fe3ec8eee4779812cd16aae0ddaf1256d2e8e93cdd5914a3d3e01355dc1',
      'Mac':
          '4f0aca6ec66281be94c3045550ae15a73befa59c32396112abda0030ef22e9b6',
      'Win':
          '7e848f2a586ea01effc51f5776dee6ffbeae0865eec6ca8a73ac9565ed299f8e',
  }

  platform_folder = _get_platform_folder(repo_ctx.os.name)
  _download_chromium_clang(repo_ctx, platform_folder, package_version,
                           checksums[platform_folder], out_folder)
