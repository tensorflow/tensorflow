"""Safe archive extraction helpers.

Prevents path traversal vulnerabilities when extracting tar or zip archives
by validating each member's final destination is within the target directory.
"""
import os
import tarfile
import zipfile


def _is_within_directory(directory, target):
  abs_directory = os.path.abspath(directory)
  abs_target = os.path.abspath(target)
  try:
    common = os.path.commonpath([abs_directory, abs_target])
  except Exception:
    return False
  return common == abs_directory


def safe_extract_tar(tar: tarfile.TarFile, path: str = '.'):  # pylint: disable=missing-function-docstring
  for member in tar.getmembers():
    member_path = os.path.join(path, member.name)
    if not _is_within_directory(path, member_path):
      raise Exception('Attempted Path Traversal in Tar File')
  tar.extractall(path)


def safe_extract_zip(zf: zipfile.ZipFile, path: str = '.'):  # pylint: disable=missing-function-docstring
  for member in zf.namelist():
    member_path = os.path.join(path, member)
    if not _is_within_directory(path, member_path):
      raise Exception('Attempted Path Traversal in Zip File')
  zf.extractall(path)
