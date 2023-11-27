import re
import os
import sys


def parse_args(argv):
  result = {}
  for arg in argv:
    k, v = arg.split("=")
    result[k] = v

  return result


def parse_version(cmake):
  pattern = re.compile('set\\(PROJECT_VERSION "([0-9]+\\.[0-9]+\\.[0-9]+)"\\)')
  with open(os.path.expanduser(cmake)) as f:
    for line in f.readlines():
      result = pattern.match(line)
      if result is not None:
        return result.group(1)

  sys.exit("Can't get the right version from ", cmake)


def get_root(header_in):
  """
    This is an assumption that the root workspace should be the same depth
    with "include" folder. It will find start from right position, so should
    handle the include/**/tensorflow/***/onednn/include/**.
    """
  pos = header_in.rindex("include")
  root = header_in[:pos]
  return root


# TODO(intel-tf): Add oneDNN commit hash if need in future.
def git_hash(header_in):
  return "N/A"


def get_cmake(header_in):
  root = get_root(header_in)
  cmake = os.path.join(root, "CMakeLists.txt")
  return cmake


def generate_version(version, header_in, header_out):
  hash_value = git_hash(header_in)

  [major, minor, patch] = version.split(".")

  with open(os.path.expanduser(header_in)) as inf:
    content = inf.read()
    content = content.replace("@DNNL_VERSION_MAJOR@", major)
    content = content.replace("@DNNL_VERSION_MINOR@", minor)
    content = content.replace("@DNNL_VERSION_PATCH@", patch)
    content = content.replace("@DNNL_VERSION_HASH@", hash_value)

    header_out = os.path.expanduser(header_out)
    header_out_dir = os.path.dirname(header_out)
    if not os.path.exists(header_out_dir):
      os.makedirs(header_out_dir, exist_ok=True)

    with open(header_out, "w") as outf:
      outf.write(content)


def main():
  args = parse_args(sys.argv[1:])
  cmake = get_cmake(args["--in"])
  version = parse_version(cmake)
  generate_version(version, args["--in"], args["--out"])


if __name__ == "__main__":
  main()
