#!/usr/bin/env python3
#
# Copyright 2026 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Script to generate GPU kernel list for oneDNN."""

import os
import re
import sys

_INCLUDE_PATTERN = re.compile(r'^\s*#include "(.*)"')

IS_V2 = False


class Kernel(object):
  """Represents a GPU kernel and its associated metadata.

  This class processes OpenCL kernel files and converts
  them into C++ source code with embedded string literals.
  """

  def __init__(self, path):
    """Initialize a Kernel object from an OpenCL kernel file.

    Args:
      path: Path to the OpenCL kernel file (.cl).
    """
    self.extern_ = self._parse_extern(path)
    self.kernels_ = self._parse_kernels(path)
    self.path_ = path

  def extern(self):
    """Generates the C++ extern declaration for the kernel.

    Returns:
      A string containing the extern declaration
      (e.g., 'extern const char *kernel_name_kernel;').
    """
    extern_prefix = "extern const char *"
    extern_suffix = "_kernel;"
    if not IS_V2:
      extern_suffix = "_kernel[];"
    return extern_prefix + self.extern_ + extern_suffix

  def entries(self):
    """Generate kernel list entries for the C++ kernel registry.

    Returns:
      A list of formatted strings, each representing a kernel entry mapping
      kernel name to its extern symbol.
    """
    format_extern = lambda x: '        {{ "{}", {}_kernel }},'.format(
        x, self.extern_
    )
    return [format_extern(entry) for entry in self.kernels_]

  def content(self, inc_dirs):
    """Generate the kernel content with expanded includes.

    Args:
      inc_dirs: List of include directories to search for header files.

    Returns:
      A tuple of (kernel_name, expanded_content) where expanded_content is
      the kernel source with all #include directives recursively expanded.
    """
    path = os.path.basename(self.path_)
    kernel_name, _ = os.path.splitext(path)
    return kernel_name, "\n".join(self._extend_includes(inc_dirs, self.path_))

  def subfolder(self, sub):
    """Extract the subfolder path relative to a base directory.

    Args:
      sub: The base directory name to search for in the kernel's path.

    Returns:
      The relative path after the base directory, or empty string if
      the kernel is directly in the base directory.
    """
    kernel_dir = os.path.dirname(self.path_)
    index = kernel_dir.rfind(sub)
    index += len(sub) + 1  # step to the last if possible
    if index == len(kernel_dir):
      return ""
    else:
      return kernel_dir[index:]

  def _parse_extern(self, path):
    """Parse the extern symbol name from the kernel file path.

    Args:
      path: Path to the kernel file.

    Returns:
      The extern symbol name. V2 format includes parent directory name.
    """
    dir_path = os.path.dirname(path)
    dir1 = os.path.basename(dir_path)
    path = os.path.basename(path)
    file_name, _ = os.path.splitext(path)
    if IS_V2:
      file_name = dir1 + "_" + file_name
    return file_name

  def _parse_kernels(self, path):
    """Extract kernel function names from the OpenCL source file.

    Args:
      path: Path to the kernel file.

    Returns:
      A list of kernel function names found in the file.
    """
    with open(path) as f:
      content = f.read()
      pattern = "kernel[ \n]+void[ \n]+([a-z0-9_]+)"
      return re.findall(pattern, content, re.DOTALL)

  def _extend_includes(self, inc_dirs, path):
    """Recursively expand #include directives and format lines as C++ string literals.

    Args:
      inc_dirs: List of include directories to search for header files.
      path: Path to the file to process.

    Returns:
      A list of strings, each representing a line of the source file formatted
      as a C++ string literal. #include directives are replaced with the
      contents
      of the included files (recursively expanded).
    """
    ret = []
    with open(path) as f:
      lines = f.readlines()
      for line in lines:
        result = _INCLUDE_PATTERN.match(line)
        if result is not None:
          inc_file = result.group(1)
          for inc_dir in inc_dirs:
            inc_path = os.path.join(inc_dir, inc_file)
            if not os.path.exists(inc_path):
              continue
            inc_lines = self._extend_includes(inc_dirs, inc_path)
            ret.extend(inc_lines)
            break
        else:
          if IS_V2:
            line = line.strip()
            line = line.replace("\n", "")
            quoted_line = 'R"==({})==""\\n"'.format(line)
          else:
            line = line.replace("\\", "\\\\")
            line = line.replace('"', '\\"')
            line = line.replace("\n", "\\n")
            quoted_line = '"{}",'.format(line)
          ret.append(quoted_line)

    return ret


class Header(Kernel):
  """Represents a GPU header and its associated metadata.

  This class extends Kernel to handle header files, which
  are processed similarly but have different naming conventions
  and content formatting requirements.
  """

  def extern(self):
    """Generate the C++ extern declaration for the header.

    Returns:
      A string containing the extern declaration
      (e.g., 'extern const char *header_name_header;').
    """
    extern_prefix = "extern const char *"
    extern_suffix = "_header;"
    return extern_prefix + self.extern_ + extern_suffix

  def values(self):
    """Generate header value entries for the C++ header array.

    Returns:
      A list containing a single formatted string referencing the header extern
      symbol.
    """
    return ["        {}_header,".format(self.extern_)]

  def entries(self):
    """Generate header list entries for the C++ header registry.

    Returns:
      A formatted string mapping the header's relative path (from src/) to its
      extern symbol.
    """
    dir_path = os.path.dirname(self.path_)
    src_index = dir_path.rfind("src")
    header_path = self.path_[src_index + len("src") + 1 :]
    return '        {{ "{}", {}_header }},'.format(header_path, self.extern_)

  def name(self):
    """Generate the header name entry for the C++ name list.

    Returns:
      A formatted string containing the header's relative path (from src/).
    """
    dir_path = os.path.dirname(self.path_)
    src_index = dir_path.rfind("src")
    header_path = self.path_[src_index + len("src") + 1 :]
    return '        "{}",'.format(header_path)


class KernelList(object):
  """Represents a collection of GPU kernels and headers.

  This class manages the discovery and processing of OpenCL kernel
  and header files, and provides methods to generate C++ code with
  embedded source as string literals.
  """

  def __init__(self, folder, header_dir):
    """Initialize a KernelList by discovering and loading kernels and headers.

    Args:
      folder: Directory to search for OpenCL kernel files (.cl).
      header_dir: Directory to search for header files (.h), used only in v2
        format.
    """
    self.kernels_ = []
    self.headers_ = []

    cl_files = self._get_cl_suffix_files(folder)
    for clf in cl_files:
      self.kernels_.append(Kernel(clf))

    if IS_V2:
      header_files = self._get_header_files(header_dir)
      for hf in header_files:
        self.headers_.append(Header(hf))

  def generate_list(self, src, target):
    """Generate the main kernel list C++ file from a template.

    Reads a template file and replaces placeholders with generated extern
    declarations,
    kernel entries, and (in v2 format) header information. Writes the result to
    target.

    Args:
      src: Path to the template file (ocl_kernel_list.cpp.in).
      target: Path where the generated C++ file should be written.
    """
    externs = []
    entries = ["\n"]  # for style
    for kernel in self.kernels_:
      externs.append(kernel.extern())
      entries.extend(kernel.entries())

    externs_content = "\n".join(externs)
    entries_content = "\n".join(entries)

    if IS_V2:
      header_externs = []
      header_values = ["\n"]
      header_names = ["\n"]
      header_entries = ["\n"]
      for header in self.headers_:
        header_externs.append(header.extern())
        header_values.extend(header.values())
        header_names.append(header.name())
        header_entries.append(header.entries())

      header_externs_content = "\n".join(header_externs)
      header_values_content = "\n".join(header_values)
      header_names_content = "\n".join(header_names)
      header_entries_content = "\n".join(header_entries)

    with open(src) as f:
      content = f.read()
      content = content.replace("@KER_LIST_EXTERN@", externs_content)
      content = content.replace("@KER_LIST_ENTRIES@", entries_content)
      if IS_V2:
        content = content.replace(
            "@KER_HEADERS_EXTERN@", header_externs_content
        )
        content = content.replace("@KER_HEADERS@", header_values_content)
        content = content.replace("@KER_HEADER_NAMES@", header_names_content)
        content = content.replace(
            "@KER_HEADER_LIST_ENTRIES@", header_entries_content
        )

    with open(target, "w") as f:
      f.write(content)

  def _generate(self, inc_dirs, root, sub, impl, suffix):
    """Generate a single C++ file from a kernel or header source.

    Creates output directories as needed and writes formatted C++ content
    with embedded source code as string literals.

    Args:
      inc_dirs: List of include directories for resolving #include directives.
      root: Root output directory.
      sub: Subdirectory path relative to root.
      impl: Kernel or Header object to generate from.
      suffix: Suffix for the output file ("kernel" or "header").
    """
    impl_name, content = impl.content(inc_dirs)
    more_sub = impl.subfolder(sub)
    # Constructs the file xxx_suffix.cpp
    file_name = impl_name + "_" + suffix + ".cpp"
    target = os.path.join(root, sub, more_sub)

    if not os.path.exists(target):
      os.makedirs(target, exist_ok=True)

    if more_sub:
      prefix = more_sub
    else:
      prefix = re.split(r"\/", sub)[-1]
    with open(os.path.join(target, file_name), "w") as f:
      f.write(self.format_file_content(impl_name, suffix, content, prefix))

  def generate_kernel(self, inc_dirs, root, sub):
    """Generate C++ files from OpenCL kernels, preserving source directory structure.

    Expands kernel code into C++ string literals and writes _kernel.cpp files.
    Unlike CMake builds (which flatten to src/gpu/ocl/), this preserves
    subdirectories
    for Bazel compatibility (e.g., src/gpu/ocl/gemm/xxx_kernel.cpp).

    Args:
      inc_dirs: List of include directories.
      root: Root output directory.
      sub: Subdirectory path relative to root.
    """
    for kernel in self.kernels_:
      self._generate(inc_dirs, root, sub, kernel, "kernel")

  def generate_header(self, inc_dirs, root, sub):
    """Generate C++ files from header sources, preserving directory structure.

    Similar to generate_kernel but processes .h files into _header.cpp files.
    Only runs when v2 format is enabled.

    Args:
      inc_dirs: List of include directories.
      root: Root output directory.
      sub: Subdirectory path relative to root.
    """
    for header in self.headers_:
      self._generate(inc_dirs, root, sub, header, "header")

  def format_file_content(self, name, suffix, content, prefix):
    """Format the kernel/header content into a C++ file with proper namespace wrapping.

    Args:
      name: Base name of the kernel or header file.
      suffix: Type suffix ("kernel" or "header").
      content: The formatted source content as C++ string literals.
      prefix: Namespace prefix (usually the parent directory name).

    Returns:
      Complete C++ source code with content wrapped in dnnl::impl::gpu::intel
      namespace.
    """
    header = """
namespace dnnl::impl::gpu::intel {{
    const char* {}_{}_{} =
{};
}} // namespace dnnl::impl::gpu::intel
        """

    if not IS_V2:
      header = """
namespace dnnl::impl::gpu::intel {{
    const char* {}_kernel[] = {{
{}
        nullptr
    }};
}} // namespace dnnl::impl::gpu::intel
"""
      return header.format(name, content)

    return header.format(prefix, name, suffix, content)

  def _get_suffix_files(self, folder, suffix):
    """Recursively find all files with a specific suffix in a directory.

    Args:
      folder: Root directory to search.
      suffix: File extension to match (e.g., ".cl", ".h").

    Returns:
      Sorted list of absolute paths to matching files.
    """
    files = []
    for root, _, filenames in os.walk(folder):
      for filename in sorted(filenames):
        s = os.path.splitext(filename)[-1]
        if s == suffix:
          f = os.path.join(root, filename)
          files.append(f)
    return files

  def _get_cl_suffix_files(self, folder):
    """Find all OpenCL kernel files (.cl) in a directory.

    Args:
      folder: Root directory to search.

    Returns:
      Sorted list of absolute paths to .cl files.
    """
    return self._get_suffix_files(folder, ".cl")

  def _get_header_files(self, folder):
    """Find all header files (.h) in a directory.

    Args:
      folder: Root directory to search.

    Returns:
      Sorted list of absolute paths to .h files.
    """
    return self._get_suffix_files(folder, ".h")


OCL_IMPL_DIR = "src/gpu/intel"
HEADER_ROOT_DIR = OCL_IMPL_DIR
IN_FILE = "ocl_kernel_list.cpp.in"


class FilesHelper(object):
  """Helper class for managing file paths for kernel generation.

  Attributes:
    inc_dirs: List of include directories.
    ocl_impls_dir: Directory containing OCL implementations.
    gen_kernel_list_cpp_in: Path to input template file.
    out_root: Root output directory.
    out_subfolder: Output subfolder.
    gen_kernel_list_cpp: Path to generated C++ file.
    header_dir: Directory containing headers.
    header_subfolder: Header subfolder.
  """

  def __init__(self, in_file, out_dir):
    """Initialize path manager for kernel generation.

    Computes all input/output paths based on oneDNN's assumed directory
    structure:
    OCL implementations at src/gpu/intel/ocl, headers at src/gpu/intel.

    Args:
      in_file: Input template file path.
      out_dir: Output directory.
    """
    in_file = os.path.expanduser(in_file)
    out_dir = os.path.expanduser(out_dir)

    in_dir = os.path.dirname(in_file)
    in_dir = in_dir[: -len(OCL_IMPL_DIR)]

    self.inc_dirs = [
        os.path.join(in_dir, "src"),
        os.path.join(in_dir, "include"),
    ]
    self.ocl_impls_dir = os.path.join(in_dir, OCL_IMPL_DIR)
    self.gen_kernel_list_cpp_in = os.path.join(in_dir, OCL_IMPL_DIR, IN_FILE)

    self.out_root = out_dir
    self.out_subfolder = OCL_IMPL_DIR

    self.gen_kernel_list_cpp = os.path.join(
        out_dir, OCL_IMPL_DIR, os.path.splitext(IN_FILE)[0]
    )

    kernels_out = os.path.join(self.out_root, self.out_subfolder)
    if not os.path.exists(kernels_out):
      os.makedirs(kernels_out)

    self.header_dir = os.path.join(in_dir, HEADER_ROOT_DIR)
    self.header_subfolder = HEADER_ROOT_DIR


def parse_args(argv: list[str]) -> dict[str, str]:
  result = {}
  for arg in argv:
    k, v = arg.split("=")
    result[k] = v

  return result


def enable_v2(in_file):
  with open(in_file, "r") as f:
    for line in f:
      if "KER_HEADERS" in line:
        global IS_V2
        IS_V2 = True
        break


def main():
  args = parse_args(sys.argv[1:])

  # The --in argument is the ocl_kernel_list.cpp.in file path.
  # The --out argument is the output folder
  files_helper = FilesHelper(args["--in"], args["--out"])

  only_gen_header = False
  if args["--header"].lower() == "true":
    only_gen_header = True

  enable_v2(files_helper.gen_kernel_list_cpp_in)

  kernel_list = KernelList(files_helper.ocl_impls_dir, files_helper.header_dir)
  if not only_gen_header:
    kernel_list.generate_list(
        files_helper.gen_kernel_list_cpp_in, files_helper.gen_kernel_list_cpp
    )
    kernel_list.generate_kernel(
        files_helper.inc_dirs, files_helper.out_root, files_helper.out_subfolder
    )
  if only_gen_header and IS_V2:
    kernel_list.generate_header(
        files_helper.inc_dirs,
        files_helper.out_root,
        files_helper.header_subfolder,
    )


if __name__ == "__main__":
  main()
