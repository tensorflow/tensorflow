"""
create-datafiles.py - tool to create a windows def file to export
  symbols from tensorflow.dll to enable tf.load_library().
  Because the linker allows only 64K sumbols to be exported per dll
  we filter the symbols down to the essentials. The regular expressions
  we use for this are specific to tensorflow.

  TODO: this seems to work fine but there is an issue with exporting
  'const char * const' and importing it from a user_ops. The problem is
   on the importing end and using __declspec(dllimport) works around it.
"""
import argparse
import io
import os
import re
import sys
import tempfile
from subprocess import Popen, PIPE

# external tools we use that come with visual studio sdk
# we assume that the caller has the correct PATH to the sdk
UNDNAME = "undname.exe"
DUMPBIN = "dumpbin.exe"

# exclude if matched
EXCLUDE_RE = re.compile(r"deleting destructor|::internal::")

# include if matched before exclude
INCLUDEPRE_RE = re.compile(r"tensorflow::internal::LogMessage|" +
                           r"tensorflow::internal::CheckOpMessageBuilder")

# include if matched after exclude
INCLUDE_RE = re.compile(r"^(TF_\w*)$|" +
                        r"tensorflow::|" +
                        r"functor::|" +
                        r"perftools::gputools")


def get_args():
    """Parse command line."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="input library", required=True)
    parser.add_argument("--output", help="output deffile", required=True)
    args = parser.parse_args()
    return args


def main():
    """main."""
    args = get_args()

    # Pipe dumpbin to extract all linkable symbols from a lib.
    # Good symbols are collected in candidates and also written to
    # a temp file.
    candidates = []
    tmpfile = tempfile.NamedTemporaryFile(mode="w", delete=False)
    proc = Popen([DUMPBIN, "/nologo", "/linkermember:1", args.input], stdout=PIPE)
    for line in io.TextIOWrapper(proc.stdout, encoding="utf-8"):
        cols = line.split()
        if len(cols) < 2:
            continue
        sym = cols[1]
        tmpfile.file.write(sym + "\n")
        candidates.append(sym)
    tmpfile.file.close()
    proc.wait()

    # Run the symbols through undname to get their undecorated name
    # so we can filter on something readable.
    with open(args.output, "w") as def_fp:
        # track dupes
        taken = set()

        # header for the def file. Since the tensorflow.dll is actually called
        # _pywrap_tensorflow.pyd in the python wheel, hint that in the def file.
        def_fp.write("LIBRARY _pywrap_tensorflow_internal.pyd\n")
        def_fp.write("EXPORTS\n")
        def_fp.write("\t ??1OpDef@tensorflow@@UEAA@XZ\n")

        # each symbols returned by undname matches the same position in candidates.
        # We compare on undname but use the decorated name from candidates.
        dupes = 0
        proc = Popen([UNDNAME, tmpfile.name], stdout=PIPE)
        for idx, line in enumerate(io.TextIOWrapper(proc.stdout, encoding="utf-8")):
            decorared = candidates[idx]
            if decorared in taken:
                # symbol is already in output, done.
                dupes += 1
                continue

            if not INCLUDEPRE_RE.search(line):
                if EXCLUDE_RE.search(line):
                    continue
                if not INCLUDE_RE.search(line):
                    continue

            def_fp.write("\t" + decorared + "\n")
            taken.add(decorared)
    proc.wait()
    os.unlink(tmpfile.name)

    print("symbols={}, taken={}, dupes={}".format(len(candidates), len(taken), dupes))
    return 0


if __name__ == "__main__":
    sys.exit(main())
