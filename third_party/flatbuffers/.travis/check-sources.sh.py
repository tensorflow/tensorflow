import os
import re
import sys

def check_encoding(encoding, scan_dir, regex_pattern):
  fname = None
  try:
    assert encoding in ['ascii', 'utf-8'], "unexpected encoding"
    cmp = re.compile(regex_pattern)
    for root, dirs, files in os.walk(scan_dir):
      fname = root
      cmp_list = [f for f in files if cmp.search(f) is not None]
      for f in cmp_list:
        fname = os.path.join(root, f)
        with open(fname, mode='rb') as test_file:
          btext = test_file.read()
        # check encoding
        btext.decode(encoding=encoding, errors="strict")
        if encoding == "utf-8" and btext.startswith(b'\xEF\xBB\xBF'):
          raise ValueError("unexpected BOM in file")
        # check LF line endings
        LF = btext.count(b'\n')
        CR = btext.count(b'\r')
        if CR!=0:
          raise ValueError("invalid line endings: LF({})/CR({})".format(LF, CR))
  except Exception as err:
    print("ERROR with [{}]: {}".format(fname, err))
    return -1
  else:
    return 0

if __name__ == "__main__":
  # python check-sources.sh.py 'ascii' '.' '.*\.(cpp|h)$'
  res = check_encoding(sys.argv[1], sys.argv[2], sys.argv[3])
  sys.exit(0 if res == 0 else -1)
