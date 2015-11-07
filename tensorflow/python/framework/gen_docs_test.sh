#!/bin/bash -eux
DIR=$TEST_SRCDIR/tensorflow/python
$DIR/gen_docs_combined --out_dir $TEST_TMPDIR
echo "PASS"
