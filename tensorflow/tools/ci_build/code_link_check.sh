#!/usr/bin/env bash

# please run this at root directory of tensorflow
success=1

for i in `grep -onI https://www.tensorflow.org/code/\[a-zA-Z0-9/._-\]\* -r tensorflow`
do
  filename=`echo $i|awk -F: '{print $1}'`
  linenumber=`echo $i|awk -F: '{print $2}'`
  target=`echo $i|awk -F: '{print $4}'|tail -c +27`

  # skip files in tensorflow/models
  if [[ $target == tensorflow_models/* ]] ; then
    continue
  fi

  if [ ! -f $target ] && [ ! -d $target ]; then
    success=0
    echo Broken link $target at line $linenumber of file $filename
  fi
done

if [ $success == 0 ]; then
  echo Code link check fails.
  exit 1
fi

echo Code link check success.
