#!/bin/bash
# This scripts downloads the ptb data and unzips it.

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
cd $DIR

echo "Downloading..."

mkdir -p data && cd data
wget --continue http://russellsstewart.com/static/content/inception_v1.ckpt
wget --continue http://download.tensorflow.org/models/resnet_v1_101_2016_08_28.tar.gz
if [[ "$2" == '--load_experimental' ]]; then
    wget --continue http://download.tensorflow.org/models/inception_resnet_v2_2016_08_30.tar.gz
    wget --continue http://download.tensorflow.org/models/mobilenet_v1_1.0_224_2017_06_14.tar.gz
fi

mkdir -p overfeat_rezoom && cd overfeat_rezoom
wget --continue http://russellsstewart.com/static/content/save.ckpt-150000v2
cd ..
echo "Extracting..."
tar xf resnet_v1_101_2016_08_28.tar.gz
if [[ "$2" == '--load_experimental' ]]; then
    tar xf inception_resnet_v2_2016_08_30.tar.gz
    tar xf mobilenet_v1_1.0_224_2017_06_14.tar.gz
fi

if [[ "$1" == '--travis_tiny_data' ]]; then
    wget --continue http://russellsstewart.com/s/brainwash_tiny.tar.gz
    echo "Extracting from tar file."
    tar xf brainwash_tiny.tar.gz
    echo "Done."
else
    wget --continue https://stacks.stanford.edu/file/druid:sx925dc9385/brainwash.tar.gz
    echo "Extracting from tar file."
    tar xf brainwash.tar.gz
    echo "Done."
fi
