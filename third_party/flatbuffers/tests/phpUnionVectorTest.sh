#!/bin/bash

set -e

../flatc --php -o php union_vector/union_vector.fbs
php phpUnionVectorTest.php

echo 'PHP union vector test passed'
