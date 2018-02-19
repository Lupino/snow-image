#!/usr/bin/env bash

git clone https://github.com/BVLC/caffe.git caffe-src

mkdir build
cd build

cmake -DCPU_ONLY=ON -DBUILD_python=OFF -DBUILD_SHARED_LIBS=OFF -DUSE_OPENCV=OFF ../caffe-src

make
make install
