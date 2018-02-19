#!/usr/bin/env bash

git clone https://github.com/caffe2/caffe2.git caffe2-src

cd caffe2
git submodules update --init
cd ..

mkdir build
cd build

cmake ../caffe2-src -DBLAS=OpenBLAS \
         -DUSE_OPENCV=off \
         -DPYTHON_EXECUTABLE=/usr/local/bin/python3 \
         -DUSE_MPI=OFF \
         -DUSE_CUDA=OFF \
         -DUSE_NNPACK=OFF \
         -DUSE_LEVELDB=OFF \
         `python3 ../caffe2/scripts/get_python_cmake_flags.py`

make

make install
