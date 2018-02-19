#!/usr/bin/env bash

ROOT=`pwd`

CAFFE_SRC=${ROOT}/caffe2-src
BUILD_PATH=${ROOT}/build

if [ -d ${CAFFE_SRC} ];then
    cd ${CAFFE_SRC}
    git reset --hard
    git clean -f
    git pull
else
    cd ${ROOT}
    git clone https://github.com/caffe2/caffe2.git ${CAFFE_SRC}
fi

cd ${CAFFE_SRC}
git submodule update --init

mkdir -p ${BUILD_PATH}
cd ${BUILD_PATH}

cmake ${CAFFE_SRC} -DBLAS=OpenBLAS \
         -DUSE_OPENCV=off \
         -DPYTHON_EXECUTABLE=/usr/local/bin/python3 \
         -DUSE_MPI=OFF \
         -DUSE_CUDA=OFF \
         -DUSE_NNPACK=OFF \
         -DUSE_LEVELDB=OFF \
         `python3 ${CAFFE_SRC}/scripts/get_python_cmake_flags.py`

make

make install
