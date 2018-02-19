#!/usr/bin/env bash

ROOT=`pwd`

CAFFE_SRC=${ROOT}/caffe-src
BUILD_PATH=${ROOT}/build

if [ -d ${CAFFE_SRC} ];then
    cd ${CAFFE_SRC}
    git reset --hard
    git clean -f
    git pull
else
    cd ${ROOT}
    git clone https://github.com/BVLC/caffe.git ${CAFFE_SRC}
fi

mkdir -p ${BUILD_PATH}
cd ${BUILD_PATH}

cmake -DCPU_ONLY=ON \
      -DBUILD_python=OFF \
      -DBUILD_SHARED_LIBS=OFF \
      -DUSE_OPENCV=OFF \
      ${CAFFE_SRC}

make
make install
