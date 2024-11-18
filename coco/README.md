# Environment Preparation
* git clone --recursive https://github.com/ujay-zheng/tvm.git
* cd tvm && git checkout multi_stream && cd coco/docker
* download softwares to coco/docker/.deps
  * clang+llvm-10.0.0-x86_64-linux-gnu-ubuntu-18.04
  * cmake-3.25.0-rc1-linux-x86_64
  * NsightSystems-linux-cli-public-2023.4.1.97-3355750
* docker build --rm . -t tvm:multi_stream
* docker run -d --privileged --gpus all -v $TVM_PATH:/tvm --name coco tvm:multi_stream
* docker exec -it coco /bin/bash
* cd /tvm && mkdir build
* cp coco/docker/A100_config.cmake build/config.cmake
* cd build && cmake .. && make -j