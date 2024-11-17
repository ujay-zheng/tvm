FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND noninteravtive
RUN ln -fs /usr/share/zoneinfo/Asia/Shanghai /etc/localtime
RUN sed -i 's/archive.ubuntu.com/mirrors.ustc.edu.cn/g' /etc/apt/sources.list && apt-get update
RUN apt-get install -y python3 python3-pip python3-dev python3-setuptools gcc libtinfo-dev zlib1g-dev build-essential cmake libedit-dev libxml2-dev git libglib2.0-dev
ADD .deps/* /opt/
ENV PATH /opt/clang+llvm-10.0.0-x86_64-linux-gnu-ubuntu-18.04/bin/:/opt/cmake-3.25.0-rc1-linux-x86_64/bin:$PATH
RUN dpkg -i /opt/NsightSystems-linux-cli-public-2023.4.1.97-3355750.deb
RUN pip install pytest numpy decorator pandas decorator attrs typing-extensions scipy psutil tornado 'xgboost>=1.1.0' cloudpickle torch torchvision torchaudio matplotlib transformers ipykernel onnxoptimizer -i https://mirrors.ustc.edu.cn/pypi/web/simple
RUN pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121 -i https://mirrors.ustc.edu.cn/pypi/web/simple
ENV TVM_HOME /tvm
ENV PYTHONPATH $TVM_HOME/python:${PYTHONPATH}