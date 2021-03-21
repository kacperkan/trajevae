FROM ubuntu:16.04

# mitsuba part

RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y cmake vim git wget 

RUN apt-get install -y \
    build-essential \
    scons \
    git \
    qt5-default \
    libqt5opengl5-dev \
    libqt5xmlpatterns5-dev \
    libpng12-dev \
    libjpeg-dev \
    libilmbase-dev \
    libxerces-c-dev \
    libboost-all-dev \
    libopenexr-dev \
    libglewmx-dev \
    libxxf86vm-dev \
    libpcrecpp0v5 \
    libeigen3-dev \
    libfftw3-dev \
    libcollada-dom2.4-dp0 \ 
    libcollada-dom2.4-dp-dev \
    zlib1g-dev \
    && apt-get clean \
    && apt-get autoclean \
    && apt-get autoremove


RUN DEBIAN_FRONTEND=noninteractive apt-get install -y \
    libx11-dev \
    libxxf86vm-dev \
    x11-xserver-utils \
    x11proto-xf86vidmode-dev \
    x11vnc \
    xpra \
    xserver-xorg-video-dummy \
    && apt-get clean \
    && apt-get autoclean \
    && apt-get autoremove


WORKDIR /mitsuba

RUN git clone https://github.com/kacperkan/mitsuba
RUN cd mitsuba && cp build/config-linux-gcc.py config.py \
    && scons -j`nproc`

ENV MITSUBA_PYVER=3.6

RUN apt-get clean \
    && apt-get autoclean \
    && apt-get autoremove

RUN mkdir renders
VOLUME [ "/mitsuba/renders" ]

COPY xorg.conf /etc/X11/xorg.conf
ENV DISPLAY :0

### wrapper to start headless xserver when using mtsimport
COPY mtsimport-headless.sh /mitsuba/mitsuba/wrapper/mtsimport

# miniconda part

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

RUN apt-get update --fix-missing && \
    apt-get install -y wget bzip2 ca-certificates libglib2.0-0 libxext6 libsm6 libxrender1 git mercurial subversion && \
    apt-get clean

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-4.7.12-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -tipsy && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc && \
    find /opt/conda/ -follow -type f -name '*.a' -delete && \
    find /opt/conda/ -follow -type f -name '*.js.map' -delete && \
    /opt/conda/bin/conda clean -afy

RUN conda install python=3.7.3 flask
RUN conda install -c conda-forge opencv
RUN pip install gunicorn

RUN apt-get install -y zip

COPY service.py .
EXPOSE 8000

CMD /bin/bash -c "source /mitsuba/mitsuba/setpath.sh && gunicorn -w 4 -b 0.0.0.0:8000 --timeout 3600 service:app"

