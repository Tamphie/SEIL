# Use Ubuntu 20.04 as base
FROM ubuntu:20.04

# Set environment variables to avoid interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive
ENV COPPELIASIM_ROOT=/home/user/CoppeliaSim
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
ENV QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT

# Install required tools and dependencies
RUN apt-get update && apt-get install -y \
    wget \
    python3-pip \
    python3-venv \
    libgl1-mesa-dev \
    libxcb-xinerama0 \
    libxcb-xinput0 \
    libxkbcommon-x11-0 \
    libxcb-render-util0 \
    libx11-6 \
    x11-apps \
    qt5-default \
    git \
    eog \
    && rm -rf /var/lib/apt/lists/*

# Install Python tools and libraries
RUN pip3 install --upgrade pip setuptools && \
    pip3 install virtualenv testresources PyQt5 gymnasium

# Install CoppeliaSim
RUN wget --no-check-certificate https://downloads.coppeliarobotics.com/V4_1_0/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz && \
    mkdir -p $COPPELIASIM_ROOT && \
    tar -xf CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz -C $COPPELIASIM_ROOT --strip-components 1 && \
    rm -rf CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz

# Install RLBench and other dependencies
RUN pip3 install --upgrade pip && \
    pip3 install git+https://github.com/stepjam/RLBench.git && \
    pip3 install opencv-python-headless

# Clone and install RobotIL and robomimic
RUN git clone https://github.com/RobotIL-rls/RobotIL.git /home/user/RobotIL --recursive && \
    git clone https://github.com/RobotIL-rls/robomimic.git /home/user/robomimic && \
    pip3 install -e ./RobotIL && \
    pip3 install -e ./robomimic && \
    git clone https://github.com/your-repo/SEIL.git /home/user/SEIL


# Set up the entry point
WORKDIR /home/user/SEIL
CMD ["/bin/bash"]
