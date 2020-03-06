FROM tensorflow/tensorflow:1.3.0-devel-gpu-py3

RUN mkdir -p /capstone/training && \
    cd /capstone/training && \
    git clone https://github.com/bwosh/CarND-TrafficLights.git && \
    /bin/bash -c "pip install tensorflow-gpu==1.3.0 keras==2.0.8"

ENTRYPOINT /bin/bash -c "cd /capstone/training && jupyter notebook --no-browser --allow-root --ip=0.0.0.0 --port=8888" 