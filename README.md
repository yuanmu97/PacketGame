# PacketGame

PacketGame is a pre-decoding plug-in for concurrent video inference at scale.

Our paper "PacketGame: Multi-Stream Packet Gating for Concurrent Video Inference at Scale" is going to appear at *ACM SIGCOMM 2023*.

## Installation

OS: Ubuntu 20.04

### FFmpeg with nv-codec

To use FFmpeg with NVIDIA GPU, we need to compile in from source (refers to [NVIDIA doc](https://docs.nvidia.com/video-technologies/video-codec-sdk/11.1/ffmpeg-with-nvidia-gpu/index.html)).

Install ffnvcodec:
```bash
git clone https://git.videolan.org/git/ffmpeg/nv-codec-headers.git
cd nv-codec-headers
make install
```

Install necessary packages:
```bash
apt-get install yasm cmake 
# codecs: h.264, h.265, vp9, jp2k
apt-get install libx264-dev libx265-dev libvpx-dev libopenjp2-7-dev 
```

Download ([v5.1](https://github.com/FFmpeg/FFmpeg/tree/release/5.1)) and install FFmpeg:
```bash
cd FFmpeg-release-5.1/
./configure --enable-nonfree --enable-cuda-nvcc --enable-libnpp --extra-cflags=-I/usr/local/cuda/include --extra-ldflags=-L/usr/local/cuda/lib64 --disable-static --enable-shared --enable-gpl --enable-libx264 --enable-libx265 --enable-libvpx --enable-libopenjpeg
make -j 8
make install
# test
ffmpeg
-------------------------------------------------------------------
ffmpeg version 5.1.2 Copyright (c) 2000-2022 the FFmpeg developers
  built with gcc 9 (Ubuntu 9.4.0-1ubuntu1~20.04.1)
  configuration: --enable-nonfree --enable-cuda-nvcc --enable-libnpp --extra-cflags=-I/usr/local/cuda/include --extra-ldflags=-L/usr/local/cuda/lib64 --disable-static --enable-shared --enable-gpl --enable-libx264 --enable-libx265 --enable-libvpx --enable-libopenjpeg
  libavutil      57. 28.100 / 57. 28.100
  libavcodec     59. 37.100 / 59. 37.100
  libavformat    59. 27.100 / 59. 27.100
  libavdevice    59.  7.100 / 59.  7.100
  libavfilter     8. 44.100 /  8. 44.100
  libswscale      6.  7.100 /  6.  7.100
  libswresample   4.  7.100 /  4.  7.100
  libpostproc    56.  6.100 / 56.  6.100
Hyper fast Audio and Video encoder
usage: ffmpeg [options] [[infile options] -i infile]... {[outfile options] outfile}...
```

## Packet Parser

The first step is to parse the video and save its metadata (packet size and picture type).

```bash
mkdir build
cd build
cmake ..
make
# test
./parser ../test/sample_video.h265 ../test/sampel_video_meta.txt
-----------------------------------------------------------------
251 packets parsed in 0.007604 seconds.
```

## Concurrent Decoding

Platform: 12 Intel Core i7-5930K CPUs / NVIDIA TITAN X GPU

```bash
cd test
# set USEGPU=1
python concurrent_decode.py
----------------------------------------
  concurrency    time cost (s)      fps
-------------  ---------------  -------
            1          1.51226  165.316
            5          3.70409  337.465
           10          6.72792  371.586
           20         13.6818   365.449
           30         19.5902   382.845
           35         18.7045   467.803
           40         23.5301   424.988
           45         25.1256   447.75
           50         27.077    461.646
```

## License

PacketGame is licensed under the [MIT License](./LICENSE).