import subprocess
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import time
from tabulate import tabulate

USEGPU = 0

cmd_list = []

vid_dir = "video_h1/"
for vid in os.listdir(vid_dir):
    if vid.endswith(".h265"):
        vid_path = os.path.join(vid_dir, vid)
        # reference: https://trac.ffmpeg.org/wiki/Null#:~:text=ffmpeg%20%2Di%20input%20%2Df%20null%20%2D
        if USEGPU:
            cmd_list.append(f"ffmpeg -hwaccel cuda -i {vid_path} -f null -".split())
        else:
            cmd_list.append(f"ffmpeg -i {vid_path} -f null -".split())

concur_levels = [10, 50]
outputs = []
for concur in concur_levels:

    processes = []
    for cmd in cmd_list[:concur]:
        p = subprocess.Popen(cmd)
        processes.append(p)
    
    start_t = time.time()
    for p in processes:
        p.wait()
    t = time.time() - start_t
    fps = 250*concur / t

    outputs.append([concur, t, fps])

print(tabulate(outputs, headers=["concurrency", "time cost (s)", "fps"]))