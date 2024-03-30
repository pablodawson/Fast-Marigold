import argparse
import numpy as np
from PIL import Image
from marigold.marigold_pipeline import MarigoldPipeline
from flow_estimation import FlowEstimator
import torch
import cv2
from utils import warp_with_flow, ghosting_mask, inpaint_depth
import tqdm
import os
import time

parser = argparse.ArgumentParser(description='Extract depth from video frames.')
parser.add_argument('--video_path', type=str, help='Path to the directory containing input frames')
parser.add_argument('--duration', type=int, default=120, help='Duration in frames to be processed')
parser.add_argument('--fps', type=int, default=30, help='Output video fps')
parser.add_argument('--noise_ratio', type=float, default=0.65, help='How much to base the prediction on the previous depth map')
parser.add_argument('--blend_ratio', type=float, default=0.2, help='How much to blend in the previous depth map')

parser.add_argument('--remove_ghosting', type=bool, default=True, help='Detect and remove ghosting artifacts')
parser.add_argument('--moved_ratio_threshold', type=float, default=0.01, help='Threshold for movement detection')
parser.add_argument('--depth_diff_threshold', type=float, default=0.01, help='Threshold for depth difference detection')

parser.add_argument('--output', type=str, default='depth.mp4', help='Path to save the output frames')
parser.add_argument('--debug', type=bool, default=False)

args = parser.parse_args()

pipe = MarigoldPipeline.from_pretrained(
    "prs-eth/marigold-lcm-v1-0",
    torch_dtype=torch.float16
)
pipe.to("cuda")

flow_estimator = FlowEstimator("gmflow/pretrained/gmflow_things-e9887eda.pth", "cuda")

#os.makedirs(args.output_path, exist_ok=True)

cap = cv2.VideoCapture(args.video_path)
ret,frame = cap.read()
frame_height, frame_width, _ = frame.shape
out = cv2.VideoWriter(args.output ,cv2.VideoWriter_fourcc('X','V','I','D'), args.fps, (frame_width,frame_height))

prev_image = None
idx = 0

while cap.isOpened():
    timestart = time.time()
    ret,frame = cap.read()
    if not ret:
        break

    cur_image = Image.fromarray(frame)

    if prev_image is not None:
        flow = flow_estimator.estimate_flow(np.array(prev_image), np.array(cur_image))
        warped_depth = warp_with_flow(flow, prev_depth)

        if args.remove_ghosting:
            mask = ghosting_mask(flow, warped_depth, prev_depth, 
                                 moved_ratio_threshold=args.moved_ratio_threshold, depth_diff_threshold=args.depth_diff_threshold,
                                 debug=args.debug, idx=idx)

            timestart = time.time()
            
            d_in = Image.fromarray(inpaint_depth(warped_depth, mask))
        else:
            d_in = Image.fromarray(warped_depth * 255)

    if idx==0:
        pipeline_output = pipe(cur_image, input_depth=None, denoising_steps=1, ensemble_size=5, show_progress_bar=False)
        depth = pipeline_output.depth_np
        prev_depth = depth
    else:
        pipeline_output = pipe(cur_image, input_depth=d_in, denoising_steps=1,
                               ensemble_size=1, noise_ratio=args.noise_ratio, input_depth_mix=args.blend_ratio,
                               show_progress_bar=False)
        prev_depth = pipeline_output.depth_np
    
    idx += 1
    prev_image = cur_image

    if idx > args.duration:
        break

    print(f"Frame {idx} processed in {time.time() - timestart} seconds")
    
    # video has a transient for some reason, comment condition to save all frames instead
    if idx>10:
        out.write(cv2.cvtColor(np.array(pipeline_output.depth_colored), cv2.COLOR_RGB2BGR))

cap.release()
out.release()