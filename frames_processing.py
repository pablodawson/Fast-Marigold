import argparse
import numpy as np
from PIL import Image
from marigold.marigold_pipeline import MarigoldPipeline
from flow_estimation import FlowEstimator
import torch
import glob
from utils import warp_with_flow, ghosting_mask, inpaint_depth
import tqdm
import os
import time

parser = argparse.ArgumentParser(description='Extract depth from video frames.')
parser.add_argument('--frames_path', type=str, help='Path to the directory containing input frames')
parser.add_argument('--noise_ratio', type=float, default=0.65, help='How much to base the prediction on the previous depth map')
parser.add_argument('--blend_ratio', type=float, default=0.2, help='How much to blend in the previous depth map')

parser.add_argument('--remove_ghosting', type=bool, default=True, help='Detect and remove ghosting artifacts')
parser.add_argument('--moved_ratio_threshold', type=float, default=0.01, help='Threshold for movement detection')
parser.add_argument('--depth_diff_threshold', type=float, default=0.01, help='Threshold for depth difference detection')

parser.add_argument('--output_path', type=str, default='video_out_standard', help='Path to save the output frames')
parser.add_argument('--debug', type=bool, default=True)

args = parser.parse_args()

pipe = MarigoldPipeline.from_pretrained(
    "prs-eth/marigold-lcm-v1-0",
    torch_dtype=torch.float16
)
pipe.to("cuda")

flow_estimator = FlowEstimator("gmflow/pretrained/gmflow_things-e9887eda.pth", "cuda")

os.makedirs(args.output_path, exist_ok=True)

images = glob.glob(args.frames_path + "/*.png")
images.sort()

for i, image_path in tqdm.tqdm(enumerate(images), total=len(images)):
    cur_image = Image.open(image_path)
    prev_image = Image.open(images[i - 1]) if i > 0 else None

    if prev_image is not None:
        flow = flow_estimator.estimate_flow(np.array(prev_image), np.array(cur_image))
        warped_depth = warp_with_flow(flow, prev_depth)

        if args.remove_ghosting:
            mask = ghosting_mask(flow, warped_depth, prev_depth, 
                                 moved_ratio_threshold=args.moved_ratio_threshold, depth_diff_threshold=args.depth_diff_threshold,
                                 debug=args.debug, idx=i)

            timestart = time.time()
            
            d_in = Image.fromarray(inpaint_depth(warped_depth, mask))

            if args.debug:
                print(f"Inpainting: {time.time() - timestart}")

        else:
            d_in = Image.fromarray(warped_depth * 255)

    if True:
        pipeline_output = pipe(cur_image, input_depth=None, denoising_steps=1, ensemble_size=5)
        depth = pipeline_output.depth_np
        prev_depth = depth
    else:
        pipeline_output = pipe(cur_image, input_depth=d_in, denoising_steps=1,
                               ensemble_size=1, noise_ratio=args.noise_ratio, input_depth_mix=args.blend_ratio,
                               show_progress_bar=False)
        prev_depth = pipeline_output.depth_np
    
    pipeline_output.depth_colored.save(os.path.join(args.output_path, f"{str(i).zfill(4)}.png"))