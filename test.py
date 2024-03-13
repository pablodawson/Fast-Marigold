import numpy as np
from PIL import Image
#from diffusers import DiffusionPipeline
from marigold.marigold_pipeline import MarigoldPipeline
from diffusers.utils import load_image
import torch

pipe = MarigoldPipeline.from_pretrained(
    "Bingxin/Marigold",
    torch_dtype=torch.float16,                # (optional) Run with half-precision (16-bit float).
)

pipe.to("cuda")

image: Image.Image = Image.open("input/man.png")
depth: Image.Image = Image.open("input/man_depth.png")

pipeline_output = pipe(
    image,                  # Input image.
    input_depth= depth,                  # Input depth map.
    denoising_steps=10,     # (optional) Number of denoising steps of each inference pass. Default: 10.
    ensemble_size=10,       # (optional) Number of inference passes in the ensemble. Default: 10.
    # processing_res=768,     # (optional) Maximum resolution of processing. If set to 0: will not resize at all. Defaults to 768.
    # match_input_res=True,   # (optional) Resize depth prediction to match input resolution.
    # batch_size=0,           # (optional) Inference batch size, no bigger than `num_ensemble`. If set to 0, the script will automatically decide the proper batch size. Defaults to 0.
    # color_map="Spectral",   # (optional) Colormap used to colorize the depth map. Defaults to "Spectral". Set to `None` to skip colormap generation.
    # show_progress_bar=True, # (optional) If true, will show progress bars of the inference progress.
)

depth: np.ndarray = pipeline_output.depth_np                    # Predicted depth map
depth_colored: Image.Image = pipeline_output.depth_colored      # Colorized prediction

# Save as uint16 PNG
depth_uint16 = (depth * 65535.0).astype(np.uint16)
Image.fromarray(depth_uint16).save("./ensemble10steps10.png", mode="I;16")

# Save colorized depth map
depth_colored.save("./depth_colored.png")