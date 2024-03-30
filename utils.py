import cv2
import numpy as np
from PIL import Image
from simple_lama_inpainting import SimpleLama
simple_lama = SimpleLama()

def warp_with_flow(flow, curImg):
    h, w = flow.shape[:2]
    flow = -flow
    flow[:,:,0] += np.arange(w)
    flow[:,:,1] += np.arange(h)[:,np.newaxis]
    nextImg = cv2.remap(curImg, flow, None, cv2.INTER_LINEAR)
    return nextImg

def ghosting_mask(flow, warpedDepth, prevDepth, 
                  moved_ratio_threshold=0.01, depth_diff_threshold=0.01, debug=True, idx=0):
    '''
    Ghosting mask
    Condition: It moved (has flow) and depth remained similar
    '''
    # --- Flow magnitude ---
    flow_hor = np.abs(flow[:,:,0])
    flow_ver = np.abs(flow[:,:,1])

    mask1 = np.zeros_like(flow_hor)
    mask2 = np.zeros_like(flow_ver)

    threshold = flow.shape[1] * moved_ratio_threshold
    
    mask1[flow_hor > threshold] = 1
    mask2[flow_ver > threshold] = 1

    mask2 = (mask2 * 255).astype(np.uint8)
    mask1 = (mask1 * 255).astype(np.uint8)

    movement_mask = cv2.bitwise_or(mask1, mask2)
    
    # --- Depth difference ---
    diff = cv2.absdiff(warpedDepth, prevDepth)
    _, diff = cv2.threshold(diff, depth_diff_threshold, 255, cv2.THRESH_BINARY)
    diff = diff.astype(np.uint8)
    # Remove noise
    kernel = np.ones((3,3),np.uint8)
    diff = cv2.erode(diff, kernel, iterations=1)
    diff = cv2.dilate(diff, kernel, iterations=2)

    ghosting_mask = cv2.bitwise_and(movement_mask, diff)
    ghosting_mask = cv2.dilate(ghosting_mask, kernel, iterations=3)

    if debug:
        cv2.imwrite(f"debug/depth_diff_theshold{idx}.png", diff)
        cv2.imwrite(f"debug/movement_mask{idx}.png", movement_mask)
        cv2.imwrite(f"debug/ghosting_mask{idx}.png", ghosting_mask)
        w_vis = (warpedDepth * 255).astype(np.uint8)
        cv2.imwrite(f"debug/warped_depth{idx}.png", w_vis)
        
    return ghosting_mask

def inpaint_depth(depth, mask, method="lama", downscale=2):
    og_size = depth.shape
    
    if downscale > 1:
        depth = cv2.resize(depth, (depth.shape[1] // downscale, depth.shape[0] // downscale))
        mask = cv2.resize(mask, (mask.shape[1] // downscale, mask.shape[0] // downscale))
    
    if method == "cv2":
        depth = (depth * 255).astype(np.uint8)
        inpainted_depth = cv2.inpaint(depth, mask, 3, cv2.INPAINT_TELEA)
    else:
        depth = Image.fromarray((depth * 255).astype(np.uint8)).convert("RGB")
        mask_im = Image.fromarray(mask).convert("L")
        inpainted_depth = simple_lama(depth, mask_im)
        inpainted_depth = np.array(inpainted_depth)
    
    return cv2.resize(inpainted_depth, (og_size[1], og_size[0]))