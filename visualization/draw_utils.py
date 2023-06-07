import torch
from copy import deepcopy
import visualization.image_utils as iutils



def draw_sketch_with_batch_to_tensor(batch):
    if "sketch" not in batch or batch["sketch"] is None:
        return torch.zeros_like(batch["image"])
    else:
        return batch["sketch"]["values"]

def draw_depth_with_batch_to_tensor(batch):
    if "depth" not in batch or batch["depth"] is None:
        return torch.zeros_like(batch["image"])
    else:
        return batch["depth"]["values"]

def draw_boxes_with_batch_to_tensor(batch):
    if "box" not in batch or batch["box"] is None:
        return torch.zeros_like(batch["image"])

    batch_size = batch["image"].size(0)
    box_drawing = []
    for i in range(batch_size):
        if "box" in batch:
            info_dict = {"image": batch["image"][i], "boxes": batch["box"]["values"][i]}
            boxed_img = iutils.vis_boxes(info_dict)
        else:
            boxed_img = torch.randn_like(batch["image"][i])
        box_drawing.append(boxed_img)
    box_tensor = torch.stack(box_drawing)
    return box_tensor

def draw_keypoints_with_batch_to_tensor(batch):
    if "keypoint" not in batch or batch["keypoint"] is None:
        return torch.zeros_like(batch["image"])

    batch_size = batch["image"].size(0)
    keypoint_drawing = []
    for i in range(batch_size):
        if "keypoint" in batch:
            info_dict = {"image": batch["image"][i], "points": batch["keypoint"]["values"][i]}
            keypointed_img = iutils.vis_keypoints(info_dict)
        else:
            keypointed_img = torch.randn_like(batch["image"][i])
        keypoint_drawing.append(keypointed_img)
    keypoint_tensor = torch.stack(keypoint_drawing)
    return keypoint_tensor

def draw_color_palettes_with_batch_to_tensor(batch):
    if "color_palette" not in batch or batch["color_palette"] is None:
        return torch.zeros_like(batch["image"])
    try:
        batch_size = batch["image"].size(0)
        color_palette_drawing = []
        for i in range(batch_size):
            if "color_palette" in batch:
                color_hist = deepcopy(batch["color_palette"]["values"][i])
                color_palette = iutils.vis_color_palette(color_hist, batch["image"].shape[-1])
            else:
                color_palette = torch.randn_like(batch["image"][i])
            color_palette_drawing.append(color_palette)
        color_palette_tensor = torch.stack(color_palette_drawing)
        return color_palette_tensor
    except:
        print(f">> Exception occured in draw_color_palettes_with_batch_to_tensor(batch)..")
        return torch.zeros_like(batch["image"])
    
def draw_image_embedding_with_batch_to_tensor(batch):
    if "image_embedding" not in batch or batch["image_embedding"] is None:
        return -torch.ones_like(batch["image"])
    else:
        if "image" in batch["image_embedding"]:
            return batch["image_embedding"]["image"]
        else: return -torch.ones_like(batch["image"])


