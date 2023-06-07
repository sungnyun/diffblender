import numpy as np
import cv2
from os.path import join

import torch
import torch.nn.functional as F
from torchvision.utils import save_image, make_grid
import torchvision

from PIL import Image, ImageDraw
from skimage.color import hsv2rgb, rgb2lab, lab2rgb


PP_RGB = 1
PP_SEGM = 2

def convert_zero2one(image_tensor):
    return (image_tensor.detach().cpu()+1)/2.0

def postprocess(image_tensors, pp_type):
    """
        image_tensors:
            (B x C=3 x H x W)
            torch.tensors
        pp_type:
            int
    """
    if pp_type == PP_RGB:
        return convert_zero2one(image_tensors)
    elif pp_type == PP_SEGM:
        if image_tensors.size(1) == 3:
            return image_tensors.detach().cpu()
        else:
            if image_tensors.ndim == 4:
                return image_tensors.detach().cpu().repeat(1, 3, 1, 1)
            elif image_tensors.ndim == 5:
                return image_tensors.detach().cpu().repeat(1, 1, 3, 1, 1)
            else:
                raise NotImplementedError
    else:
        raise NotImplementedError


def get_scale_factor(image_tensors, feature_tensors):
    """
        image_tensors:
            (B x C=3 x H x W)
        feature_tensors:
            (B x C x h x w)
    """
    B, _, H, W = image_tensors.size()
    _, _, h, w = feature_tensors.size()
    scale_factor = int(H / h)
    return scale_factor

def do_scale(image_tensors, feature_tensors):
    """
        image_tensors:
            (B x C=3 x H x W)
        feature_tensors:
            (B x C x h x w)
    """
    scale_factor = get_scale_factor(image_tensors, feature_tensors)
    scaled_tensor = F.interpolate(feature_tensors, scale_factor=scale_factor)
    return scaled_tensor

def save_images_from_dict(
    image_dict, dir_path, file_name, n_instance, 
    is_save=False, save_per_instance=False, return_images=False,
    save_per_instance_idxs=[]):
    """
        image_dict:
            [
                {
                    "tensors": tensors:tensor, 
                    "n_in_row": int, 
                    "pp_type": int
                },
                ...
    """
    
    n_row = 0

    for each_item in image_dict:
        tensors = each_item["tensors"]
        bs = tensors.size(0)
        n_instance = min(bs, n_instance)
        
        n_in_row = each_item["n_in_row"]
        n_row += n_in_row
        
        pp_type = each_item["pp_type"]
        post_tensor = postprocess(tensors, pp_type=pp_type)
        each_item["tensors"] = torch.clamp(post_tensor, min=0, max=1)

    if save_per_instance:
        for i in range(n_instance):
            image_list = []
            for each_item in image_dict:
                if each_item["n_in_row"] == 1:
                    image_list.append(each_item["tensors"][i].unsqueeze(0))
                else:
                    for j in range(each_item["n_in_row"]):
                        image_list.append(each_item["tensors"][i, j].unsqueeze(0))
            images = torch.cat(image_list, dim=0)
            if len(save_per_instance_idxs) > 0:
                save_path = join(dir_path, f"{file_name}_{save_per_instance_idxs[i]}.png")
            else:
                save_path = join(dir_path, f"{file_name}_{i}.png")
            if is_save:
                save_image(images, save_path, padding=0, pad_value=0.5, nrow=n_row)
    else:
        save_path = join(dir_path, f"{file_name}.png")
        image_list = []
        for i in range(n_instance):
            for each_item in image_dict:
                if each_item["n_in_row"] == 1:
                    image_list.append(each_item["tensors"][i].unsqueeze(0))
                else:
                    for j in range(each_item["n_in_row"]):
                        image_list.append(each_item["tensors"][i, j].unsqueeze(0))
        
        images = torch.cat(image_list, dim=0)
        concated_image = make_grid(images, padding=2, pad_value=0.5, nrow=n_row)
        if is_save:
            save_image(concated_image, save_path, nrow=1)
    
    if return_images:
        return concated_image

### =========================================================================== ###
"""
    functions to visualize boxes
"""
def draw_box(img, boxes):
    colors = ["red", "olive", "blue", "green", "orange", "brown", "cyan", "purple"]
    draw = ImageDraw.Draw(img)
    for bid, box in enumerate(boxes):
        draw.rectangle([box[0], box[1], box[2], box[3]], outline=colors[bid % len(colors)], width=4)
        # draw.rectangle([box[0], box[1], box[2], box[3]], outline ="red", width=2) # x0 y0 x1 y1 
    return img 

def vis_boxes(info_dict):

    device = info_dict["image"].device
    # img = torchvision.transforms.functional.to_pil_image( info_dict["image"]*0.5+0.5 )
    canvas = torchvision.transforms.functional.to_pil_image( torch.zeros_like(info_dict["image"]) )  # unused
    W, H = canvas.size

    boxes = []
    for box in info_dict["boxes"]:    
        x0,y0,x1,y1 = box
        boxes.append( [float(x0*W), float(y0*H), float(x1*W), float(y1*H)] )
    canvas = draw_box(canvas, boxes)
    
    return  torchvision.transforms.functional.to_tensor(canvas).to(device)
### =========================================================================== ###

### =========================================================================== ###
"""
    functions to visualize keypoints
"""
def draw_points(img, points):
    colors = ["red", "yellow", "blue", "green", "orange", "brown", "cyan", "purple", "deeppink", "coral", "gold", "darkblue", "khaki", "lightgreen", "snow", "yellowgreen", "lime"]
    colors = colors * 100
    draw = ImageDraw.Draw(img)
    
    r = 3
    for point, color in zip(points, colors):
        if point[0] == point[1] == 0:
            pass 
        else:
            x, y = float(point[0]), float(point[1])
            draw.ellipse( [ (x-r,y-r), (x+r,y+r) ], fill=color   )
        # draw.rectangle([box[0], box[1], box[2], box[3]], outline ="red", width=2) # x0 y0 x1 y1 
    return img 

def vis_keypoints(info_dict):

    device = info_dict["image"].device
    # img =    torchvision.transforms.functional.to_pil_image( info_dict["image"]*0.5+0.5 )
    _, H, W = info_dict["image"].size()
    canvas = torchvision.transforms.functional.to_pil_image( torch.ones_like(info_dict["image"])*0.2   )
    assert W==H
    img = draw_points( canvas, info_dict["points"]*W )

    return torchvision.transforms.functional.to_tensor(img).to(device)
### =========================================================================== ###



### ================================ Color palette-related functions ================================== ###


class Palette(object):
    """
    Create a color palette (codebook) in the form of a 2D grid of colors,
    as described in the parameters list below.
    Further, the rightmost column has num_hues gradations from black to white.
    Parameters
    ----------
    num_hues : int
        number of colors with full lightness and saturation, in the middle
    sat_range : int
        number of rows above middle row that show
        the same hues with decreasing saturation.
    light_range : int
        number of rows below middle row that show
        the same hues with decreasing lightness.
    Returns
    -------
    palette: rayleigh.Palette
    """

    def __init__(self, num_hues=8, sat_range=2, light_range=2):
        height = 1 + sat_range + (2 * light_range - 1)
        # generate num_hues+1 hues, but don't take the last one:
        # hues are on a circle, and we would be oversampling the origin
        hues = np.tile(np.linspace(0, 1, num_hues + 1)[:-1], (height, 1))
        if num_hues == 8:
            hues = np.tile(np.array(
                [0.,  0.10,  0.15,  0.28, 0.51, 0.58, 0.77,  0.85]), (height, 1))
        if num_hues == 9:
            hues = np.tile(np.array(
                [0.,  0.10,  0.15,  0.28, 0.49, 0.54, 0.60, 0.7, 0.87]), (height, 1))
        if num_hues == 10:
            hues = np.tile(np.array(
                [0.,  0.10,  0.15,  0.28, 0.49, 0.54, 0.60, 0.66, 0.76, 0.87]), (height, 1))
        elif num_hues == 11:
            hues = np.tile(np.array(
                [0.0, 0.0833, 0.166, 0.25,
                 0.333, 0.5, 0.56333,
                 0.666, 0.73, 0.803,
                 0.916]), (height, 1))
        
        sats = np.hstack((
            np.linspace(0, 1, sat_range + 2)[1:-1],
            1,
            [1] * (light_range),
            [.4] * (light_range - 1),
        ))
        lights = np.hstack((
            [1] * sat_range,
            1,
            np.linspace(1, 0.2, light_range + 2)[1:-1],
            np.linspace(1, 0.2, light_range + 2)[1:-2],
        ))

        sats = np.tile(np.atleast_2d(sats).T, (1, num_hues))
        lights = np.tile(np.atleast_2d(lights).T, (1, num_hues))
        colors = hsv2rgb(np.dstack((hues, sats, lights)))
        grays = np.tile(
            np.linspace(1, 0, height)[:, np.newaxis, np.newaxis], (1, 1, 3))

        self.rgb_image = np.hstack((colors, grays))

        # Make a nice histogram ordering of the hues and grays
        h, w, d = colors.shape
        color_array = colors.T.reshape((d, w * h)).T
        h, w, d = grays.shape
        gray_array = grays.T.reshape((d, w * h)).T
        
        self.rgb_array = np.vstack((color_array, gray_array))
        self.lab_array = rgb2lab(self.rgb_array[None, :, :]).squeeze()
        self.hex_list = [rgb2hex(row) for row in self.rgb_array]

    def output(self, dirname):
        """
        Output an image of the palette, josn list of the hex
        colors, and an HTML color picker for it.
        Parameters
        ----------
        dirname : string
            directory for the files to be output
        """
        pass # we do not need this for visualization 

def color_hist_to_palette_image(color_hist, palette, percentile=90,
                                width=200, height=50, filename=None):
    """
    Output the main colors in the histogram to a "palette image."
    Parameters
    ----------
    color_hist : (K,) ndarray
    palette : rayleigh.Palette
    percentile : int, optional:
        Output only colors above this percentile of prevalence in the histogram.
    filename : string, optional:
        If given, save the resulting image to file.
    Returns
    -------
    rgb_image : ndarray
    """
    ind = np.argsort(-color_hist)
    ind = ind[color_hist[ind] > np.percentile(color_hist, percentile)]
    hex_list = np.take(palette.hex_list, ind)
    values = color_hist[ind]
    rgb_image = palette_query_to_rgb_image(dict(zip(hex_list, values)), width, height)
    if filename:
        imsave(filename, rgb_image)
    return rgb_image

def palette_query_to_rgb_image(palette_query, width=200, height=50):
    """
    Convert a list of hex colors and their values to an RGB image of given
    width and height.
    Args:
        - palette_query (dict):
            a dictionary of hex colors to unnormalized values,
            e.g. {'#ffffff': 20, '#33cc00': 0.4}.
    """
    hex_list, values = zip(*palette_query.items())
    values = np.array(values)
    values /= values.sum()
    nums = np.array(values * width, dtype=int)
    rgb_arrays = (np.tile(np.array(hex2rgb(x)), (num, 1))
                  for x, num in zip(hex_list, nums))
    rgb_array = np.vstack(list(rgb_arrays))
    rgb_image = rgb_array[np.newaxis, :, :]
    rgb_image = np.tile(rgb_image, (height, 1, 1))
    return rgb_image

def rgb2hex(rgb_number):
    """
    Args:
        - rgb_number (sequence of float)
    Returns:
        - hex_number (string)
    """
    return '#{:02x}{:02x}{:02x}'.format(*tuple([int(np.round(val * 255)) for val in rgb_number]))

def hex2rgb(hexcolor_str):
    """
    Args:
        - hexcolor_str (string): e.g. '#ffffff' or '33cc00'
    Returns:
        - rgb_color (sequence of floats): e.g. (0.2, 0.3, 0)
    """
    color = hexcolor_str.strip('#')
    return tuple(round(int(color[i:i+2], 16) / 255., 5) for i in (0, 2, 4))

def vis_color_palette(color_hist, shape):
    if color_hist.sum() == 0:
        color_hist[-1] = 1.0
    palette = Palette(num_hues=11, sat_range=5, light_range=5)
    color_palette = color_hist_to_palette_image(color_hist.cpu().numpy(), palette, percentile=90)
    color_palette = torch.tensor(color_palette.transpose(2,0,1)).unsqueeze(0)
    color_palette = F.interpolate(color_palette, size=(shape, shape), mode='nearest')
    return color_palette.squeeze(0)

