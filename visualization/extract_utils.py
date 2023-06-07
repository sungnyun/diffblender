import torch
import numpy as np

from PIL import Image
from transformers import CLIPProcessor, CLIPModel

from skimage.color import hsv2rgb, rgb2lab, lab2rgb
from skimage.io import imread, imsave
from sklearn.metrics import euclidean_distances


@torch.no_grad()
def get_clip_feature(img_path, model, processor):
    # clip_features = dict()
    image = Image.open(img_path).convert("RGB")
    inputs = processor(images=[image], return_tensors="pt", padding=True)
    inputs['pixel_values'] = inputs['pixel_values'].cuda()
    inputs['input_ids'] = torch.tensor([[0,1,2,3]]).cuda()
    outputs = model(**inputs)
    feature = outputs.image_embeds
    return feature.squeeze().cpu().numpy() # dim: [768,]





'---------------------------------------- color converter ----------------------------------------'

def rgb2hex(rgb_number):
    """
    Args:
        - rgb_number (sequence of float)
    Returns:
        - hex_number (string)
    """
    # print(rgb_number, [np.round(val*255) for val in rgb_number])
    return '#{:02x}{:02x}{:02x}'.format(*tuple([int(np.round(val * 255)) for val in rgb_number]))


def hex2rgb(hexcolor_str):
    """
    Args:
        - hexcolor_str (string): e.g. '#ffffff' or '33cc00'
    Returns:
        - rgb_color (sequence of floats): e.g. (0.2, 0.3, 0)
    """
    color = hexcolor_str.strip('#')
    # rgb = lambda x: round(int(x, 16) / 255., 5)
    return tuple(round(int(color[i:i+2], 16) / 255., 5) for i in (0, 2, 4))



'---------------------------------------- color palette histogram ----------------------------------------'

def histogram_colors_smoothed(lab_array, palette, sigma=10,
                              plot_filename=None, direct=True):
    """
    Returns a palette histogram of colors in the image, smoothed with
    a Gaussian. Can smooth directly per-pixel, or after computing a strict
    histogram.
    Parameters
    ----------
    lab_array : (N,3) ndarray
        The L*a*b color of each of N pixels.
    palette : rayleigh.Palette
        Containing K colors.
    sigma : float
        Variance of the smoothing Gaussian.
    direct : bool, optional
        If True, constructs a smoothed histogram directly from pixels.
        If False, constructs a nearest-color histogram and then smoothes it.
    Returns
    -------
    color_hist : (K,) ndarray
    """
    if direct:
        color_hist_smooth = histogram_colors_with_smoothing(
            lab_array, palette, sigma)
    else:
        color_hist_strict = histogram_colors_strict(lab_array, palette)
        color_hist_smooth = smooth_histogram(color_hist_strict, palette, sigma)
    if plot_filename is not None:
        plot_histogram(color_hist_smooth, palette, plot_filename)
    return color_hist_smooth

def smooth_histogram(color_hist, palette, sigma=10):
    """
    Smooth the given palette histogram with a Gaussian of variance sigma.
    Parameters
    ----------
    color_hist : (K,) ndarray
    palette : rayleigh.Palette
        containing K colors.
    Returns
    -------
    color_hist_smooth : (K,) ndarray
    """
    n = 2. * sigma ** 2
    weights = np.exp(-palette.distances / n)
    norm_weights = weights / weights.sum(1)[:, np.newaxis]
    color_hist_smooth = (norm_weights * color_hist).sum(1)
    color_hist_smooth[color_hist_smooth < 1e-5] = 0
    return color_hist_smooth

def histogram_colors_with_smoothing(lab_array, palette, sigma=10):
    """
    Assign colors in the image to nearby colors in the palette, weighted by
    distance in Lab color space.
    Parameters
    ----------
    lab_array (N,3) ndarray:
        N is the number of data points, columns are L, a, b values.
    palette : rayleigh.Palette
        containing K colors.
    sigma : float
        (0,1] value to control the steepness of exponential falloff.
        To see the effect:
    >>> from pylab import *
    >>> ds = linspace(0,5000) # squared distance
    >>> sigma=10; plot(ds, exp(-ds/(2*sigma**2)), label='$\sigma=%.1f$'%sigma)
    >>> sigma=20; plot(ds, exp(-ds/(2*sigma**2)), label='$\sigma=%.1f$'%sigma)
    >>> sigma=40; plot(ds, exp(-ds/(2*sigma**2)), label='$\sigma=%.1f$'%sigma)
    >>> ylim([0,1]); legend();
    >>> xlabel('Squared distance'); ylabel('Weight');
    >>> title('Exponential smoothing')
    >>> #plt.savefig('exponential_smoothing.png', dpi=300)
        sigma=20 seems reasonable: hits 0 around squared distance of 4000.
    Returns:
    color_hist : (K,) ndarray
        the normalized, smooth histogram of colors.
    """
    dist = euclidean_distances(palette.lab_array, lab_array, squared=True).T
    n = 2. * sigma ** 2
    weights = np.exp(-dist / n)
    
    # normalize by sum: if a color is equally well represented by several colors
    # it should not contribute much to the overall histogram
    normalizing = weights.sum(1)
    normalizing[normalizing == 0] = 1e16
    normalized_weights = weights / normalizing[:, np.newaxis]

    color_hist = normalized_weights.sum(0)
    color_hist /= lab_array.shape[0]
    color_hist[color_hist < 1e-5] = 0
    return color_hist

def histogram_colors_strict(lab_array, palette, plot_filename=None):
    """
    Return a palette histogram of colors in the image.
    Parameters
    ----------
    lab_array : (N,3) ndarray
        The L*a*b color of each of N pixels.
    palette : rayleigh.Palette
        Containing K colors.
    plot_filename : string, optional
        If given, save histogram to this filename.
    Returns
    -------
    color_hist : (K,) ndarray
    """
    # This is the fastest way that I've found.
    # >>> %%timeit -n 200 from sklearn.metrics import euclidean_distances
    # >>> euclidean_distances(palette, lab_array, squared=True)
    dist = euclidean_distances(palette.lab_array, lab_array, squared=True).T
    min_ind = np.argmin(dist, axis=1)
    num_colors = palette.lab_array.shape[0]
    num_pixels = lab_array.shape[0]
    color_hist = 1. * np.bincount(min_ind, minlength=num_colors) / num_pixels
    if plot_filename is not None:
        plot_histogram(color_hist, palette, plot_filename)
    return color_hist

def plot_histogram(color_hist, palette, plot_filename=None):
    """
    Return Figure containing the color palette histogram.
    Args:
        - color_hist (K, ndarray)
        - palette (Palette)
        - plot_filename (string) [default=None]:
                Save histogram to this file, if given.
    Returns:
        - fig (Figure)
    """
    fig = plt.figure(figsize=(5, 3), dpi=150)
    ax = fig.add_subplot(111)
    ax.bar(
        range(len(color_hist)), color_hist,
        color=palette.hex_list)
    ax.set_ylim((0, 0.1))
    ax.xaxis.set_ticks([])
    ax.set_xlim((0, len(palette.hex_list)))
    if plot_filename:
        fig.savefig(plot_filename, dpi=150, facecolor='none')
    return fig


'---------------------------------------- histogram to image ----------------------------------------'
# only for visualization
# for extracting color palette, color histogram is enough.

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
    rgb_image = palette_query_to_rgb_image(dict(zip(hex_list, values)))
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
    rgb_array = np.vstack(rgb_arrays)
    rgb_image = rgb_array[np.newaxis, :, :]
    rgb_image = np.tile(rgb_image, (height, 1, 1))
    return rgb_image




'---------------------------------------- Color Palette ----------------------------------------'

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
        #assert(np.all(self.rgb_array == self.rgb_array[None, :, :].squeeze()))

        self.distances = euclidean_distances(self.lab_array, squared=True)


'---------------------------------------- Image Wrapper for lab_array ----------------------------------------'

class ColorImage(object):
    """
    Read the image at the URL in RGB format, downsample if needed,
    and convert to Lab colorspace.
    Store original dimensions, resize_factor, and the filename of the image.
    Image dimensions will be resized independently such that neither width nor
    height exceed the maximum allowed dimension MAX_DIMENSION.
    Parameters
    ----------
    url : string
        URL or file path of the image to load.
    id : string, optional
        Name or some other id of the image. For example, the Flickr ID.
    """

    MAX_DIMENSION = 240 + 1

    def __init__(self, url, _id=None):
        self.id = _id
        self.url = url
        img = imread(url)

        # Handle grayscale and RGBA images.
        # TODO: Should be smarter here in the future, but for now simply remove
        # the alpha channel if present.
        if img.ndim == 2:
            img = np.tile(img[:, :, np.newaxis], (1, 1, 3))
        elif img.ndim == 4:
            img = img[:, :, :3]
        elif img.ndim == 3 and img.shape[2] == 4:
            img = img[:, :, :3]
        
        # Downsample for speed.
        #
        # NOTE: I can't find a good method to resize properly in Python!
        # scipy.misc.imresize uses PIL, which needs 8bit data.
        # Anyway, this is faster and almost as good.
        #
        # >>> def d(dim, max_dim): return arange(0, dim, dim / max_dim + 1).shape
        # >>> plot(range(1200), [d(x, 200) for x in range(1200)])
        h, w, d = tuple(img.shape)
        self.orig_h, self.orig_w, self.orig_d = tuple(img.shape)
        h_stride = h // self.MAX_DIMENSION + 1
        w_stride = w // self.MAX_DIMENSION + 1
        img = img[::h_stride, ::w_stride, :]

        # Convert to L*a*b colors.
        h, w, d = img.shape
        self.h, self.w, self.d = img.shape
        self.lab_array = rgb2lab(img).reshape((h * w, d))

    def as_dict(self):
        """
        Return relevant info about self in a dict.
        """
        return {'id': self.id, 'url': self.url,
                'resized_width': self.w, 'resized_height': self.h,
                'width': self.orig_w, 'height': self.orig_h}

    def output_quantized_to_palette(self, palette, filename):
        """
        Save to filename a version of the image with all colors quantized
        to the nearest color in the given palette.
        Parameters
        ----------
        palette : rayleigh.Palette
            Containing K colors.
        filename : string
            Where image will be written.
        """
        dist = euclidean_distances(
            palette.lab_array, self.lab_array, squared=True).T
        min_ind = np.argmin(dist, axis=1)
        quantized_lab_array = palette.lab_array[min_ind, :]
        img = lab2rgb(quantized_lab_array.reshape((self.h, self.w, self.d)))
        imsave(filename, img)


def get_color_palette(img_path):
    palette = Palette(num_hues=11, sat_range=5, light_range=5)
    assert len(palette.hex_list) == 180
    query_img = ColorImage(url=img_path)
    color_hist = histogram_colors_smoothed(query_img.lab_array, palette, sigma=10, direct=False)
    return color_hist


@torch.no_grad()
def get_box(locations, phrases, model, processor):
    boxes = torch.zeros(30, 4)
    masks = torch.zeros(30)
    text_embeddings = torch.zeros(30, 768)
    
    text_features = []
    image_features = []
    for phrase in phrases:
        inputs = processor(text=phrase,  return_tensors="pt", padding=True)
        inputs['input_ids'] = inputs['input_ids'].cuda()
        inputs['pixel_values'] = torch.ones(1,3,224,224).cuda() # placeholder 
        inputs['attention_mask'] = inputs['attention_mask'].cuda()
        outputs = model(**inputs)
        feature = outputs.text_model_output.pooler_output
        text_features.append(feature.squeeze())

    for idx, (box, text_feature) in enumerate(zip(locations, text_features)):
        boxes[idx] = torch.tensor(box)
        masks[idx] = 1
        text_embeddings[idx] = text_feature

    return boxes, masks, text_embeddings


def get_keypoint(locations):
    points = torch.zeros(8*17,2)
    idx = 0 
    for this_person_kp in locations:
        for kp in this_person_kp:
            points[idx,0] = kp[0]
            points[idx,1] = kp[1]
            idx += 1
    
    # derive masks from points
    masks = (points.mean(dim=1)!=0) * 1 
    masks = masks.float()
    return points, masks


def transform_image_from_pil_to_numpy(pil_image):
    np_image, trans_info = center_crop_arr(pil_image, image_size=512)
    
    if np_image.ndim == 2: # when we load the image with "L" option
        np_image = np.expand_dims(np_image, axis=2)
    return np_image, trans_info

def flip_image_from_numpy_to_numpy(np_image, trans_info, is_flip=False):
    if is_flip:
        np_image = np_image[:, ::-1]
        trans_info["performed_flip"] = True
        return np_image, trans_info
    else:
        return np_image, trans_info

def convert_image_from_numpy_to_tensor(np_image, type_mask=False):
    if type_mask:
        """
        value range : (0, 1), for sketch, segm, mask, 
        """
        np_image = np_image.astype(np.float32) / 255.0
    else:
        """
        value range : (-1, 1), for rgb
        """
        np_image = np_image.astype(np.float32) / 127.5 - 1
    np_image = np.transpose(np_image, [2,0,1])
    return torch.tensor(np_image)

def invert_image_from_numpy_to_numpy(np_image):
    return 255.0 - np_image
    
def center_crop_arr(pil_image, image_size):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    WW, HH = pil_image.size

    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)

    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.Resampling.BICUBIC
    )

    # at this point, the min of pil_image side is desired image_size
    performed_scale = image_size / min(WW, HH)

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    
    info = {"performed_scale":performed_scale, 'crop_y':crop_y, 'crop_x':crop_x, "WW":WW, 'HH':HH, "performed_flip": False}

    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size], info


def get_sketch(img_path):
    pil_sketch = Image.open(img_path).convert('RGB')
    np_sketch,_  = transform_image_from_pil_to_numpy(pil_sketch)
    np_sketch = invert_image_from_numpy_to_numpy(np_sketch)
    sketch_tensor = convert_image_from_numpy_to_tensor(np_sketch, type_mask=True) # mask type: range 0 to 1
    return sketch_tensor


def get_depth(img_path):
    pil_depth = Image.open(img_path).convert('RGB')
    np_depth,_  = transform_image_from_pil_to_numpy(pil_depth)
    depth_tensor = convert_image_from_numpy_to_tensor(np_depth, type_mask=True) # mask type: range 0 to 1
    return depth_tensor

