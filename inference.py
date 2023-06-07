import os
import argparse
import random
from PIL import Image
from omegaconf import OmegaConf
from copy import deepcopy
from functools import partial

import torch
import torchvision
import torchvision.transforms as tf
from torch.utils.data import DataLoader
import pytorch_lightning

from ldm.util import instantiate_from_config
from ldm.models.diffusion.plms import PLMSSampler
from transformers import CLIPProcessor, CLIPModel
from visualization.extract_utils import get_sketch, get_depth, get_box, get_keypoint, get_color_palette, get_clip_feature
import visualization.image_utils as iutils
from visualization.draw_utils import *


device = "cuda"


def read_official_ckpt(ckpt_path, no_model=False):      
    state_dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    out = {}
    out["model"] = {}
    out["text_encoder"] = {}
    out["autoencoder"] = {}
    out["unexpected"] = {}
    out["diffusion"] = {}

    for k,v in state_dict.items():
        if k.startswith('model.diffusion_model'):
            if no_model:
                continue
            out["model"][k.replace("model.diffusion_model.", "")] = v 
        elif k.startswith('cond_stage_model'):
            out["text_encoder"][k.replace("cond_stage_model.", "")] = v 
        elif k.startswith('first_stage_model'):
            out["autoencoder"][k.replace("first_stage_model.", "")] = v 
        elif k in ["model_ema.decay", "model_ema.num_updates"]:
            out["unexpected"][k] = v  
        else:
            out["diffusion"][k] = v     

    if no_model: 
        del state_dict
    return out 

def batch_to_device(batch, device):
    for k in batch:
        if isinstance(batch[k], torch.Tensor):
            batch[k] = batch[k].to(device)
        
        elif isinstance(batch[k], dict):
            for k_2 in batch[k]:
                if isinstance(batch[k][k_2], torch.Tensor):
                    batch[k][k_2] = batch[k][k_2].to(device)
    return batch

def load_ckpt(ckpt_path, official_ckpt_path='./sd-v1-4.ckpt'):

    saved_ckpt = torch.load(ckpt_path)
    config = saved_ckpt["config_dict"]["_content"]
    
    model = instantiate_from_config(config['model']).to(device).eval()
    autoencoder = instantiate_from_config(config['autoencoder']).to(device).eval()
    text_encoder = instantiate_from_config(config['text_encoder']).to(device).eval()
    diffusion = instantiate_from_config(config['diffusion']).to(device)

    # donot need to load official_ckpt for self.model here, since we will load from our ckpt
    missing, unexpected = model.load_state_dict( saved_ckpt['model'], strict=False )
    assert missing == []
    # print('unexpected keys:', unexpected)
    
    official_ckpt = read_official_ckpt(official_ckpt_path)
    autoencoder.load_state_dict( official_ckpt["autoencoder"]  )
    text_encoder.load_state_dict( official_ckpt["text_encoder"]  )
    diffusion.load_state_dict( official_ckpt["diffusion"]  )
    
    if model.use_autoencoder_kl:
        for mode, input_type in zip(model.input_modalities, model.input_types):
            if input_type == "image":
                model.condition_nets[mode].autoencoder = deepcopy(autoencoder)
                model.condition_nets[mode].set = True

    return model, autoencoder, text_encoder, diffusion, config

def set_alpha_scale(model, alpha_scale):
    from ldm.modules.multimodal_attention import GatedCrossAttentionDense, GatedSelfAttentionDense
    from ldm.modules.diffusionmodules.multimodal_openaimodel import UNetModel
    alpha_scale_sp, alpha_scale_nsp, alpha_scale_image = alpha_scale
    for name, module in model.named_modules():
        if type(module) == GatedCrossAttentionDense or type(module) == GatedSelfAttentionDense:
            if '.sp_fuser' in name:
                module.scale = alpha_scale_sp
            elif '.nsp_fuser' in name:
                module.scale = alpha_scale_nsp
        elif type(module) == UNetModel:
            module.scales = [alpha_scale_image] * 4
            
def alpha_generator(length, config):
    """
    length is total timestpes needed for sampling.
    type should be a list containing three values which sum should be 1

    It means the percentage of three stages:
    alpha=scale stage
    linear deacy stage
    alpha=0 stage.

    For example if length=100, type=[0.8,0.1,0.1,_scale_]
    then the first 800 stpes, alpha will be _scale_, and then linearly decay to 0 in the next 100 steps,
    and the last 100 stpes are 0.
    """

    alpha_schedule_sp = config['alpha_type_sp']
    alpha_schedule_nsp = config['alpha_type_nsp']
    alpha_schedule_image = config['alpha_type_image']

    alphas_ = list()
    for alpha_schedule in [alpha_schedule_sp, alpha_schedule_nsp, alpha_schedule_image]:

        assert len(alpha_schedule)==4
        assert alpha_schedule[0] + alpha_schedule[1] + alpha_schedule[2] == 1

        stage0_length = int(alpha_schedule[0]*length)
        stage1_length = int(alpha_schedule[1]*length)
        stage2_length = length - stage0_length - stage1_length

        if stage1_length != 0:
            decay_alphas = alpha_schedule[3] * np.arange(start=0, stop=1, step=1/stage1_length)[::-1]
            decay_alphas = list(decay_alphas)
        else:
            decay_alphas = []

        alphas = [alpha_schedule[3]]*stage0_length + decay_alphas + [0]*stage2_length

        assert len(alphas) == length
        alphas_.append(alphas)

    return list(zip(*alphas_))

def preprocess(prompt="",
               sketch=None,
               depth=None,
               phrases=None,
               locations=None,
               keypoints=None,
               color=None,
               reference=None):
    batch = dict()
    null_conditions = []
    
    batch = torch.load('./images/dummy.pth', map_location='cpu')  # dummy var
    batch["caption"] = [prompt]

    if sketch is not None:
        sketch_tensor = get_sketch(sketch)
        selected_sketch = dict()
        selected_sketch["values"] = sketch_tensor.unsqueeze(0)
        selected_sketch["masks"] = torch.tensor([[1.]])
        batch["sketch"] = selected_sketch
    else:
        null_conditions.append("sketch")

    if depth is not None:
        depth_tensor = get_depth(depth)
        selected_depth = dict()
        selected_depth["values"] = depth_tensor.unsqueeze(0)
        selected_depth["masks"] = torch.tensor([[1.]])
        batch["depth"] = selected_depth
    else:
        null_conditions.append("depth")

    if locations is not None and phrases is not None:
        version = "openai/clip-vit-large-patch14"
        clip_model = CLIPModel.from_pretrained(version).cuda()
        clip_processor = CLIPProcessor.from_pretrained(version)
        boxes, masks, text_embeddings = get_box(locations, phrases, clip_model, clip_processor)
        selected_box = dict()
        selected_box["values"] = boxes.unsqueeze(0)
        selected_box["masks"] = masks.unsqueeze(0)
        selected_box["text_embeddings"] = text_embeddings.unsqueeze(0)
        batch["box"] = selected_box
    else:
        null_conditions.append("box")

    if keypoints is not None:
        points, masks = get_keypoint(keypoints)
        selected_keypoint = dict()
        selected_keypoint["values"] = points.unsqueeze(0)
        selected_keypoint["masks"] = masks.unsqueeze(0)
        batch["keypoint"] = selected_keypoint
    else:
        null_conditions.append("keypoint")

    if color is not None:
        selected_color_palette = dict()
        # color_palette = get_color_palette(color)  # for .png file
        # selected_color_palette["values"] = torch.tensor(color_palette, dtype=torch.float32).unsqueeze(0)
        selected_color_palette["values"] = torch.load(color).unsqueeze(0)
        selected_color_palette["masks"] = torch.tensor([[1.]])
        batch["color_palette"] = selected_color_palette
    else:
        null_conditions.append("color_palette")

    if reference is not None:
        version = "openai/clip-vit-large-patch14"
        clip_model = CLIPModel.from_pretrained(version).cuda()
        clip_processor = CLIPProcessor.from_pretrained(version)
        clip_features = get_clip_feature(reference, clip_model, clip_processor)
        selected_image_embedding = dict()
        selected_image_embedding["values"] = torch.tensor(clip_features).unsqueeze(0)
        selected_image_embedding["masks"] = torch.tensor([[1.]])
        batch["image_embedding"] = selected_image_embedding
        batch["image_embedding"]["image"] = tf.ToTensor()(tf.Resize((512,512))(Image.open(reference).convert('RGB'))).unsqueeze(0)
    else:
        batch["image_embedding"]["image"] = -torch.ones_like(batch["image"])
        null_conditions.append("image_embedding")

    return batch, null_conditions



@torch.no_grad()
def run(selected_batch_,
        config,
        model, 
        autoencoder, 
        text_encoder, 
        diffusion, 
        condition_null_generator_dict,
        idx,
        NULL_CONDITION,
        SAVE_NAME,
        seed):
    
    #### Starting noise fixed ####
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    starting_noise = torch.randn(1, 4, 64, 64).to(device)

    selected_batch = deepcopy(selected_batch_)
    uc_batch = deepcopy(selected_batch_)
    for mode in condition_null_generator_dict:
        if mode in NULL_CONDITION:
            condition_null_generator = condition_null_generator_dict[mode]
            condition_null_generator.prepare(selected_batch[mode])
            selected_batch[mode] = condition_null_generator.get_null_input(selected_batch[mode])
        else:
            if mode in ["sketch", "depth"]:
                continue
        condition_null_generator = condition_null_generator_dict[mode]
        condition_null_generator.prepare(uc_batch[mode])
        uc_batch[mode] = condition_null_generator.get_null_input(uc_batch[mode])

    selected_batch = batch_to_device(selected_batch, device)
    uc_batch = batch_to_device(uc_batch, device)
    
    torch.cuda.empty_cache()

    batch_here = config['batch_size']
    context = text_encoder.encode(selected_batch["caption"])
    # you can set negative prompts here
    # uc = text_encoder.encode(batch_here*["longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"])
    uc = text_encoder.encode(batch_here*[""])

    # plms sampling
    alpha_generator_func = partial(alpha_generator, config=config)
    sampler = PLMSSampler(diffusion, model, alpha_generator_func=alpha_generator_func, set_alpha_scale=set_alpha_scale)
    steps = 50
    shape = (batch_here, model.in_channels, model.image_size, model.image_size)

    input_dict = dict(x = starting_noise,
                      timesteps = None,
                      context = context,
                      inpainting_extra_input = None,
                      condition = selected_batch )

    uc_dict    = dict(context = uc,
                      condition = uc_batch )

    samples = sampler.sample(S=steps, shape=shape, input=input_dict, uc_dict=uc_dict, guidance_scale=config['guidance_scale'])
    pred_image = autoencoder.decode(samples)

    image_dict = [
        # {"tensors": selected_batch["image"], "n_in_row": 1, "pp_type": iutils.PP_RGB},
        {"tensors": draw_sketch_with_batch_to_tensor(selected_batch), "n_in_row": 1, "pp_type": iutils.PP_SEGM},
        {"tensors": draw_depth_with_batch_to_tensor(selected_batch), "n_in_row": 1, "pp_type": iutils.PP_SEGM},
        {"tensors": draw_boxes_with_batch_to_tensor(selected_batch), "n_in_row": 1, "pp_type": iutils.PP_SEGM},
        {"tensors": draw_keypoints_with_batch_to_tensor(selected_batch), "n_in_row": 1, "pp_type": iutils.PP_SEGM},
        {"tensors": draw_image_embedding_with_batch_to_tensor(selected_batch), "n_in_row": 1, "pp_type": iutils.PP_RGB},  
        {"tensors": draw_color_palettes_with_batch_to_tensor(selected_batch), "n_in_row": 1, "pp_type": iutils.PP_SEGM},  # range 0~1
        {"tensors": pred_image, "n_in_row": 1, "pp_type": iutils.PP_RGB},
    ]
    os.makedirs(os.path.join("inference", SAVE_NAME), exist_ok=True)
    iutils.save_images_from_dict(
        image_dict, dir_path=os.path.join("inference", SAVE_NAME), file_name="sampled_{:4d}".format(idx),
        n_instance=config['batch_size'], is_save=True, return_images=False
    )
    save_path = os.path.join("inference", SAVE_NAME, 'captions.txt')
    with open(save_path, "a") as f:
        f.write( 'idx ' + str(idx) + ':\n' )
        for cap in selected_batch['caption']:
            f.write( cap + '\n' )
        f.write( '\n' )
    print("Save images and its corresponding captions.. done")
    
    return pred_image.detach().cpu()



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, default="./diffblender_checkpoints/checkpoint_latest.pth", help="pretrained checkpoint path")
    parser.add_argument("--official_ckpt_path", type=str,  default="/path/to/sd-v1-4.ckpt", help="official SD path")
    parser.add_argument("--save_name", type=str, default="SAVE_NAME", help="")

    parser.add_argument("--alpha_type_sp", nargs='+', type=float, default=[0.3, 0.0, 0.7, 1.0], help="alpha scheduling type for spatial cond.")
    parser.add_argument("--alpha_type_nsp", nargs='+', type=float, default=[0.3, 0.0, 0.7, 1.0], help="alpha scheduling type for non-spatial cond.")
    parser.add_argument("--alpha_type_image", nargs='+', type=float, default=[1.0, 0.0, 0.0, 0.7], help="alpha scheduling type for image-form cond.")
    parser.add_argument("--guidance_scale", type=float, default=5.0, help="classifier-free guidance scale")
 
    args = parser.parse_args()
    

    model, autoencoder, text_encoder, diffusion, config = load_ckpt(ckpt_path=args.ckpt_path, official_ckpt_path=args.official_ckpt_path)
    condition_null_generator_dict = dict()
    for mode in config['condition_null_generator']['input_modalities']:
        condition_null_generator_dict[mode] = instantiate_from_config(config['condition_null_generator'][mode])

    # replace config
    config['batch_size'] = 1
    config['alpha_type_sp'] = args.alpha_type_sp 
    config['alpha_type_nsp'] = args.alpha_type_nsp
    config['alpha_type_image'] = args.alpha_type_image
    config['guidance_scale'] = args.guidance_scale

    kwargs_dict = dict(
        config=config,
        model=model,            
        autoencoder=autoencoder, 
        text_encoder=text_encoder, 
        diffusion=diffusion, 
        condition_null_generator_dict=condition_null_generator_dict,
        SAVE_NAME=args.save_name,
    )


    meta_list = [  # change
            
        dict(
            prompt = "jeep",
            sketch = "images/jeep_sketch.png",
            depth = "images/jeep_depth.png",
            color = "images/color1.pth",  # can also use image file via get_color_palette func
            reference = "images/fire.png",
        ),

        dict(
            prompt = "swimming rabbits",
            phrases = ["rabbit", "rabbit", "rabbit"],
            locations = [ [0.3500, 0.5000, 1.0000, 0.9500], [0.2000, 0.2500, 0.6000, 0.5500], [0.0500, 0.0500, 0.4000, 0.3000] ],
            color = "images/color2.pth",
        ),

        dict(
            prompt = "jumping astronaut",
            sketch = "images/partial_sketch.png",
            phrases = ["astronaut"],
            locations = [[0.1158, 0.1053, 0.5140, 0.6111]],
            keypoints = [
                [ [0.2767, 0.2025],
                  [0.2617, 0.1875],
                  [0.2917, 0.1875],
                  [0.0000, 0.0000],
                  [0.3117, 0.1800],
                  [0.2192, 0.2375],
                  [0.3392, 0.2425],
                  [0.1942, 0.2850],
                  [0.3967, 0.3075],
                  [0.1667, 0.3475],
                  [0.4142, 0.3675],
                  [0.2592, 0.3775],
                  [0.3242, 0.3700],
                  [0.2717, 0.4425],
                  [0.3992, 0.4375],
                  [0.2367, 0.5550],
                  [0.4067, 0.5225], ]
            ],
            reference = "images/nature.png",
        ),
            
    ]

    seed_list = [40, 10, 20]  # change

    for idx, (meta, seed) in enumerate(zip(meta_list, seed_list)):
        batch, null_conditions = preprocess(**meta)
        kwargs_dict['idx'] = idx
        kwargs_dict['seed'] = seed 
        kwargs_dict['NULL_CONDITION'] = null_conditions
        pred_image = run(batch, **kwargs_dict)
