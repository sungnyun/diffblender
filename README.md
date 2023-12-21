# DiffBlender: Scalable and Composable Multimodal Text-to-Image Diffusion Models 🔥

<a href='https://sungnyun.github.io/diffblender/'><img src='https://img.shields.io/badge/Project-Page-yellow'></a>
<a href='https://arxiv.org/abs/2305.15194'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>  <a href=#bibtex><img src='https://img.shields.io/badge/Paper-BibTex-Green'></a> 
<a href='https://huggingface.co/sungnyun/diffblender'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-DiffBlender_Model-blue'></a>


- **DiffBlender** successfully synthesizes complex combinations of input modalities. It enables flexible manipulation of conditions, providing the customized generation aligned with user preferences. 
- We designed its structure to intuitively extend to additional modalities while achieving a low training cost through a partial update of hypernetworks. 

<p align="center">
<img width="1369" alt="teaser" src="./assets/fig1.png">
</p>

## 🗓️ TODOs

- [x] Project page is open: [link](https://sungnyun.github.io/diffblender/)
- [x] DiffBlender model: code & checkpoint
- [x] Release inference code
- [ ] Release training code & pipeline
- [ ] Gradio UI

## 🚀 Getting Started
Install the necessary packages with:
```sh
$ pip install -r requirements.txt
```

Download DiffBlender model checkpoint from this [Huggingface model](https://huggingface.co/sungnyun/diffblender), and place it under `./diffblender_checkpoints/`.    
Also, prepare the SD model from this [link](https://huggingface.co/CompVis/stable-diffusion-v-1-4-original) (we used CompVis/sd-v1-4.ckpt).

## ⚡️ Try Multimodal T2I Generation with DiffBlender
```sh
$ python inference.py --ckpt_path=./diffblender_checkpoints/{CKPT_NAME}.pth \
                      --official_ckpt_path=/path/to/sd-v1-4.ckpt \
                      --save_name={SAVE_NAME} 
```

Results will be saved under `./inference/{SAVE_NAME}/`, in the format as {conditions + generated image}.


 
## BibTeX
```
@article{kim2023diffblender,
  title={DiffBlender: Scalable and Composable Multimodal Text-to-Image Diffusion Models},
  author={Kim, Sungnyun and Lee, Junsoo and Hong, Kibeom and Kim, Daesik and Ahn, Namhyuk},
  journal={arXiv preprint arXiv:2305.15194},
  year={2023}
}
```
