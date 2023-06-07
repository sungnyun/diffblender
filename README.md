# DiffBlender: Scalable and Composable Multimodal Text-to-Image Diffusion Models
[[Project Page](https://sungnyun.github.io/diffblender/)] [[ArXiv](https://arxiv.org/pdf/2305.15194.pdf)] [[BibTeX](#bibtex)] 


- **DiffBlender** successfully synthesizes complex combinations of input modalities. It enables flexible manipulation of conditions, providing the customized generation aligned with user preferences.
- We designed its structure to intuitively extend to additional modalities while achieving a low training cost through a partial update of hypernetworks.

<p align="center">
<img width="1369" alt="teaser" src="./assets/fig1.png">
</p>

## To-Dos

- [x] Project page is open: [link](https://sungnyun.github.io/diffblender/)
- [x] DiffBlender model: code & checkpoint
- [x] Release inference code
- [ ] Release training code & pipeline
- [ ] Gradio UI

## Requirements
Install the necessary packages with:
```sh
$ pip install -r requirements.txt
```

Download DiffBlender model checkpoint from this [link](https://www.dropbox.com/s/vnjribkwx3xcwm6/checkpoint_latest.pth?dl=0), and place it under `./diffblender_checkpoints/`.    
Also, prepare the SD model from this [link](https://huggingface.co/CompVis/stable-diffusion-v-1-4-original) (we used CompVis/sd-v1-4.ckpt).

## Inference
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
