# DiffBlender: Scalable and Composable Multimodal Text-to-Image Diffusion Models
[[ArXiv](https://arxiv.org/pdf/2305.15194.pdf)] [[BibTeX](#bibtex)] 


- **DiffBlender** successfully synthesizes complex combinations of input modalities. It enables flexible manipulation of conditions, providing the customized generation aligned with user preferences.
- We designed its structure to intuitively extend to additional modalities while achieving a low training cost through a partial update of hypernetworks.

<p align="center">
<img width="1369" alt="teaser" src="https://github.com/sungnyun/diffblender/assets/46050900/6380e3dd-c075-4ba1-ba75-0f66216d5f2d">
</p>

## To-Dos

- [ ] Release model checkpoint
- [ ] Release test code
- [ ] Release training code
- [ ] Release Gradio UI



## Model Architecture
<p align="center">
<img width="1468" alt="architecture" src="https://github.com/sungnyun/diffblender/assets/46050900/368ca0c5-fd82-467c-91b9-dd4e918c24d7">
</p>

## Multimodal Text-to-Image Generation

### Versatile applications of DiffBlender.
<p align="center">
<img width="1000" alt="teaser2" src="https://github.com/sungnyun/diffblender/assets/46050900/76dbccd7-ba48-4212-b553-1d6ce4d10281">
</p>
&nbsp;

### Reference-guided and semantic-preserved generation.
<p align="center">
<img width="1464" alt="reference" src="https://github.com/sungnyun/diffblender/assets/46050900/5468ed9d-cbbc-4730-aa9d-37fc81b64854">
</p>
&nbsp;

<p align="center">
<img width="1464" alt="reference2" src="https://github.com/sungnyun/diffblender/assets/46050900/1113b4d8-ffba-4f9c-a7d1-94029efb7aaf">
</p>
&nbsp;

### Object reconfiguration.
<p align="center">
<img width="1475" alt="reconfiguration" src="https://github.com/sungnyun/diffblender/assets/46050900/1dbfa8fc-0244-4090-8046-8575fc7489c6">
</p>
<p align="center">
<img width="1482" alt="reconfiguration2" src="https://github.com/sungnyun/diffblender/assets/46050900/82f05fd8-1275-4617-b55f-1f5e645fa54f">
</p>
&nbsp;

### Mode-specific guidance.
<p align="center">
<img width="1438" alt="mode_guidance" src="https://github.com/sungnyun/diffblender/assets/46050900/9827e5ba-d0c6-48e2-8e6a-81f378fff971">
</p>
&nbsp;

### Interpolating non-spatial conditions.
<p align="center">
<img width="1401" alt="interpolation" src="https://github.com/sungnyun/diffblender/assets/46050900/639ed49a-e5b4-4214-aad1-e036425275d7">
</p>
&nbsp;

### Manipulating spatial conditions.
<p align="center">
<img width="895" alt="manipulating" src="https://github.com/sungnyun/diffblender/assets/46050900/321f1022-4e8b-4811-954f-8eee91da0057">
</p>
&nbsp;

### Comparison with baselines.
<p align="center">
<img width="1504" alt="comparison" src="https://github.com/sungnyun/diffblender/assets/46050900/fa93e249-022b-4e35-98bc-97dcf182fc8a">
</p>
&nbsp;



## BibTeX
```
@article{kim2023diffblender,
  title={DiffBlender: Scalable and Composable Multimodal Text-to-Image Diffusion Models},
  author={Kim, Sungnyun and Lee, Junsoo and Hong, Kibeom and Kim, Daesik and Ahn, Namhyuk},
  journal={arXiv preprint arXiv:2305.15194},
  year={2023}
}
```
