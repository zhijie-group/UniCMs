<div align="center">
<br>
<img src="docs/title.png" width="166"> <!-- Replace with your logo -->
<h3>Show-o Turbo: Towards Accelerated Unified Multimodal Understanding and Generation</h3>

[Anonymous CVPR submission]

[![ArXiv](https://img.shields.io/badge/ArXiv-PaperID12251-<COLOR>.svg)](https://arxiv.org/abs/your_paper_id)  [![Demo](https://img.shields.io/badge/Demo-ComingSoon-<COLOR>.svg)](https://your_demo_link) [![Discord](https://img.shields.io/badge/Discord-join-blueviolet?logo=discord&amp)](https://your_discord_link) 

</div>

## News
* **[2024-11-29]** We release a [256-resolution version of the weights](https://huggingface.co/SJTU-Deng-Lab/Show-o-Turbo-256) for Show-o Turbo on Hugging Face.

  

## What's New about Show-o Turbo?

Show-o Turbo builds upon Show-o to address its inefficiency issues in both image and text generation. While Show-o relies on progressive denoising for images and autoregressive decoding for text, Show-o Turbo introduces a unified denoising perspective for both modalities, leading to significantly faster generation speeds.  Show-o Turbo achieves this through several key innovations:

<p align="center">
<img src="docs/trajectory.png" style="max-width: 100%;"> <!-- Charts and graphs showcasing results -->
</p>


* **Unified Denoising:**  Show-o Turbo utilizes parallel text decoding techniques (Jacobi decoding) to reframe text generation as a denoising process, analogous to image generation. This enables a unified view of both modalities as denoising trajectories.
* **Consistency Distillation:**  Show-o Turbo employs consistency distillation, a technique inspired by diffusion model acceleration, to shorten these multimodal denoising trajectories. This allows the model to generate meaningful content faster.
* **Trajectory Segmentation and Curriculum Learning:** To enhance convergence, Show-o Turbo uses a staged training approach with decreasing trajectory segments and curriculum learning.
* **Top-k Sampling:**  Show-o Turbo utilizes top-k sampling during inference to improve sample quality, especially with fewer sampling steps.

## Results

Show-o Turbo shows significant speedups in both text-to-image and image-to-text generation, while maintaining comparable performance to Show-o.

* In text-to-image generation, it achieves performance close to that of Show-o at 8-step sampling at 4-step sampling, and surpasses Show-o at 4-step sampling at 2-step sampling.
  <p align="center">
  <img src="docs/t2i_result.png" width="777"> <!-- Charts and graphs showcasing results -->
  </p>

* In multimodal understanding tasks, it is about 1.5 times faster without much performance loss.
  <p align="center">
  <img src="docs/mmu_result.png" width="777"> <!-- Charts and graphs showcasing results -->
  </p>

## Getting Started

First, set up the environment:
```bash
pip3 install -r requirements.txt
```

### Inference

**Multimodal Understanding:**

```bash
python3 inference_mmu.py config=configs/showo_turbo_mmu.yaml \
# ... Add your MMU inference options here
```

<p align="center">
<img src="docs/mmu.png" style="max-width: 100%;"> <!-- Example output of MMU inference -->
</p>


**Text-to-Image Generation:**

```bash
python3 inference_t2i.py config=configs/showo_turbo_t2i.yaml \
# ... Add your T2I inference options here 
```

<p align="center">
<img src="docs/t2i.png" style="max-width: 100%;"> <!-- Example output of T2I inference -->
</p>



## Training pipeline

**(Coming Soon)** Details about the training process, including data preparation, scripts, and configuration options will be provided here upon release. Example command:

```bash
accelerate launch --config_file path/to/your/accelerate_config --main_process_port=8888 training/train_showo_turbo.py config=configs/showo_turbo_training.yaml
```


## TODO

- [X] Release the inference and training code.
- [X] Release the model weights.
- [ ] Conduct further experiments with larger model sizes and datasets.

## Contributing

We welcome contributions to Show-o Turbo!  If you have ideas for new features or improvements, please open an issue or submit a pull request.


## Citation

**(Coming Soon)** Citation information will be provided here upon publication.


## Acknowledgments

We would like to thank the authors of Show-o and the developers of the libraries and frameworks upon which Show-o Turbo is built, including  open-muse, Phi-1.5,  maskgit, taming-transformers, transformers, accelerate, diffusers. Thanks to all the authors for their great work.

