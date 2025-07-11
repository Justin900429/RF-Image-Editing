# RF Solver Editting with Diffusers

This is a lightweight implementation of [RF-Solver-Edit](https://github.com/wangjiangshan0725/RF-Solver-Edit/tree/main) using the diffusers library. The original paper, *Taming Rectified Flow for Inversion and Editing*, can be found [here](https://arxiv.org/abs/2411.04746).

## Installation

The environment can be set up with the following command:

```bash
uv sync
```

To execute the file or using the environment, two options are available:

```bash
# using the uv command
uv run main.py

# sourcing the environment
source .venv/bin/activate
python main.py
```

## Usage

Please refer to the notebook at `notebook/demo.ipynb`.

```python
import torch
import diffusers
from pipeline import RFImageEditingFluxPipeline

diffusers.utils.logging.set_verbosity_error()

pipe = RFImageEditingFluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16, 
)
pipe.add_processor(after_layer=20)

source_img = "assets/source/cartoon.jpg"
source_prompt = ""
target_prompt = "a cartoon style Herry Potter raising his left hand"

image = pipe(
    source_img,
    source_prompt,
    target_prompt,
    inject_step=2,
    guidance_scale=2,
    num_inference_steps=25,
    max_sequence_length=256,
    with_second_order=True,
).images[0]
image.save("edited.jpg")
```

| Source                                     | Edit Result                                |
| ------------------------------------------ | ------------------------------------------ |
| ![source_img](./assets/source/cartoon.jpg) | ![result_img](./assets/result/cartoon.jpg) |

## Citation

If you find this work helpful for your research or applications, please consider citing the original paper:

```bibtex
@inproceedings{wang2025taming,
  title={Taming Rectified Flow for Inversion and Editing},
  author={Wang, Jiangshan and Pu, Junfu and Qi, Zhongang and Guo, Jiayi and Ma, Yue and Huang, Nisha and Chen, Yuxin and Li, Xiu and Shan, Ying},
  booktitle={ICML},
  year={2025}
}
```
