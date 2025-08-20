# 🧭 Intruduction
This repository provides a batch inference pipeline using Groma, a grounded multimodal large language model (MLLM) with strong region understanding and visual grounding capabilities, for Multimodal Reasoning Competition Track1 (VG-RS). Given a set of image-question pairs, the model outputs the corresponding bounding box coordinates through fine-grained referring and grounding.

---
## 📦 Environment Prepare
1. Clone the repository
2. Create the conda environment and install dependencies
```Shell
conda create -n groma python=3.9 -y
conda activate groma
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install --upgrade pip  # enable PEP 660 support
pip install -e .

cd mmcv
MMCV_WITH_OPS=1 pip install -e .
cd ..
```
3. Install falsh-attention for training
```
pip install ninja
pip install flash-attn --no-build-isolation
```

---
## 🗂 Directory Structure

```
project_root/  
├── pyproject.toml			# environment info  
├── model/					# Folder with groma model  
│   └── groma-7b-finetune  
├── images/					# Folder with images   
│   └── `*.jpg` / `*.png`  
├── VG-RS-question.json			# Input questions and image paths
├── VG-RS-refcoco-format.json			# Input after format  
├── format_question.py				# format input  
...  
```

---
## 🔧 Model and Processor Setup
### 🧠 Download Model with ModelScope
To play with Groma, please download the [model weights](https://huggingface.co/FoundationVision/groma-7b-finetune) from huggingface.

---
## 🧪 How to Run Inference
### ✍️ Prepare Input JSON
The file `VG-RS-question.json` should be a list of entries in this format:

```
[
  { 
    "image_path": "images\example.jpg", 
    "question": "What object is next to the red car?"
    },
  ...
]
```

then run the format script

```shell
python format_question.py
```


### 🚀 Run Script
inference for one image:
```shell
CUDA_VISIBLE_DEVICES=0 python -m groma.eval.run_groma \
    --model-name ./model/groma-7b-finetune \
    --image-file ./images/24_0_106_0000010_250128.jpg \
    --query "A turned-off TV" \
```
inference for all image:
```shell
CUDA_VISIBLE_DEVICES=0 python -m groma.eval.run_groma_batch \
    --model-name ./model/groma-7b-finetune \
    --anno_path ../VG-RS-refcoco-format.json \
    --img_path ../images \
    --output_path ./VG-RS-answers_groma.json
```

---
## 📤 Output Format
The result will be saved as a JSON file containing predicted bounding boxes for each input:

```
[   
  {
    "image_path": "images\example.jpg",     
    "question": "What object is next to the red car?",    
    "result": [[x1, y1], [x2, y2]]     
  },    
  ...     
]
```

> Note: Bounding boxes are in the format `[[x_min, y_min], [x_max, y_max]]`.

---
## 📝 Reference
If you use this code and our data, please cite:
```
@article{ma2024groma,
  title={Groma: Localized Visual Tokenization for Grounding Multimodal Large Language Models},
  author={Ma, Chuofan and Jiang, Yi and Wu, Jiannan and Yuan, Zehuan and Qi, Xiaojuan},
  journal={arXiv preprint arXiv:2404.13013},
  year={2024}
}
```

---
## 🔗 Acknowledgement

This codebase is partially based on [Groma](https://github.com/FoundationVision/Groma)  
Groma is built upon the awesome works 
[LLaVA](https://github.com/haotian-liu/LLaVA/) and 
[GPT4ROI](https://github.com/jshilong/GPT4RoI).

## 📜 LICENSE
This project is licensed under the Apache License 2.0 - 
see the [LICENSE](LICENSE) file for details.

## 💬 Contact

If you encounter any issues or have questions, feel free to open an issue on GitHub.
