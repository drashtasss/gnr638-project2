# GNR 638 Project 2 — Visual MCQ Solver

## Approach

We use **Qwen2-VL-7B-Instruct**, an open-source vision-language model, to read MCQ images and predict the correct option (A/B/C/D) directly. The model:

1. Reads the image (containing question + 4 options).
2. Receives a constrained prompt that asks for a single letter answer.
3. Outputs one token, which we parse to A/B/C/D.

No training is performed — we use the model zero-shot.

## File structure

```
.
├── setup.bash          # Creates conda env, installs deps, downloads weights (with internet)
├── inference.py        # Loads model + runs inference (no internet)
├── requirements.txt    # Pinned dependencies (reference only)
├── README.md           # This file
└── model_weights/      # Created by setup.bash, contains Qwen2-VL weights
```

## How to run

```bash
bash setup.bash
conda activate gnr_project_env
python inference.py --test_dir /absolute/path/to/test_dir
```

This produces `submission.csv` in the current working directory.

## Expected test directory structure

```
test_dir/
├── test.csv           # contains image_id column
├── submission.csv     # dummy file, ignored
└── images/
    ├── img_001.png
    ├── img_002.png
    └── ...
```

## Output format

```
image_id,answer
img_001.png,A
img_002.png,C
...
```

## Hardware

- Tested on: NVIDIA L40s (48 GB VRAM)
- Model: Qwen2-VL-7B-Instruct, fp16 (~15 GB VRAM)
- CUDA: 12.6 (uses 12.4 PyTorch wheels for compatibility)
