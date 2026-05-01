"""
Visual MCQ Solver - Inference Script
GNR 638 Project 2

Usage:
    python inference.py --test_dir <absolute_path_to_test_dir>

Reads images from <test_dir>/images/ based on <test_dir>/test.csv
and writes submission.csv in the current working directory.
"""

import argparse
import os
import re
import sys
import traceback

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------
# Local path where setup.bash downloaded the model weights.
# This MUST exist before inference.py runs (no internet at inference time).
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model_weights')

# The prompt that constrains the model to output a single letter.
PROMPT = (
    "You are an expert in deep learning and machine learning. "
    "The image shows a multiple-choice question with options labeled (A), (B), (C), and (D). "
    "Carefully read the question and all four options. Identify the single correct answer. "
    "Output ONLY one character: A, B, C, or D. "
    "No explanation, no punctuation, no extra words. Just the letter."
)

# Default fallback if model output cannot be parsed
FALLBACK_ANSWER = 'A'


# -------------------------------------------------------------------
# HELPERS
# -------------------------------------------------------------------
def parse_answer(text: str) -> str:
    """Extract A/B/C/D from raw model output. Robust to extra whitespace/text."""
    if not text:
        return FALLBACK_ANSWER
    text = text.strip().upper()
    # Look for a standalone letter A/B/C/D
    match = re.search(r'\b([ABCD])\b', text)
    if match:
        return match.group(1)
    # Fallback: first character if it's a valid letter
    if text and text[0] in 'ABCD':
        return text[0]
    return FALLBACK_ANSWER


def load_model_and_processor(model_path: str):
    """Load the VLM and processor from local disk (no internet)."""
    print(f'[inference] Loading model from {model_path}', flush=True)
    if not os.path.isdir(model_path):
        raise FileNotFoundError(
            f'Model weights not found at {model_path}. '
            f'Did setup.bash run successfully?'
        )

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map='auto',
        local_files_only=True,
    )
    processor = AutoProcessor.from_pretrained(
        model_path,
        local_files_only=True,
    )
    model.eval()
    print('[inference] Model loaded.', flush=True)
    return model, processor


def predict_single(model, processor, image_path: str) -> str:
    """Run inference on one image, return one of A/B/C/D."""
    messages = [
        {
            'role': 'user',
            'content': [
                {'type': 'image', 'image': image_path},
                {'type': 'text', 'text': PROMPT},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors='pt',
    ).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=8,
            do_sample=False,
        )
    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    return parse_answer(output_text)


# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_dir', type=str, required=True,
                        help='Absolute path to the test directory.')
    args = parser.parse_args()

    test_dir = args.test_dir
    test_csv_path = os.path.join(test_dir, 'test.csv')
    images_dir = os.path.join(test_dir, 'images')

    # ---- Validate inputs ----
    if not os.path.isfile(test_csv_path):
        print(f'[inference] ERROR: test.csv not found at {test_csv_path}',
              file=sys.stderr)
        sys.exit(1)
    if not os.path.isdir(images_dir):
        print(f'[inference] ERROR: images dir not found at {images_dir}',
              file=sys.stderr)
        sys.exit(1)

    test_df = pd.read_csv(test_csv_path)
    print(f'[inference] Loaded test.csv with {len(test_df)} rows.', flush=True)
    print(f'[inference] Columns: {list(test_df.columns)}', flush=True)

    # ADJUST IF FORMAT DIFFERS:
    # The standard column name is "image_id". If the actual test.csv uses
    # a different column (e.g., "filename" or "id"), change it here.
    if 'image_id' not in test_df.columns:
        # Try common alternatives
        for alt in ['filename', 'id', 'image', 'img_id']:
            if alt in test_df.columns:
                test_df = test_df.rename(columns={alt: 'image_id'})
                print(f'[inference] Renamed column {alt!r} -> image_id', flush=True)
                break
        else:
            # Fall back to first column
            first_col = test_df.columns[0]
            test_df = test_df.rename(columns={first_col: 'image_id'})
            print(f'[inference] Using first column {first_col!r} as image_id',
                  flush=True)

    # ---- Load model ----
    try:
        model, processor = load_model_and_processor(MODEL_PATH)
    except Exception as e:
        print(f'[inference] FATAL: failed to load model: {e}', file=sys.stderr)
        traceback.print_exc()
        # Write a fallback submission so the grader still gets a file
        fallback = pd.DataFrame({
            'image_id': test_df['image_id'],
            'answer': [FALLBACK_ANSWER] * len(test_df),
        })
        fallback.to_csv('submission.csv', index=False)
        sys.exit(1)

    # ---- Inference loop ----
    predictions = []
    for image_id in tqdm(test_df['image_id'], desc='Predicting'):
        image_path = os.path.join(images_dir, str(image_id))
        try:
            answer = predict_single(model, processor, image_path)
        except Exception as e:
            # Don't crash on one bad image — log and use fallback
            print(f'[inference] WARN: failed on {image_id}: {e}', file=sys.stderr)
            answer = FALLBACK_ANSWER
        predictions.append(answer)

    # ---- Write submission ----
    submission = pd.DataFrame({
        'image_id': test_df['image_id'],
        'answer': predictions,
    })
    output_path = os.path.join(os.getcwd(), 'submission.csv')
    submission.to_csv(output_path, index=False)
    print(f'[inference] Wrote {output_path} with {len(submission)} rows.',
          flush=True)


if __name__ == '__main__':
    main()
