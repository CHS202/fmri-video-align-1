#!/bin/bash
set -e

# guarded: only deactivate if a venv is currently active (the `deactivate`
# function only exists once some venv's activate script has defined it --
# calling it before that, e.g. on a fresh shell, is a no-op here instead of
# an error)
type deactivate &>/dev/null && deactivate

# source ~/llava-env/bin/activate
# python eval_vlm_baseline.py --model llava --task design \
#     --annotation_path video_id_rt.csv \
#     --splits 1 2 3 4 --runs 1 2 3 4 5 6 7 \
#     --num_frames 16

# python eval_vlm_baseline.py --model llava --task annot-reg \
#     --annotation_path video_id_rt_annot.csv \
#     --splits 4 --runs 6 7 \
#     --num_frames 16

# deactivate

source ~/qwen-env/bin/activate
# export GEMINI_API_KEY=AQ.Ab8RN6LbEe4z9ZkxfynTHplORQA85W3qaKmwWL8PHgaqFlqwHg # CHI27
# export GEMINI_API_KEY=AQ.Ab8RN6KP_alu78_1OLl0c_TRYKvQ0DaRjBP8v3mNf9KZwAZsog # physionet-data
# export GEMINI_API_KEY=AQ.Ab8RN6Ieu_R_Jsw06LfSuMoA2TJbp6k3PV5vZ2Udzq-28JR5Rg # free
export GEMINI_API_KEY=AQ.Ab8RN6L64uidz5OdhaiRox4F7jLdOCebcTPmJOl_XC9-iLktdA # default-gemini-project
# python eval_vlm_baseline.py --model qwen --task design \
#     --annotation_path video_id_rt.csv \
#     --splits 1 2 3 4 --runs 1 2 3 4 5 6 7 \
#     --num_frames 16
# python eval_vlm_baseline.py --model qwen --task annot-reg \
#     --annotation_path video_id_rt_annot.csv \
#     --splits 1 2 3 4 --runs 1 2 3 4 5 6 7 \
#     --num_frames 16

# python eval_vlm_baseline.py --model gemini --task describe \
#     --annotation_path video_id_rt_full.csv \
#     --num_frames 32
# python eval_vlm_baseline.py --model gemini --task design \
#     --annotation_path video_id_rt_full.csv \
#     --num_frames 32
# python eval_vlm_baseline.py --model gemini --task annot-reg \
#     --annotation_path video_id_rt_full_annot.csv \
#     --num_frames 32
python eval_vlm_baseline.py --model gemini --task annot-reg --temperature 2 --top_p 1 \
    --annotation_path video_id_rt_full_annot.csv \
    --num_frames 32
deactivate