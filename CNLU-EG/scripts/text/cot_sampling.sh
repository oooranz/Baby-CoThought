python main.py \
    --task text \
    --task_file_path text.txt \
    --task_start_index 70000 \
    --task_end_index 70001 \
    --naive_run \
    --prompt_sample cot \
    --n_generate_sample 2 \
    --temperature 1.0 \
    ${@}