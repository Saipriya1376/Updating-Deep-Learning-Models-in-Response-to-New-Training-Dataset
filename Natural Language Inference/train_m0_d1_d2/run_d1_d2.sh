# !/bin/bash
echo $(pwd)
echo $(pwd)
echo $(pwd)

python3 train_d1_d2.py --dataset_name multinli_1.0 --output_model_directory MNLI_MODEL_d1_union_d2_on_m0 --output_tokenizer_directory MNLI_MODEL_d1_union_d2_on_m0

python3 test.py --input_model_path MNLI_MODEL_d1_union_d2_on_m0/BestModel \
                --mnli_matched_path ./multinli_1.0/multinli_1.0_dev_matched.txt \
                --mnli_mismatched_path ./multinli_1.0/multinli_1.0_dev_mismatched.txt \
                --hans_file1_path ./HANS/hans1.txt \
                --hans_file2_path ./HANS/hans2.txt
