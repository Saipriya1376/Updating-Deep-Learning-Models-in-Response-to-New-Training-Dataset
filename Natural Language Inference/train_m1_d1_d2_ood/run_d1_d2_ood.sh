# !/bin/bash
echo $(pwd)

python3 train_d1_d2.py --m1_model_path MNLI_MODEL_M1/BestModel \
                        --dataset_name multinli_1.0 \
                        --output_model_directory MNLI_MODEL_d1_OOD_d2_on_m1 \
                        --output_tokenizer_directory MNLI_MODEL_d1_OOD_d2_on_m1 \
                        --train_sample_percent 10 \
                        --id_ood ./MSP_corrected.txt

python3 test.py --input_model_path MNLI_MODEL_d1_OOD_d2_on_m1/BestModel \
                --mnli_matched_path ./multinli_1.0/multinli_1.0_dev_matched.txt \
                --mnli_mismatched_path ./multinli_1.0/multinli_1.0_dev_mismatched.txt \
                --hans_file1_path ./HANS/hans1.txt \
                --hans_file2_path ./HANS/hans2.txt
