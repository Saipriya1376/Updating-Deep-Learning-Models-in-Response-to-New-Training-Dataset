# !/bin/bash
echo $(pwd)

python3 train_OOD_corrected_only_d2.py --m1_model_path QQP_MODEL/BestModel \
			--output_model_directory QQP_MODELOOD_d2_on_m1 \
			--output_tokenizer_directory QQP_MODEL_OOD_d2_on_m1 \
			--train_sample_percent 20 \
	                --id_ood ./MSP_train_ood.txt

python3 test.py --input_model_path QQP_MODELOOD_d2_on_m1/BestModel 
