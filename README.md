### Data

- Run `preprocess_data.py` to save the json splits into a pkl format. Make sure to have the source github repo folder (Article-Bias-Prediction) in the root directory.
- Run `load_data_and_explore.py` to load the data and explore it in a notebook.

### Finetuning a Classifier

- Run `finetune_base_auto.py` to finetune the base classifier with auto_model.
- Run `finetune_base_biasclassifier.py` to finetune the base bias_classifier.

### Finetuning a Classifier atop a Triplet Model (with or without LoRA)

- Run `pretrain_triplet_wo_lora.py` to train the triplet model without LoRA.
- Run `finetune_no_lora_triplet_auto.py` to finetune the base auto classifier atop the non-lora triplet model.
- Run `finetune_no_lora_triplet_biasclassifier.py` to finetune the base biasclassifier atop the non-lora triplet model.
- Run `pretrain_triplet_with_lora.py` to train the triplet model with LoRA.
- Run `finetune_lora_triplet_auto.py` to finetune the base auto classifier atop the lora triplet model.

### Evaluation

- Run `evaluate_finetuned_biasclassifier_model.py` to evaluate biasclassifier finetuned models.
- Run `evaluate_finetuned_auto_model.py` to evaluate auto classifier finetuned models.