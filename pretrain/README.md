# How to Pretrain RoBERTa

Use the following command to pretrain RoBERTa:

```shell
python RoBERTa.py RoBERTa_config.json
```

## Parameters in 'RoBERTa_config.json' 

| Parameter                       | Description                                           |
| ------------------------------- | ----------------------------------------------------- |
| "seed": 0                       | The random seed number. |
| "device": "cuda:1"              | The GPU device number. Note: This configuration uses Huggingface's Trainer, which by default uses **all** GPUs for training. We modified it to use only one specific GPU. If you need to use all GPUs like the default setting, modify `class customTrainingArguments` or `training_args = TrainingArguments`. |
| "num_epochs": 5                 | The number of epochs for training. |
| "max_seq_length": 512           | The maximum sequence length. This is a parameter for `RobertaTokenizer.from_pretrained`. |
| "per_device_train_batch_size" : 32 | The batch size per device. This is a parameter for `training_args`. |
| "logging_per_n_step" : 1000     | The step interval for logging. |
| "path_tokenizer": "./RoBERTa/RoBERTa_Tokenizer" | The path to the tokenizer. |
| "path_data_train": "./babylm_data/babylm_100M/all100.train" | The path to the training data. |
| "path_data_dev": "./babylm_data/babylm_dev/all.dev" | The path to the development data (this might be unused). |
| "model_vocab_size": 52000       | The vocabulary size for the model. This is a parameter for Roberta. |
| "model_max_position_embeddings": 512 | The maximum position embeddings. This is a parameter for Roberta. |
| "model_num_attention_heads": 12 | The number of attention heads. This is a parameter for Roberta. |
| "model_num_hidden_layers": 12   | The number of hidden layers. This is a parameter for Roberta. |
| "model_type_vocab_size": 1      | The type vocabulary size. This is a parameter for Roberta. |
| "path_folder_prefix": ""        | The prefix for the output folder. |
| "mlm_probability": 0.15         | The masked language model probability. |
