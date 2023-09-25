# How to Pretrain RoBERTa

Use the following command to pretrain RoBERTa:

```shell
python RoBERTa.py RoBERTa_config.json
```

## Content

file `RoBERTa.py` is our code for (pre)training a RoBERTa model. 

file `RoBERTa_config.json` is its config file. Note: do now write any comment in this file. 

this will create a folder to save trained model, and logging file.

The logging file will record the configuration, and the training steps.

for training dataset and tokenizer, please find them on huggingface.

## Parameters in `RoBERTa_config.json` 

| Parameter                       | Description                                           |
| ------------------------------- | ----------------------------------------------------- |
| "seed": 0                       | The random seed number. |
| "device": "cuda:1"              | The GPU device number. Note: This configuration uses Huggingface's Trainer, which by default uses **all** GPUs for training. We used customTrainingArguments to specify a certain GPU, for a friendly allocation on a public scientific server shared with colleagues. If you need to use all GPUs like the default setting, replace `class customTrainingArguments` with `training_args = TrainingArguments`. |
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

have fun with coding :-)
