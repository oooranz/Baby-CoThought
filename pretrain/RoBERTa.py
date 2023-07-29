from pathlib import Path
from tokenizers import ByteLevelBPETokenizer
from transformers import RobertaConfig
from transformers import RobertaTokenizer
from transformers import RobertaForMaskedLM
import torch
from transformers import LineByLineTextDataset
import sys
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
import os
import gc
from transformers import TrainerCallback
import logging
import json
import time



##########################################

# Utility Functions

##########################################

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))






if __name__ == "__main__":
    path_config_json = sys.argv[1]
    # load config_json
    with open(path_config_json, 'r') as f:
        config_json = json.load(f)

    ##########################################

    # Load Config

    ##########################################

    # random seed
    if config_json["seed"] != "undefine":
        seed = config_json["seed"]
    else:
        seed = 42
    
    torch.manual_seed(seed)
    
    # file path
    output_path_suffix = time.strftime("%m_%d_%H_%M_%S", time.localtime())
    output_path = f"{config_json['path_folder_prefix']}RoBERTa-{output_path_suffix}"
    os.makedirs(output_path)

    path_output_log     = f"{output_path}/log.txt"
    path_output_model   = f"{output_path}/RoBERTa-{output_path_suffix}"
    # path_output_linear  = f"{output_path}/linear-{output_path_suffix}"

    # device, Note: here I used Trainer. Thus NO need to model.to(device), @see: customTrainingArguments
    if config_json["device"] == "undefine":
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') 
    else:
        device = torch.device(config_json["device"])

    

    # Define Logging
    logging.basicConfig(level=logging.DEBUG)
    file_formatter  = logging.Formatter(fmt="%(asctime)s   %(message)s",
                                        datefmt="%m/%d/%Y %H:%M:%S", )
    file_handler    = logging.FileHandler(path_output_log)
    file_handler.setFormatter(file_formatter)
    logging.root.addHandler(file_handler)

    logging.debug(f"config seed actural                 [{seed}]")
    logging.debug(f"config seed defined                 [{config_json['seed']}]")
    logging.debug(f"config device                       [{config_json['device']}]")
    logging.debug(f"config num_epochs                   [{config_json['num_epochs']}]")
    logging.debug(f"config max_seq_length               [{config_json['max_seq_length']}]")
    logging.debug(f"config per_device_train_batch_size  [{config_json['per_device_train_batch_size']}]")
    logging.debug(f"config logging_per_n_step           [{config_json['logging_per_n_step']}]")
    logging.debug(f"config path_tokenizer               [{config_json['path_tokenizer']}]")
    logging.debug(f"config path_data_train              [{config_json['path_data_train']}]")
    logging.debug(f"config path_data_dev                [{config_json['path_data_dev']}]")
    logging.debug(f"config path_folder_prefix           [{config_json['path_folder_prefix']}]")
    logging.debug(f"config mlm_probability              [{config_json['mlm_probability']}]")



    # tokenizer
    tokenizer = RobertaTokenizer.from_pretrained(config_json["path_tokenizer"], 
                                                max_length=config_json["max_seq_length"])

    
    # model
    RoBERTa_config = RobertaConfig(
        vocab_size              = config_json["model_vocab_size"]               , # 52_000
        max_position_embeddings = config_json["model_max_position_embeddings"]  , # 512,
        num_attention_heads     = config_json["model_num_attention_heads"]      , # 12,
        num_hidden_layers       = config_json["model_num_hidden_layers"]        , # 12,
        type_vocab_size         = config_json["model_type_vocab_size"]          , # 1,
    )
    model = RobertaForMaskedLM(config=RoBERTa_config).to(device)

    logging.debug(f"config model_vocab_size                 [{config_json['model_vocab_size']}]")
    logging.debug(f"config model_max_position_embeddings    [{config_json['model_max_position_embeddings']}]")
    logging.debug(f"config model_num_attention_heads        [{config_json['model_num_attention_heads']}]")
    logging.debug(f"config model_num_hidden_layers          [{config_json['model_num_hidden_layers']}]")
    logging.debug(f"config model_type_vocab_size            [{config_json['model_type_vocab_size']}]")
    logging.debug(f"size of parameters                      [{model.num_parameters()}]")

    ##########################################

    logging.debug(f"loading dataset train ...")
    # Dataset

    ##########################################
    logging.debug(f"loading dataset train ...")

    dataset_train = LineByLineTextDataset(
        tokenizer   = tokenizer,
        #  file_path='./babylm_data_test/babylm_10M/aochildes.train',
        # file_path='./babylm_data/babylm_10M/all10.train',
        file_path   = config_json['path_data_train'],
        block_size  = 128,
    )
    logging.debug(f"loading dataset train finished!")

    logging.debug(f"loading dev dataset ...")
    dataset_dev = LineByLineTextDataset(
        tokenizer   = tokenizer,
        #  file_path='../babylm_data_test/babylm_10M/aochildes.train',
        # file_path   = './babylm_data/babylm_dev/all.dev',
        file_path   = config_json['path_data_dev'],
        block_size  = 128,
    )
    logging.debug(f"loading dev dataset finished!")

    logging.debug(f"loading data_collator ...")
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=config_json["mlm_probability"]
    )
    logging.debug(f"loading data_collator finished!")


    ##########################################
    # Trainer
    # Note 1: here I used Trainer
    #         originally it will use ALL GPUs, but it is bad
    #         the following class is almost the only elegent way to use a specific GPU
    #         If you want to use all GPU, you can just call training_args = TrainingArguments()
    # Note 2: I also used custom callback function, to make loggings
    #         originally function doesn't work at all :-(
    ##########################################

    class customTrainingArguments(TrainingArguments):
        def __init__(self,*args, **kwargs):
            super(customTrainingArguments, self).__init__(*args, **kwargs)
        @property
        def device(self) -> "torch.device":
            return device
        @property
        def n_gpu(self):
            self._n_gpu = 1
            return self._n_gpu



    training_args = customTrainingArguments(
        output_dir                  = output_path,
        num_train_epochs            = config_json["num_epochs"],
        seed                        = seed,

        overwrite_output_dir        = True,
        # evaluation_strategy         = "epoch",
        per_device_train_batch_size = config_json["per_device_train_batch_size"],
        save_steps                  = config_json["logging_per_n_step"],
        save_total_limit            = 2,
        #  logging_dir              = './log',  # useless?
        #  log_level                = 'warning',
        logging_strategy            = "steps", # "epoch"
        #  logging_steps            = 1,
        # logging_strategy            = "epoch",
        logging_first_step          = True,
        #  no_cuda                  = True,
    )


    # rewrite the logging function fall trainer
    class LogCallback(TrainerCallback):
        def on_train_begin( self, args, state, control, **kwargs ):
            logging.debug(f"train_begin")
                
        def on_train_end( self, args, state, control, **kwargs ):
            for key, value in state.log_history[-1].items():
                logging.debug(f"train_end__  Epoch [{state.epoch}] [{key}] : [{value}]")
            logging.debug(f"-----------")

        def on_epoch_begin( self, args, state, control, **kwargs ):
            logging.debug(f"-----------")
            logging.debug(f"epoch_begin  Epoch [{state.epoch}]")

        def on_epoch_end( self, args, state, control, **kwargs ):
            for key, value in state.log_history[-1].items():
                logging.debug(f"_epoch_end_  Epoch [{state.epoch}] [{key}] : [{value}]")

        def on_evaluate(self, args, state, control, **kwargs):
            for key, value in state.log_history[-1].items():
                logging.debug(f"_evaluate__  Epoch [{state.epoch}] [{key}] : [{value}]")
        
        def on_step_end( self, args, state, control, **kwargs ):
            if state.global_step % config_json["logging_per_n_step"] == 0:
                logging.debug(f"_step_end__  Epoch [{round(state.epoch, 4) * 100} %]")
            # if len(state.log_history) > 1:
            #     for key, value in state.log_history[-1].items():
            #         logging.debug(f"_step_end__  Epoch [{state.epoch}] [{key}] : [{value}]")


    trainer = Trainer(
        model              = model,
        args               = training_args,
        data_collator      = data_collator,
        eval_dataset       = dataset_dev,
        train_dataset      = dataset_train,
        callbacks          = [LogCallback]
    )


    trainer.train()

    trainer.save_model(output_path)
    trainer.evaluate()

