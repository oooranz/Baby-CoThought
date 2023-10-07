# Baby-CoThought

This repository contains the code for the paper: 

>[Baby's CoThought: Leveraging Large Language Models for Enhanced Reasoning in Compact Models](https://arxiv.org/pdf/2308.01684v1.pdf). 

In this work, we apply our "CoThought" pipeline to pretrain a Baby Language Model (BabyLM) with human-like smaller corpus data.

The pretraining data is provided by [Warstadt et al. (2023)](https://arxiv.org/abs/2301.11796) in the framework of the [BabyLM Challenge](https://babylm.github.io/), which has the goal of sample-efficient pretraining on a developmentally plausible corpus at a small human-like data scale.

The restructured data for BabyLM pretraining is available [here](https://huggingface.co/datasets/yaanhaan/Baby-CoThought-Data).

Our pre-trained BabyLM is available [here](https://huggingface.co/yaanhaan/Baby-CoThought).

![](./figures/baby-cothought.png)

## Contents

- `CNLU-EG`: Contains the code for the Creative NLU-Example Generation (CNLU-EG).
- `pretrain`: Contains the code and instructions for pretraining RoBERTa model.
- `eval`: Contains the code for a shared evaluation pipeline from [Warstadt et al. (2023)](https://arxiv.org/abs/2301.11796).

## Creative NLU-Example Generation
1. Download [babylm_data](https://github.com/babylm/babylm.github.io/raw/main/babylm_data.zip), run `./CNLU-EG/data/text/cat_data.py` to merge them.
    ```bash
    cd ./babylm_data/babylm_100M
    cat aochildes.train bnc_spoken.train cbt.train cbt.train children_stories.train open_subtitles.train qed.train switchboard.train > merged_data.txt
    python cat_data.py merged_data.txt text.txt
    ```
   Then we can get the raw data for the next step.
2. Use LLMs to generate the new dataset consists of NLU-Examples.
    ```bash
    cd ./CNLU-EG/scripts/text
    bash cot_sampling.sh
    ```
   
## Pre-Train the BabyLM
- Pre-train the BabyLM with our generated dataset. 
- The generated dataset can be downloaded from [here](https://huggingface.co/datasets/yaanhaan/Baby-CoThought-Data).
   ```shell
   cd ./pretrain
   python RoBERTa.py RoBERTa_config.json
   ```

## Evaluation
- Evaluate the trained BabyLM on a shared pipeline, hosted at [this GitHub link](https://github.com/babylm/evaluation-pipeline ).
- The public validation data utilized is a blend of BLiMP and (Super)GLUE tasks. Additional tasks will be held out for the final evaluation of submitted models.


## Citation

If you found the resources in this repository useful, please cite:

```
@inproceedings{zhang-et-al-2023-babys,
    author = {Zheyu Zhang and Han Yang and Bolei Ma and David Rügamer and Ercong Nie},
    title = {Baby's {C}o{T}hought: {L}everaging Large Language Models for Enhanced Reasoning in Compact Models},
    booktitle = {Proceedings of the {B}aby{LM} {C}hallenge},
    year = {2023},
    month = {December},
    publisher = {Association for Computational Linguistics (ACL)},
}
```
