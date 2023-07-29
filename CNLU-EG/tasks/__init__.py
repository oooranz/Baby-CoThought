#!/bin/bash
# Copyright 2020 Google and DeepMind.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Yao, Shunyu and Yu, Dian and Zhao, Jeffrey and Shafran, Izhak and Griffiths, Thomas L and Cao, Yuan and Narasimhan, Karthik
# (2023), GitHub repository: https://github.com/princeton-nlp/tree-of-thought-llm

def get_task(name, file=None, model=None):
    if name == 'task':
        from .task import IntrinsicTask
        return IntrinsicTask(file)
    elif name == 'text':
        from .text import TextTask
        return TextTask(file, model)
    else:
        raise NotImplementedError
