cot_prompt = '''
Use the given sentences to create an example paragraph of an NLU task and its corresponding labels. The 5 sentences are: {input}.

Make a plan then write and determine. Your output should be of the following format:

Plan:
Your plan here.

Paragraph:
Your paragraph here.

Task:
[Only the NLU task name here, without additional information.]

Labels:
[Only the labels here, without additional information.]
'''


vote_prompt = '''Given an instruction and several choices, decide which choice is most promising. Analyze each choice in detail, then conclude in the last line "The best choice is {s}", where s the integer id of the choice.
'''

compare_prompt = '''Briefly analyze the coherency of the following two paragraphs. Conclude in the last line "The more coherent passage is 1", "The more coherent passage is 2", or "The two passages are similarly coherent".
'''

score_prompt = '''Analyze the following paragraph, then at the last line conclude "Thus the coherency score is {s}", where s is an integer from 1 to 10.
'''