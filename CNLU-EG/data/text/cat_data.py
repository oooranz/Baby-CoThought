import argparse

# Example usage: python cat_data.py text_shuf.txt text.txt

def concatenate_sentences(file_path, output_path, group_size):
    with open(file_path, 'r') as file:
        sentences = file.readlines()

    grouped_sentences = [sentences[i:i + group_size] for i in range(0, len(sentences), group_size)]

    with open(output_path, 'w') as output_file:
        for group in grouped_sentences:
            concatenated_line = ' '.join(group).replace('\n', '')
            output_file.write(concatenated_line + '\n')

def parse_args():
    parser = argparse.ArgumentParser(description='Concatenate sentences from a text file')
    parser.add_argument('input_file', type=str, help='Path to the input file containing sentences')
    parser.add_argument('output_file', type=str, help='Path to the output file to save the concatenated sentences')
    parser.add_argument('--group_size', type=int, default=5, help='Number of sentences to concatenate per line')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    concatenate_sentences(args.input_file, args.output_file, args.group_size)


