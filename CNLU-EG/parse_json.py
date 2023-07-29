import os
import json

# Set the directory path where the JSON files are located
directory = "./logs/text"

# Set the path of the output text file
output_file = "pretrain_data.txt"

# Open the output file in append mode
with open(output_file, "a") as f:
    # Iterate over each file in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            file_path = os.path.join(directory, filename)

            # Read the JSON file
            with open(file_path, "r") as json_file:
                data = json.load(json_file)

                # Iterate over each object in the JSON array
                for obj in data:
                    infos_values = obj["infos"]
                    ys_values = obj["ys"]

                    # Iterate over r_values and ys_values simultaneously
                    for infos, ys in zip(infos_values, ys_values):
                        # Check if the "r" value is greater or equal to 7.0
                        if infos['r'] >= 7.0:
                            # Write the "ys" sentence to the output file
                            f.write(ys + "\n--------\n")
