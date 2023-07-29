import pandas as pd

# Read the data from the txt file
with open("pretrain_data.txt", "r") as file:
    data = file.read()

# Split the data into individual examples
examples = data.split("--------\n")

# Create a DataFrame to store the examples
df = pd.DataFrame(examples, columns=["Example"])

# Extract the task from each example
df["Task"] = df["Example"].str.extract(r"Task:\n(.*)\n")

# Count the frequency of each task
task_counts = df["Task"].value_counts().reset_index()
task_counts.columns = ["Task", "Frequency"]

# Sort the examples by task frequency
df = df.merge(task_counts, on="Task").sort_values("Frequency", ascending=False)

# Group the examples by task and concatenate the paragraphs
grouped_df = df.groupby("Task")["Example"].apply(lambda x: "--------\n".join(x)).reset_index()

# Save the grouped examples to a file
grouped_df.to_csv("pretrain.txt", sep="\n", index=False, header=False, quoting=0)

# Output the distribution of tasks
print(task_counts)
