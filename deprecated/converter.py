from datasets import load_dataset
import opencc

# Login to save

# Initialize the converter
converter = opencc.OpenCC("s2t.json")

# Load the dataset
dataset_dict = load_dataset("alphrc/lilm")

# Define a function to convert text
def convert_text(example):
    example["input"] = converter.convert(example["instruction"])
    example["output"] = converter.convert(example["output"])
    return example

# Apply the conversion function to the training set
train_ds = dataset_dict["train"].map(convert_text, remove_columns="instruction")
dataset_dict["train"] = train_ds

# Print the first example to verify the conversion
print(dataset_dict["train"][0])
dataset_dict.push_to_hub(
    repo_id="TC_Canton_Dialogue",
)

