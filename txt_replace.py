import os
import re

# List of words/phrases to replace (case-insensitive)
replace_phrases = [
    "a photograph",
    "a image",
    "a picture",
    "a photo",
    "an image",
    "an photo",
    "an picture",
    "photograph",
    "photo",
    "picture"
]


# Function to replace phrases in a file
def replace_in_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read()

    # Replace the phrases with 'video', using case-insensitive matching
    for phrase in replace_phrases:
        content = re.sub(rf'\b{re.escape(phrase)}\b', 'video', content, flags=re.IGNORECASE)

    with open(file_path, 'w') as file:
        file.write(content)


# Function to process all text files in a directory
def process_directory(directory_path):
    for filename in os.listdir(directory_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory_path, filename)
            replace_in_file(file_path)
            print(f"Processed file: {file_path}")


# Specify the directory where the text files are stored
directory_path = '/image_dataset_here'  # Update with the correct path
process_directory(directory_path)
