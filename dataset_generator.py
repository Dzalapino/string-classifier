"""
Script stores the logic for generating a dataset of strings and saving it to a file
The dataset is generated by generating random strings and checking if they match a given regular expression
Half of the strings in the dataset will match the regular expression and half won't
"""

import random
import re
import pandas as pd


def generate_string(length: int, letters: list[str]) -> str:
    """
    Generates a random string of given length from given letters
    :param length: The length of the string to generate
    :param letters: The letters to generate the string from
    :return: String of given length from given letters
    """
    return ''.join(random.choice(letters) for _ in range(length))


def contains_regex(s: str, regex: str) -> bool:
    """
    Checks if a given string contains a given regular expression
    :param s: The string to check
    :param regex: The regular expression to check against
    :return: Boolean indicating whether the string matches the regular expression
    """
    return re.search(regex, s) is not None


def generate_dataset(dataset_size=10000, string_length=15, from_letters=None,
                     matching_regex='badac') -> pd.DataFrame:
    """
    Generates a dataset of strings and saves it to a file.
    :param dataset_size: The number of strings to generate
    :param string_length: The length of the strings to generate
    :param from_letters: Letters to generate the strings from
    :param matching_regex: Regular expression that approximately 50% strings in the dataset will match
    :return: train and test datasets in the form of pandas DataFrames
    """
    if from_letters is None:
        from_letters = ['a', 'b', 'c', 'd']  # Default letters to generate strings from

    half_size = dataset_size // 2
    data = []  # List of tuples of strings and whether they match the regular expression
    matching_count = 0  # Number of strings that match the regular expression
    not_matching_count = 0  # Number of strings that don't match the regular expression

    print("Generating dataset...")
    # Generate strings until the dataset is full
    while len(data) < dataset_size:
        # Generate a random string
        s = generate_string(string_length, from_letters)

        # Check if the string matches the regular expression
        if contains_regex(s, matching_regex):
            # Check if there is still room for more matching strings
            if matching_count < half_size:
                data.append((s, 1))
                matching_count += 1
        else:
            # Check if there is still room for more non-matching strings
            if not_matching_count < dataset_size - half_size:  # Possibility of not even dataset size
                data.append((s, 0))
                not_matching_count += 1

        print(f'\r    Generated {len(data)} strings out of {dataset_size}...', end='', flush=False)

    print('\nDataset generated successfully!\nSaving dataset separated to train and test parts to file...')

    df = pd.DataFrame(data, columns=['string', 'matches_regex'])  # Create a DataFrame from the data
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle the dataset
    df.to_csv('strings_dataset.csv', index=False)  # Save the dataset to a file

    return df


def load_dataset_from_file(file_name: str) -> pd.DataFrame:
    """
    Returns a dataset from a file without indexing the rows
    :param file_name: The name of the file to read the dataset from
    :return: The dataset read from the file
    """
    return pd.read_csv(file_name)
