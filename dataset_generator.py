"""
Script meant to generate the dataset of strings for the project
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


def matches_regex(s: str, regex: str) -> bool:
    """
    Checks if a given string contains a given regular expression
    :param s: The string to check
    :param regex: The regular expression to check against
    :return: Boolean indicating whether the string matches the regular expression
    """
    return re.search(regex, s) is not None


def generate_dataset(dataset_size=10000, string_length=15, from_letters=None,
                     matching_regex='badac') -> None:
    """
    Generates a dataset of strings and saves it to a file.
    :param dataset_size: The number of strings to generate
    :param string_length: The length of the strings to generate
    :param from_letters: Letters to generate the strings from
    :param matching_regex: Regular expression that approximately 50% strings in the dataset will match
    :return: None
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
        if matches_regex(s, matching_regex):
            # Check if there is still room for more matching strings
            if matching_count < half_size:
                data.append((s, True))
                matching_count += 1
        else:
            # Check if there is still room for more non-matching strings
            if not_matching_count < dataset_size - half_size:  # Possibility of not even dataset size
                data.append((s, False))
                not_matching_count += 1

        print(f'\r    Generated {len(data)} strings out of {dataset_size}...', end='', flush=False)

    print('\nDataset generated successfully!\nSaving dataset to file...')

    # Save the dataset to a file
    df = pd.DataFrame(data, columns=['string', 'matches_regex'])

    # Shuffle the dataset to better distribute the matching and non-matching strings
    df = df.sample(frac=1).reset_index(drop=True)

    # Create training and testing sets in the ratio 80:20
    train_size = int(dataset_size * 0.8)
    train_df = df[:train_size]
    test_df = df[train_size:]

    train_df.to_csv('strings_train.csv', index=False)
    test_df.to_csv('strings_test.csv', index=False)
