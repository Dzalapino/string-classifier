from dataset_generator import generate_dataset, load_dataset_from_file
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow import one_hot


def create_basic_model() -> Sequential:  # Simple model with one filter
    model = Sequential()
    model.add(Conv1D(1, 5, activation='relu', input_shape=(15, 4)))
    model.add(MaxPooling1D(3))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def create_experiment_model() -> Sequential:  # More complex model with 3 filters and 2 dense layers
    model = Sequential()
    model.add(Conv1D(3, 5, activation='relu', input_shape=(15, 4)))
    model.add(MaxPooling1D(3))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def main():

    dataset_df = load_dataset_from_file('strings_dataset.csv')

    # Split the dataset to train and test parts
    train_df, test_df = train_test_split(dataset_df, test_size=0.2, shuffle=False, random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=0.2, shuffle=False, random_state=42)
    train_x = train_df['string']
    train_y = train_df['matches_regex']
    val_x = val_df['string']
    val_y = val_df['matches_regex']
    test_x = test_df['string']
    test_y = test_df['matches_regex']

    # Initialize a Tokenizer
    tokenizer = Tokenizer(char_level=True)

    # Fit the Tokenizer on your text data
    tokenizer.fit_on_texts(train_x)

    # Convert strings to sequences of integers
    train_sequences = tokenizer.texts_to_sequences(train_x)
    val_sequences = tokenizer.texts_to_sequences(val_x)
    test_sequences = tokenizer.texts_to_sequences(test_x)

    # Subtract 1 from the sequences (we don't need 0s for padding as all strings have the same length)
    train_sequences = [[num - 1 for num in seq] for seq in train_sequences]
    val_sequences = [[num - 1 for num in seq] for seq in val_sequences]
    test_sequences = [[num - 1 for num in seq] for seq in test_sequences]

    # Convert sequences of integers to one-hot encoded sequences
    train_x = one_hot(train_sequences, depth=len(tokenizer.word_index))
    val_x = one_hot(val_sequences, depth=len(tokenizer.word_index))
    test_x = one_hot(test_sequences, depth=len(tokenizer.word_index))

    basic_model = create_basic_model()

    print('==================================================')
    print('Train the basic model:')
    basic_model.fit(train_x, train_y, epochs=10, validation_data=(val_x, val_y))

    # Get the weights of the first layer (Conv1D)
    basic_filter_weights = basic_model.layers[0].get_weights()[0]

    print("\nConv1D filter values:")
    print(basic_filter_weights)

    print('\nEvaluate the basic model')
    basic_model.evaluate(test_x, test_y)

    print('\nMake predictions')
    basic_predictions = basic_model.predict(test_x[:10])
    print(f'First 10 labels in the test set:\n{test_y[:10]}')
    print(f'Predictions for the first 10 strings in the test set:\n{basic_predictions}')

    print('\n==================================================')
    print('Train the experiment model:')
    experimental_model = create_experiment_model()
    experimental_model.fit(train_x, train_y, epochs=10, validation_data=(val_x, val_y))

    # Get the weights of the first layer (Conv1D)
    experimental_filter_weights = experimental_model.layers[0].get_weights()[0]

    print("\nConv1D filter values:")
    print(experimental_filter_weights)

    print('\nEvaluate the experimental model')
    experimental_model.evaluate(test_x, test_y)

    print('\nMake predictions')
    experimental_predictions = experimental_model.predict(test_x[:10])
    print(f'First 10 labels in the test set:\n{test_y[:10]}')
    print(f'Predictions for the first 10 strings in the test set:\n{experimental_predictions}')


if __name__ == '__main__':
    main()
