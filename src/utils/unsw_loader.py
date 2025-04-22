import pandas as pd

def load_unsw_datasets(train_path, test_path):
    """
    Load the UNSW-NB15 training and testing datasets.

    Args:
        train_path (str): Path to the training CSV file.
        test_path (str): Path to the testing CSV file.

    Returns:
        tuple: (X_train, y_train, X_test, y_test)
    """
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Drop non-numeric or identifier columns if present
    drop_cols = ['id', 'attack_cat', 'proto', 'service', 'state']
    for col in drop_cols:
        if col in train_df.columns:
            train_df = train_df.drop(columns=col)
        if col in test_df.columns:
            test_df = test_df.drop(columns=col)

    # Separate features and labels
    y_train = train_df['label']
    X_train = train_df.drop(columns='label')

    y_test = test_df['label']
    X_test = test_df.drop(columns='label')

    return X_train, y_train, X_test, y_test