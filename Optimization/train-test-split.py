from sklearn.model_selection import train_test_split

# Step 1: split train+validate and test
train_val, test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: split train and validate
train, val = train_test_split(
    train_val, test_size=0.25, random_state=42)  # 0.25*0.8=0.2


#! stratify in train test split

# lets split the data into training and test sets
data1_train, data1_test = train_test_split(
    data1,
    test_size=0.3,
    # stratify ensures the train and test sets keep the same class/group proportions
    # as the original data.
    stratify=data1['group'],
    random_state=123)
