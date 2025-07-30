import datasets
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def data_loading():
   
    # Loading Dataset
    dataset = datasets.load_dataset("ag_news")
   
    # Dataset Structure
    print("Dataset structure:")
    print(dataset)
    
    train_dataset = dataset['train']
    test_dataset = dataset['test']
    
    # From here we will get the labels name
    label_names = train_dataset.features['label'].names
    print(f"Label names: {label_names}\n")
   
    # label_names_test = test_dataset.features['label'].names
    # print(f"Label names: {label_names_test}")
   
    # Convert to Pandas DataFrame
    train_data = pd.DataFrame(train_dataset)
    test_data = pd.DataFrame(test_dataset)
    
    # Per category articles distribution- Training Dataset
    plt.figure(figsize=(10, 6))
    sns.countplot(x='label', data=train_data)
    plt.title('Distribution of News Categories in Training Data')
    plt.xlabel('Category')
    plt.ylabel('Number of Articles')
    plt.xticks(ticks=range(len(label_names)), labels=label_names)
    plt.savefig("class_distribution_train.png")

    # Per category articles distribution- Testing Dataset
    plt.figure(figsize=(10, 6))
    sns.countplot(x='label', data=test_data)
    plt.title('Distribution of News Categories in Testing Data')
    plt.xlabel('Category')
    plt.ylabel('Number of Articles')
    plt.xticks(ticks=range(len(label_names)), labels=label_names)
    plt.savefig("class_distribution_test.png")

    return dataset
    

# data_loading()
