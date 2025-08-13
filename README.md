# Depression-Detection-TF-IDF-
This project is a text classification system that uses a classical NLP approach to classify text. It leverages TF-IDF (Term Frequency-Inverse Document Frequency) to vectorize text data, which is then used to train a traditional machine learning model for classification

# TF-IDF based Depression Detection System

This project is a text classification system that uses a classical natural language processing (NLP) approach to classify text. It leverages **TF-IDF** (Term Frequency-Inverse Document Frequency) to vectorize text data, which is then used to train a traditional machine learning model for classification. The system is lightweight and efficient, providing an alternative to deep learning models for text analysis.

---

## Key Features

* **TF-IDF Vectorization**: Uses the TF-IDF technique to convert raw text into a numerical format that captures word importance.
* **Traditional Machine Learning Model**: Employs a classical machine learning model for classification, which is lightweight and fast to train.
* **Efficient Workflow**: The training script handles the entire training process, including text vectorization and model saving.
* **Prediction**: A separate prediction script allows for easy testing of the trained model on new text inputs.

---

## Prerequisites

To run this project, you need to have Python installed. It's recommended to set up a virtual environment and install the required libraries. The primary dependencies for this project are:

`pip install scikit-learn pandas numpy`

---

## Getting Started

### 1. Dataset

This project requires a dataset named `depression_dataset.csv`. The file should be placed in the root directory of this project, next to the `Training.py` file. The dataset is expected to have at least two columns: one for the text and one for the label.

### 2. Clone the Repository

Clone this repository to your local machine

### 3. Install Dependencies

Navigate into the project directory and install the required libraries.

`cd [Your Project Directory]`
`pip install -r requirements.txt`

### 4. Train the Model

To train the model, ensure your `depression_dataset.csv` file is in the correct location. The `Training.py` script will automatically load this data, apply the TF-IDF vectorizer, train the classifier, and save both the vectorizer and the model.

Run the training script with this command:

`python Training.py`

### 5. Make Predictions

After training is complete, the `predict.py` script can be used to test the model on new text. The script will load the saved model and vectorizer and enter an interactive loop.

Run the prediction script with this command:

`python predict.py`

---

## Example Usage

Here is an example of what the user can expect to see when they run the prediction script:

```bash
# Example of running the prediction script
$ python predict.py
Enter a sentence to analyze (or type 'quit' to exit): I am feeling so sad and lonely
The text is classified as: Depressed
