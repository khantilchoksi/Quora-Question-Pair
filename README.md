# Quora-Question-Pair

## Process

### Step 1 - Cleaning

***File*** - `Preprocessing-With-Split.py`

***Input*** - `train.csv` from https://www.kaggle.com/c/quora-question-pairs/data

Steps taken -
1. Lower case
2. Expand abbreviations and replace common misspellings
3. Remove punctuation
4. Lemmatization (skipped due to no improvement in the meaning of sentence)
5. Stop Words Removal
6. Stemming (skipped due to loss of information)

***Output*** - `questions.csv` and `indexes.csv`

### Step 2 - Data Generation

#### 2A. Word Embeddings for Approach 2

***File*** - `FastText-Word-Vectors.py`

***Input*** - `wiki.simple.vec` (FastText Embeddings) Download from https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.simple.zip

Steps -
1. Read the questions from `questions.csv`
2. For unknown words, generate the embeddings using FastText source code. This code generates `queries.txt` [each line is an out of vocabulary word], input this and Simple English model [https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.simple.zip] to FastText code to get the `qq.oov.vec`, ie. embeddings for out of vocabulary words.
3. For each question, generate a sequence of embeddings using the `wiki.simple.vec` and `qq.oov.vec` embeddings.
4. Creates `*.npy` files in embeddings folder (not uploaded here due to large size) for each question. This can be referenced using the `indexes.csv`.

***Output*** - `queries.txt` and `embeddings`

#### 2B. Data Generation for Approach 1

***File*** - `Cleaned-Data.py`

***Input*** - `wiki.simple.vec`, `qq.oov.vec`, `questions.csv`, `indexes.csv`

Steps -
1. Read the embeddings `wiki.simple.vec` and `qq.oov.vec`, and combine them into `fasttext.pkl`.
2. Join `questions.csv` and `indexes.csv` to get `cleaned-data.csv`.

***Output*** - `fasttext.pkl` and `cleaned-data.csv`

### Step 3 - Approach 1 - Baseline Results

***File*** - `Quora-Question-Pairs.py`

***Inputs*** - `fasttext.pkl` and `cleaned-data.csv`

Steps Taken -
1. Load embeddings `fasttext.pkl` and data `cleaned-data.csv`.
2. Generate features by concatenating average embeddings of both questions, along with the product of average embeddings of both questions. This generates a 900-dimension dense feature representation.
3. Apply Logistic Regression with Grid Search.
4. Apply Gaussian Naive Bayes.
5. Apply Random Forest with Grid Search.
6. Apply Feed Forward with 1 hidden layer.
7. Store the results and metrics.

***Outputs***

Models
- `final_logistic.pkl` - Logistic Regression Grid Search
- `final_gaussian.pkl` - Gaussian Naive Bayes
- `final_random_forest.pkl` - Random Forest Grid Search
- `final_feed_forward.h5` - Feed Forward Neural Network

Results and Plots
- `feed_forward.csv` - evaluated metrics at each epoch for Feed Forward
- `metrics.csv` - evaluated metrics for the models
- `results.pkl` - ROC values for the above models

### Step 4 - Approach 2 - RNN Models

#### Step 4A - Model Selection

***File*** - `approach-2-train.sh`

***Input*** - `attention.py` and `siamese_lstm.py`

Steps Taken -
1. The shell script does a grid search on various parameters of our RNN models.
2. The parameters are -
    - LSTM: parameters `layers` and `depth`
        - `layers` determines the number of stacked LSTM layers. Range - `1, 2`
        - `depth` determines the number of units in each LSTM cell. Range - `30, 60`
    - ATTN (Attention variant of LSTM): parameters `dropout` and `depth`
        - `dropout` the dropout regularization of Fully Connected Neurons. Range - `0.2, 0.4`
        - `depth` determines the number of units in each LSTM cell. Range - `30, 60`

***Outputs***

Models
- `ATTN_[dropout]_[depth].h5` - Attention models
- `LSTM_[layers]_[depth].h5` - LSTM models

Plots
- `ATTN_[dropout]_[depth].csv` - evaluated metrics for the Attention models
- `LSTM_[layers]_[depth].csv` - evaluated metrics for the LSTM models

### Step 4B - Evaluation

***File*** - `rnn_metrics.py`

***Input*** - `LSTM_2_60.h5`, `ATTN_0.2_60.h5`, `results.pkl` and `metrics.csv`

Steps Taken -
1. Load the models.
2. Evaluate the models using test data.
3. Append the evaluation results to the result from Approach 1.

***Outputs***
- `metrics.csv` - evaluated metrics for the above models
- `results.pkl` - ROC values for the above models

## Step 5 - Results and Metrics

***File*** - `Plots-and-Results.ipynb`

***Input*** - `results.pkl` and `metrics.csv`
