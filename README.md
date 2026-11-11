# Tweet Sentiment Classification and an LSTM Book Title Generator

## What this project is about

Two NLP projects that cover the path from classic text features to neural language models.

1. **Tweet sentiment classification.** Classify tweets as positive or negative with logistic regression, using three different text representations (Bag-of-Words, sentiment counts, TF-IDF), then study where each one fails.
2. **LSTM book title generator.** Train a word-level LSTM that predicts the next word of a book title, measure it with top-k accuracy and perplexity, and generate new titles. Then improve both the model and the way text is sampled from it.

Topics covered:
- Text preprocessing: tokenization, stop words, lemmatization, tweet-specific cleaning
- Feature engineering for text: Bag-of-Words, hand-built sentiment features, TF-IDF
- Logistic regression for binary classification; precision, recall, F1 and Jaccard (IoU)
- Error analysis: negation, emoticons, preprocessing side effects
- Sequence data preparation: n-gram prefixes, padding, masking, out-of-vocabulary tokens
- LSTM language modelling in Keras; top-k accuracy and perplexity
- Decoding strategies: greedy, temperature sampling, top-k sampling

In both notebooks, data loading and the baseline code come from a provided template. The feature extractors, the TF-IDF model, the text generation functions and all model and decoding improvements are my work.

---

## Part 1: Tweet sentiment classification

Notebook: [`notebooks/tweet_sentiment_classification.ipynb`](notebooks/tweet_sentiment_classification.ipynb)

### The dataset

- NLTK `twitter_samples`: 5,000 positive and 5,000 negative English tweets.
- Stratified split: 7,000 training and 3,000 test tweets, balanced.
- Emoticons are by far the most frequent tokens in training: `:(` appears 3,171 times and `:)` 2,497 times, almost always in tweets of the matching class. They carry much of the signal, which helps explain the very high scores below.

### How I approached it

**Preprocessing (`utils.py`).** `process_tweet()` removes stock tickers, "RT", links and the `#` sign, tokenizes with NLTK's `TweetTokenizer` (lowercase, handles stripped, repeated letters shortened), drops stop words and punctuation, and lemmatizes each word. All three approaches use this same function, so the comparison is about the features, not the cleaning.

**1. Bag-of-Words.** `extract_bow_features()` counts how often each training-vocabulary word appears in a tweet. The vocabulary has 9,197 words, so every tweet becomes a 9,197-number vector that is 99.93% zeros.

**2. Sentiment counts.** `extract_sentiment_features()` compresses a tweet into 3 numbers: a bias of 1, the sum of each word's frequency in positive training tweets, and the same sum for negative tweets. The frequencies come from a `(word, label)` dictionary built on the training set only. The question here: can 3 well-chosen features match 9,197 raw ones?

**3. TF-IDF.** `TfidfVectorizer(analyzer=process_tweet)` plugs the same preprocessing into scikit-learn. The vectorizer is fit on training tweets only and then applied to test tweets, so no information about the test set leaks into the IDF weights. I also plotted the 10 most positive and 10 most negative logistic regression weights to see which tokens drive predictions.

**Evaluation.** Classification report, F1 and Jaccard index (IoU) of the positive class on the 3,000 test tweets. Then two kinds of error analysis: every misclassified test tweet for each model, and 11 hand-written tweets that test negation and repetition ("not good", "not too bad", "great great great").

### Results

| Features | Number of features | Test Jaccard (IoU) | Misclassified test tweets (of 3,000) |
|---|---|---|---|
| Bag-of-Words | 9,197 | 0.9894 | 16 |
| Sentiment counts | 3 | 0.9855 | 22 |
| TF-IDF | 9,197 | **0.9934** | **10** |

TF-IDF: accuracy 0.9967, precision 0.9960, recall 0.9973, F1 0.9967.

What the errors show:
- **Negation is lost.** NLTK's stop-word list contains "not" and "too", so "not good" becomes `["good"]` and "not too bad" becomes `["bad"]`. Both models call "not good" positive (0.7547 BoW, 0.8801 sentiment counts). "I am bad" and "not too bad" get exactly the same score (0.6565 in BoW), because they reduce to the same tokens.
- **Spaced emoticons disappear.** `TweetTokenizer` splits ": (" into ":" and "(", which are then removed as punctuation. Tweets like "u probs had more fun with david than me : (" lose their only sentiment signal and are misclassified by every model.
- **The link regex removes too much.** `https?://.*` deletes everything after the first link, including emoticons at the end. "Get more at http://... :) http://..." keeps only "get", and TF-IDF calls it negative.
- **Repetition inflates confidence.** "great" scores 0.80 in BoW, and "great great great great" scores 0.94, because counts add up linearly.
- Three features come close to 9,197 (Jaccard 0.9855 vs 0.9894), because they capture the main signal: words that appear mostly in positive or mostly in negative tweets.

Improvements these findings suggest: keep negation words (or join them with the next word, such as "not_good"), join spaced emoticons before tokenizing, remove only the URL itself, and use bigrams.

---

## Part 2: LSTM book title generator

Notebook: [`notebooks/lstm_book_title_generator.ipynb`](notebooks/lstm_book_title_generator.ipynb)

### The dataset

- The 10,000 book titles from goodbooks-10k, downloaded from the dataset's public GitHub repository.
- Cleaning: lowercase, keep only letters and spaces. Average title length is 4.97 words.
- 80/20 split into 8,000 training and 2,000 test titles. The vocabulary is limited to the 2,500 most frequent words.
- What I noticed: series information in brackets is kept as words, so "The Hunger Games (The Hunger Games, #1)" becomes "the hunger games the hunger games".

### How I approached it

**Training data.** Each title becomes a set of prefixes: "the hunger" predicts "games", "the hunger games" predicts the next word, and so on. Prefixes are pre-padded with zeros to a fixed length, and the model predicts the last word as a 2,500-way softmax.

**Baseline model.** Embedding (100 dimensions), two LSTM layers of 100 units with dropout 0.2, and a softmax over 2,500 words. Maximum length 20, 10 epochs. It overfits: training accuracy reaches 0.61, while validation loss rises every epoch, from 6.55 to 9.46.

**1. Greedy generation.** `generate_text()` predicts the next word with argmax, appends it, and repeats. Input sequences are padded to `MAX_LENGTH - 1`, because training used the last column as the target. Result: "in the world of ai owls the rat of the". Running it again always gives the same text, because the same weights and the same input always produce the same probabilities.

**2. Temperature sampling.** `generate_text_with_temperature()` rescales the probabilities with `exp(log(p) / T)`, renormalizes, and samples. Low T widens the gaps between probabilities (close to greedy), and high T flattens them (closer to uniform). On the seed "The perfect book for", T = 0.2 kept frequent words ("and", "from", "the"), and T = 2.0 gave varied but unrelated words.

**3. Improving the model.** Five changes to data preparation and the model, with no extra data and no pretrained models:

| Change | Why |
|---|---|
| `mask_zero=True` in the Embedding layer | The LSTM stops treating padding zeros as real words |
| Maximum length 20 to 8 | Titles average 5 words, so most of the 20 slots were padding |
| Prefixes only up to the title's own length | The baseline added copies of the full title for every unused slot (152,000 training sequences, now 28,578) |
| `oov_token='<unk>'` | Rare words get their own token instead of disappearing from the input |
| 10 to 20 epochs | The model was under-trained at 10 epochs |

An engineering constraint: the cuDNN LSTM kernel does not support masking with pre-padding, so both LSTM layers use `use_cudnn=False`. Training is slower, but masking works.

**4. Improving decoding.** The improved model has a side effect: at T = 0.2 it generated only `<unk>`, because `<unk>` is now a very likely next token and low temperature amplifies it. `generate_text_improved()` changes only the sampling, not the model:
- `<unk>` gets probability 0, so it can never be picked.
- Top-k sampling (k = 10): only the 10 most likely words are kept and renormalized before sampling. This cuts the long tail of unrelated words that high temperatures pick.

### Evaluation

Top-1 and top-5 accuracy of the next word on held-out title prefixes, and perplexity (exp of the mean negative log-likelihood of the correct word; a uniform guess over 2,500 words scores 2,500). BLEU is printed by the evaluation function, but I did not use it, because unigram BLEU on a single predicted word adds nothing over top-1 accuracy.

### Results

| Model | Top-1 | Top-5 | Perplexity |
|---|---|---|---|
| Baseline | 0.25 | 0.35 | 1896.31 |
| Improved (masking, length 8, OOV token, 20 epochs) | 0.27 | 0.44 | **169.19** |

The two test sets are built with different maximum lengths and prefix rules (38,000 vs 7,105 sequences), so this is not a strictly like-for-like comparison.

Generated titles for the seed "The perfect book for":

| Decoding | T = 0.2 | T = 1.0 | T = 2.0 |
|---|---|---|---|
| Baseline, temperature | ohare and dying from the | ohare sex from poems world | tell fallen home tree diary |
| Improved model, temperature | `<unk>` `<unk>` `<unk>` `<unk>` `<unk>` | all the `<unk>` lover the | time millennium kiss fablehaven book |
| Improved model, `<unk>` blocked + top-k | **the dark the vampire chronicles** | **the complete war trilogy tales** | **all the clues cahills rhyme** |

What the numbers say:
- Perplexity fell about 11x. The baseline was close to random guessing; masking and removing duplicate prefixes let the model learn from real words instead of padding.
- The baseline put 0.91 probability on "ohare" after "The perfect book for", in line with the overfitting seen in its training curves.
- The improved model's validation loss was lowest at epoch 9 (5.53) and rose slowly after that, so early stopping around epoch 9 is a sensible next step.
- The last row uses the same trained model as the row above it. Only the decoding changed, and the output went from `<unk>` and noise to title-like phrases.
- Not solved: there is no end-of-title token, so every title has exactly 5 generated words and can stop mid-phrase. The fix is to add an `<end>` token to the training data and stop generating when it is predicted.

---

## Repository layout

```
notebooks/tweet_sentiment_classification.ipynb   Part 1
notebooks/lstm_book_title_generator.ipynb        Part 2
notebooks/utils.py                               process_tweet() and build_freqs() for Part 1
requirements.txt
```

## How to run

Both notebooks were run on Google Colab (Part 2 on a T4 GPU), and the saved outputs come from those runs. To run them locally:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook notebooks/
```

Part 1 downloads the NLTK data it needs (`twitter_samples`, `stopwords`, `wordnet`, `omw-1.4`) and imports `utils.py` from the same folder. Part 2 downloads the book list from the goodbooks-10k GitHub repository. Training Part 2 on a CPU works but is much slower.

## Tech stack

Python, NLTK, scikit-learn, TensorFlow / Keras, NumPy, pandas, Matplotlib, Seaborn, WordCloud.
