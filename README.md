[![Python Application Test](https://github.com/U1186204/Noisy-Channel-Model-Spelling-Correction/actions/workflows/python-app.yml/badge.svg)](https://github.com/U1186204/Noisy-Channel-Model-Spelling-Correction/actions/workflows/python-app.yml)
[![GitHub Repo](https://img.shields.io/badge/Repo-View_on_GitHub-blue.svg)](https://github.com/U1186204/Noisy-Channel-Model-Spelling-Correction)

# Noisy-Channel-Model-Spelling-Correction

This document outlines the design and behavior of the noisy-channel spelling corrector.

## Repository Tree
```
Noisy-Channel-Model-Spelling-Correction
├── data/
│   ├── additions.csv
│   ├── bigrams.csv
│   ├── deletions.csv
│   ├── substitutions.csv
│   ├── unigrams.csv
│   └── word_frequencies.txt
├── spelling_correction.py
├── test.py
└── README.md
```

## 1. Modeling Assumptions

The spelling corrector is based on the noisy channel model, which evaluates candidate corrections using the formula:

$$ \hat{w} = \arg\max_{w \in \text{candidates}} P(w) \cdot P(x|w) $$

Where $x$ is the misspelled word and $w$ is a potential correction. To avoid floating point underflow, we work with log probabilities:

$$ \hat{w} = \arg\max_{w \in \text{candidates}} (\log P(w) + \log P(x|w)) $$

Here are the key assumptions made in the implementation:

### Prior Model: $P(w)$

* **Word Unigram Model**: The prior probability of a word, $P(w)$, is estimated from its frequency in a large corpus of English text. It is calculated as $P(w) = \frac{\text{count}(w)}{N}$, where $N$ is the total number of words in the corpus.
* **Data Source**: A word frequency list (`word_frequencies.txt`) is used for this model. This file must be provided and contain word-count pairs.

### Channel Model: $P(x|w)$

The channel model estimates the probability of the typo $x$ given that the intended word was $w$. It is based on single-edit operations.

* **Single Error Assumption**: The model assumes that any given typo contains at most **one** error (one insertion, deletion, substitution, or transposition).
* **Smoothing**: Add-1 (Laplace) smoothing is applied to all probability calculations to handle edits that were not seen in the training data. This prevents zero probabilities.
* **Deletion Probability $P(x|w)$**: Calculated based on the deleted character and its preceding character.
    * Formula: `P(prefix | prefix + char) = (count(del[prefix, char]) + 1) / (count(bigram[prefix + char]) + V_bigram)`
    * The count for the bigram `prefix + char` is taken from `bigrams.csv`.
* **Insertion (Addition) Probability $P(x|w)$**: Calculated based on the inserted character and its preceding character.
    * Formula: `P(prefix + char | prefix) = (count(add[prefix, char]) + 1) / (count(unigram[prefix]) + V_unigram)`
    * The count for the `prefix` character is taken from `unigrams.csv`.
* **Substitution Probability $P(x|w)$**: Calculated based on the original character being substituted for another.
    * Formula: `P(sub_char | orig_char) = (count(sub[orig_char, sub_char]) + 1) / (count(unigram[orig_char]) + V_unigram)`
* **Transposition Probability $P(x|w)$**: This refers to swapping two adjacent characters (e.g., `ac` -> `ca`). Since no data file was provided for transposition counts, a simple model is used:
    * The probability is estimated as `1 / N`, where `N` is the total word count from the vocabulary. This assigns a low, uniform probability to all transpositions, assuming they are rare but possible.

### General Assumptions
* **Case Insensitive**: All words are converted to lowercase.
* **Non-Word Errors Only**: The corrector only attempts to fix words that are not already present in the vocabulary (`word_frequencies.txt`). It will return any known word as-is.

## 2. Performance Scenarios

### Scenarios Where It Works Well
The corrector excels at fixing common, single character or keyboard mistakes.

1.  **Common Substitutions**: `speling` → `spelling`. The substitution `l` for `ll` is a common error, and `spelling` has a high prior probability.
2.  **Common Deletions**: `hapy` → `happy`. Deleting the second `p` is a possible error that the model can identify. The high frequency of the word "happy" helps it become the top candidate.
3.  **Vowel Mistakes**: `wether` → `weather`. Swapping `e` and `a` is a frequent error captured in `substitutions.csv`. The model correctly identifies "weather" as the more probable intended word over "whether" if the context isn't considered.

### Scenarios Where It Could Do Better
The model's simplicity leads to predictable failures in more complex situations.

1.  **Word Nuances Errors**: In the sentence "I want a **peace** of cake," the corrector will not fix "peace" as it might be a valid word in the dictionary. The model has no mechanism to detect that it's incorrect in this context.
2.  **Multiple Errors**: A word like `inconvient` (missing 'en' and substituting 'ie') has an edit distance of 2 from `inconvenient`. The current model, which only generates candidates of edit distance 1, will fail to find the correct word.
3.  **Context Ignorance**: For the typo `acress`, the model might choose `across` over `actress` because "across" is a much more common word (higher prior probability). In a sentence like "She is a talented **acress**," a human would instantly know "actress" is correct. The unigram prior model does not evaluate sentence context.

### Here are the Test Results
```bash
--- 1. Scenarios Where the Model Should Work Well ---
Input: 'speling' -> Output: 'spelling' (Expected: 'spelling') - PASSED
Input: 'hapy' -> Output: 'happy' (Expected: 'happy') - PASSED
Input: 'cak' -> Output: 'cake' (Expected: 'cake') - PASSED
Input: 'acress' -> Output: 'across' (Expected: 'across') - PASSED
Input: 'wether' -> Output: 'weather' (Expected: 'weather') - PASSED
--> Suite Summary: 5/5 tests passed.

--- 2. Scenarios Where the Model Is Expected to Fail ---
Input: 'peace' -> Output: 'peace' (Expected: 'peace') - FAIL(AS EXPECTED)
Input: 'inconvient' -> Output: 'inconvient' (Expected: 'inconvient') - FAIL(AS EXPECTED)
Input: 'zzxyy' -> Output: 'zzxyy' (Expected: 'zzxyy') - FAIL(AS EXPECTED)
--> Suite Summary: 3/3 tests passed.
```

## 3. Analysis and Improvements
The corrector's poor decisions stem directly from its modeling assumptions: it only corrects errors for non-existing words, assumes a single mistake, and excludes context.


## Potential Improvements the model does not account for
1.  **Use a Better Language Model (Prior)**: Replace the unigram prior $P(w)$ with a **bigram or trigram model**. This would allow the model to score candidates based on the preceding word(s) (e.g., $P(\text{actress} | \text{talented})$ vs. $P(\text{across} | \text{talented})$), which would solve the `acress` example.
2.  **Implement Real Word Error Correction**: Modify the algorithm to generate correction candidates for *all* words in a sentence, not just unknown ones. The system would then score the probability of the entire candidate sentence and choose the sentence with the highest overall score.
