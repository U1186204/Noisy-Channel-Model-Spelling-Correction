import csv
import math
import re
from collections import defaultdict

class SpellingCorrector:
    def __init__(self, unigrams_path, bigrams_path, subs_path, dels_path, adds_path, vocab_path):
        """
        Initializes the spelling corrector by loading and building all necessary models.
        """
        self.alphabet = 'abcdefghijklmnopqrstuvwxyz'
        
        # Load raw count data
        self.unigram_counts = self._load_counts(unigrams_path, key_cols=['unigram'])
        self.bigram_counts = self._load_counts(bigrams_path, key_cols=['bigram'])
        self.sub_counts = self._load_counts(subs_path, key_cols=['original', 'substituted'])
        self.del_counts = self._load_counts(dels_path, key_cols=['prefix', 'deleted'])
        self.add_counts = self._load_counts(adds_path, key_cols=['prefix', 'added'])
        self.vocab = self._load_counts(vocab_path, key_cols=['word'], is_vocab=True)

        # Vocabulary and total word count for prior probability
        self.vocabulary = set(self.vocab.keys())
        self.total_word_count = sum(self.vocab.values())
        
        # Total counts for smoothing
        self.total_unigram_count = sum(self.unigram_counts.values())
        self.total_bigram_count = sum(self.bigram_counts.values())

    def _load_counts(self, path, key_cols, is_vocab=False):
        """A helper function to load CSV data into a dictionary."""
        counts = defaultdict(int)
        try:
            with open(path, 'r', encoding='utf-8') as f:
                if is_vocab:
                    # Handle space-delimited word frequency file
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 2:
                            counts[parts[0]] = int(parts[1])
                else:
                    # Handle CSV files
                    reader = csv.DictReader(f)
                    for row in reader:
                        key = tuple(row[k] for k in key_cols)
                        if len(key) == 1:
                            key = key[0]
                        counts[key] = int(row['count'])
        except FileNotFoundError:
            print(f"Error: The file at {path} was not found.")
            exit(1)
        return counts

    def _log_prior_prob(self, word):
        """Calculates the log prior probability of a word."""
        # Use add-1 smoothing for unseen words (though candidates are from vocab)
        count = self.vocab.get(word, 0)
        return math.log((count + 1) / (self.total_word_count + len(self.vocabulary)))

    def _log_channel_prob(self, typo, candidate):
        """
        Calculates the log channel model probability P(typo|candidate).
        This function identifies the single edit and calculates its probability.
        """
        typo = '#' + typo
        candidate = '#' + candidate
        
        # Deletion: P(prefix | prefix + char)
        if len(typo) < len(candidate):
            for i in range(len(candidate)):
                if i >= len(typo) or typo[i] != candidate[i]:
                    prefix = candidate[i-1]
                    deleted_char = candidate[i]
                    bigram = candidate[i-1:i+1]
                    
                    del_count = self.del_counts.get((prefix, deleted_char), 0)
                    bigram_count = self.bigram_counts.get(bigram, 0) if len(bigram) == 2 else self.unigram_counts.get(prefix, 0)
                    
                    # Add-1 smoothing
                    return math.log((del_count + 1) / (bigram_count + self.total_bigram_count))
        
        # Insertion: P(prefix + char | prefix)
        elif len(typo) > len(candidate):
            for i in range(len(typo)):
                if i >= len(candidate) or typo[i] != candidate[i]:
                    prefix = candidate[i-1]
                    added_char = typo[i]
                    
                    add_count = self.add_counts.get((prefix, added_char), 0)
                    prefix_count = self.unigram_counts.get(prefix, 0) if prefix != '#' else self.total_word_count

                    # Add-1 smoothing
                    return math.log((add_count + 1) / (prefix_count + self.total_unigram_count))
                    
        # Substitution or Transposition
        elif len(typo) == len(candidate):
            diffs = [(i, typo[i], candidate[i]) for i in range(len(typo)) if typo[i] != candidate[i]]
            
            # Substitution
            if len(diffs) == 1:
                i, typo_char, cand_char = diffs[0]
                
                sub_count = self.sub_counts.get((cand_char, typo_char), 0)
                unigram_count = self.unigram_counts.get(cand_char, 0)

                # Add-1 smoothing
                return math.log((sub_count + 1) / (unigram_count + self.total_unigram_count))
            
            # Transposition
            elif len(diffs) == 2:
                i1, t1, c1 = diffs[0]
                i2, t2, c2 = diffs[1]
                if i2 == i1 + 1 and t1 == c2 and t2 == c1:
                    # No transposition data, so use a small, uniform probability
                    return math.log(1 / self.total_word_count)

        return -float('inf') # Should not happen with edit distance 1

    def _generate_candidates(self, word):
        """Generates a set of candidate words with an edit distance of 1."""
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        
        deletes = [L + R[1:] for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
        replaces = [L + c + R[1:] for L, R in splits if R for c in self.alphabet]
        inserts = [L + c + R for L, R in splits for c in self.alphabet]

        all_edits = set(deletes + transposes + replaces + inserts)
        return {w for w in all_edits if w in self.vocabulary}

    def correct(self, original_word: str) -> str:
        """
        Provides the most likely spelling correction for a single word.
        """
        word = original_word.lower()

        # If the word is correct, return it
        if word in self.vocabulary:
            return word

        # Generate candidate corrections
        candidates = self._generate_candidates(word)

        if not candidates:
            return original_word

        # Score each candidate using the noisy channel model
        scored_candidates = {}
        for cand in candidates:
            log_prior = self._log_prior_prob(cand)
            log_channel = self._log_channel_prob(word, cand)
            scored_candidates[cand] = log_prior + log_channel
            
        # Return the best candidate
        if not scored_candidates:
             return original_word
             
        return max(scored_candidates, key=scored_candidates.get)