from spelling_correction import SpellingCorrector

def run_tests():
    """
    Instantiates the corrector and runs a series of tests.
    """
    # Initialize the corrector with paths to the data files in the 'data' folder
    corrector = SpellingCorrector(
        unigrams_path='data/unigrams.csv',
        bigrams_path='data/bigrams.csv',
        subs_path='data/substitutions.csv',
        dels_path='data/deletions.csv',
        adds_path='data/additions.csv',
        vocab_path='data/word_frequencies.txt'
    )

    # --- Test cases are organized by expected model performance ---

    print("--- 1. Scenarios Where the Model Should Work Well ---")
    good_cases = {
        "speling": "spelling",         # Common substitution
        "hapy": "happy",               # Common deletion
        "cak": "cake",                 # Deletion at the end of a word
        "acress": "across",            # Correctly picks the most frequent word
        "wether": "weather"            # Common vowel swap
    }
    # For this suite, we expect success, so is_failure_suite is False
    run_test_suite(corrector, good_cases, is_failure_suite=False)


    print("\n--- 2. Scenarios Where the Model Is Expected to Fail ---")
    bad_cases = {
        # Real-word error: 'peace' is a valid word, so the model won't correct it.
        "peace": "peace",
        # Multiple errors: 'inconvient' is edit distance 2 from 'inconvenient'.
        "inconvient": "inconvient",
        # No valid candidates: 'zzxy' has no single-edit neighbors in a typical dictionary.
        "zzxyy": "zzxyy"
    }
    # For this suite, we expect failure, so is_failure_suite is True
    run_test_suite(corrector, bad_cases, is_failure_suite=True)


def run_test_suite(corrector, test_cases, is_failure_suite=False):
    """
    Helper function to run a set of test cases and print results.
    The 'is_failure_suite' flag changes the status message for expected failures.
    """
    passed_count = 0
    for typo, expected in test_cases.items():
        result = corrector.correct(typo)
        is_correct = (result == expected)
        if is_correct:
            passed_count += 1

        # Determine the status message based on the suite type
        if is_failure_suite:
            status = "FAIL (AS EXPECTED)" if is_correct else "UNEXPECTEDLY PASSED"
        else:
            status = "PASSED" if is_correct else "FAILED"
            
        print(f"Input: '{typo}' -> Output: '{result}' (Expected: '{expected}') - {status}")

    print(f"--> Suite Summary: {passed_count}/{len(test_cases)} tests passed.")


if __name__ == "__main__":
    run_tests()