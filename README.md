# Beyond Monolingual Boundaries: Sarcasm Detection on Code-Switched Hinglish Text

A collaborative NLP project comparing three approaches to binary sarcasm detection in Hindi-English (Hinglish) social media text. The study investigates supervised multilingual fine-tuning, zero-shot large language model prompting, and cross-lingual transfer from English to Hinglish.

> **Code availability:** The materials supplied for this repository include the implementation notebook for **Approach 1 (MuRIL)**. Approaches 2 (LLaMA-4 Scout) and 3 (XLM-RoBERTa) are described in the report and poster, but their implementation notebooks/scripts were not included. This repository should therefore be understood as a partial code release documenting a three-approach group study, not a complete implementation of all three methods.

## Project team

- Akshada Rane
- Gokulakrishnan Ravi
- Jaideep Manupati

Virginia Tech · CS 5624 project

## Project overview

Code-switching is common in multilingual online communication. Hinglish posts combine Hindi and English, often in Roman script, informal spelling, and sentence-level language mixing. Sarcasm is especially challenging because the intended meaning may differ from the literal wording and rely on cultural knowledge, pragmatic context, irony, or mock praise.

For example, “Waah, kya service hai, bahut impressed!” can be sarcastic criticism rather than genuine praise. A model that relies only on positive words may misunderstand the speaker's intent.

The project compares three paradigms under a shared evaluation setup:

1. **Supervised fine-tuning:** MuRIL trained on labeled Hinglish sarcasm examples.
2. **Zero-shot prompting:** LLaMA-4 Scout 17B prompted through the Groq API without task-specific training or few-shot examples.
3. **Cross-lingual transfer:** XLM-RoBERTa-large trained on English iSarcasmEval data and evaluated on Hinglish.

## Dataset

The primary dataset is the Hinglish Sarcasm Detection Dataset (Swami et al., 2018), containing 5,250 English-Hindi code-mixed tweets with binary sarcastic/non-sarcastic labels and token-level language identification annotations.

The report describes the overall label distribution as 504 sarcastic (9.6%) and 4,746 non-sarcastic (90.4%). The experiment uses a stratified 80/10/10 train/validation/test split with `random_state=42`:

| Split | Examples |
|---|---:|
| Train | 4,200 |
| Validation | 525 |
| Test | 525 |

The notebook expects two local source files:

- `Sarcasm_tweets.txt` — tweet IDs and tweet text.
- `Sarcasm_tweet_truth.txt` — tweet IDs and YES/NO labels.

**The raw dataset files are not included.** Obtain them from the dataset's original source and follow its terms of use. The notebook will not run until these files are available and their formatting is compatible with its parsing logic.

## Approaches and reported results

The following headline metrics are transcribed from the final report's main results table.

| Approach | Model | Paradigm | Accuracy | Macro F1 | Sarcastic F1 |
|---|---|---|---:|---:|---:|
| 1 | **MuRIL** | Supervised fine-tuning | **0.9467** | **0.8683** | **0.77** |
| 2 | LLaMA-4 Scout 17B | Zero-shot prompting | 0.8971 | 0.7660 | 0.59 |
| 3 | XLM-RoBERTa-large | English-to-Hinglish cross-lingual transfer | 0.9048 | 0.4750 | 0.0000 |
| Baseline | Majority class | Always predict non-sarcastic | 0.9010 | 0.4739 | 0.0000 |

Accuracy must be interpreted carefully: roughly 90% of the dataset is non-sarcastic, so a majority-class predictor already obtains about 90% accuracy. Macro F1 and the sarcastic-class F1 help reveal whether a model recognizes the minority class.

### Approach 1 — MuRIL (notebook included)

The supplied notebook fine-tunes `google/muril-base-cased`, a multilingual encoder pretrained for Indian languages and transliterations. Its workflow includes:

1. Reading and parsing the tweet and label files, then joining records by tweet ID.
2. Converting YES/NO labels into binary values (1 = sarcastic, 0 = non-sarcastic).
3. Cleaning text by removing URLs and user mentions, retaining hashtag words while removing the hash symbol, filtering selected special characters, and normalizing whitespace.
4. Creating stratified train, validation, and test partitions.
5. Tokenizing with the MuRIL tokenizer, with padding/truncation to a maximum length of 128.
6. Fine-tuning a two-class sequence classifier for three epochs.
7. Applying class-weighted cross-entropy to address the class imbalance.
8. Evaluating accuracy and Macro F1 and examining classification results, prediction distributions, and false negatives.

The notebook calculates class weights from the training labels. Its displayed run reports approximately 0.5531 for non-sarcastic and 5.2109 for sarcastic. The report's headline result for MuRIL is 94.67% accuracy and 0.8683 Macro F1.

### Approach 2 — LLaMA-4 Scout (described in report/poster; code not supplied)

The group evaluated LLaMA-4 Scout 17B via the Groq API in a zero-shot setting. The report describes no task-specific training and no few-shot examples.

Prompt calibration was a key experimental finding. The initial prompt (v1) caused the model to predict sarcasm for approximately 86% of tweets, despite the true sarcasm rate being around 10%; its Macro F1 was approximately 0.27. The revised prompt (v2) explicitly communicated the majority-class base rate, distinguished sarcasm from anger or strong opinions, and instructed the model to choose non-sarcastic when uncertain. The revised prompt brought the predicted sarcasm rate close to 10% and achieved Macro F1 = 0.7660.

The report also notes temperature 0 for reproducibility, request delays, periodic checkpointing, and retry/backoff handling for rate limits.

### Approach 3 — XLM-RoBERTa (described in report/poster; code not supplied)

The group trained `xlm-roberta-large` on English iSarcasmEval data and evaluated it zero-shot on Hinglish. The report describes a two-phase training process: classifier-head warmup with the backbone frozen, followed by full-model fine-tuning with a smaller learning rate.

The model achieved 0.9048 accuracy but only 0.4750 Macro F1 and 0.0000 sarcastic-class F1. The report interprets this as majority-class collapse: the model did not detect sarcastic examples in the evaluated Hinglish test set. This result illustrates why accuracy alone is inadequate for imbalanced classification.

## Key takeaways

- **In-domain supervision performed best:** MuRIL achieved the strongest Macro F1 in the reported comparison.
- **Prompt calibration matters:** a single prompt revision moved LLaMA-4 from a severe over-prediction pattern to competitive zero-shot performance.
- **Cross-lingual transfer was insufficient on its own:** XLM-RoBERTa's high accuracy hid its failure to identify sarcastic examples.
- **Pragmatic and cultural context is important:** sarcasm cues such as mock praise and understatement may not transfer directly from English to Hinglish.

## Repository structure

```text
hinglish-sarcasm-detection/
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   └── Approach_1_MuRIL.ipynb
└── docs/
    ├── Final_Report.pdf
    └── Project_Poster.pdf
```

The notebook and document paths above describe the intended structure of the prepared project package. Confirm that these files have been uploaded before attempting to open them from GitHub.

## Run the available notebook

### Environment setup

Python 3.10 or 3.11 is a practical starting point. Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv

# Windows PowerShell
.venv\\Scripts\\Activate.ps1

# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

Place `Sarcasm_tweets.txt` and `Sarcasm_tweet_truth.txt` in the working directory (or change the notebook paths), then launch Jupyter:

```bash
jupyter notebook
```

Open `notebooks/Approach_1_MuRIL.ipynb` and execute the cells in order. The notebook downloads the pretrained MuRIL tokenizer/model and uses PyTorch and Hugging Face Transformers. A CUDA-capable GPU is recommended; CPU fine-tuning may be considerably slower.

Exact dependency versions were not specified in the supplied notebook, so this project does not claim a version-locked reproduction environment. The Groq API is relevant to reproducing Approach 2; do not commit API keys or other credentials.

## Limitations and reproducibility notes

1. **Partial code release:** only the Approach 1 notebook was supplied. Approaches 2 and 3 are documented in the report/poster, but their source code and complete execution logs were not provided.
2. **Dataset files are external:** the two raw tweet/label files required by the notebook are not packaged here.
3. **Script coverage:** the report notes that the planned Devanagari subset was unavailable. The reported evaluation is limited to the available Romanized/transliterated Hinglish data; it is not a complete Roman-vs-Devanagari comparison.
4. **Metric/count differences across artifacts:** the report's split table lists 52 sarcastic test examples, while its MuRIL confusion matrix and the notebook's classification report indicate 50. The poster also displays a MuRIL class-weight notation that differs from the weights shown in the notebook. Headline comparison metrics above follow the report's main-results table; the detailed count/weight differences should be reconciled with original experiment outputs before an exact reproduction.
5. **Single held-out split:** the report presents one stratified split. Additional seeds or external validation would help establish robustness.

## Future work

The report proposes few-shot prompting with Hinglish-specific examples, domain-adaptive pretraining on unlabeled Hinglish, ensembles combining MuRIL and LLM reasoning, cultural-context retrieval, extension beyond binary sarcasm detection, and evaluation using more comprehensive mixed-script data including Devanagari.

## References

1. Aguilar, G., Kar, S., and Solorio, T. (2020). *LinCE: A Centralized Benchmark for Linguistic Code-switching Evaluation.* LREC.
2. Conneau, A. et al. (2020). *Unsupervised Cross-lingual Representation Learning at Scale.* ACL.
3. Abu Farha, I. et al. (2022). *iSarcasmEval: Intended Sarcasm Detection in English and Arabic.* SemEval.
4. Khanuja, S. et al. (2021). *MuRIL: Multilingual Representations for Indian Languages.* arXiv:2103.10730.
5. Meta AI (2025). *LLaMA 4: Open Foundation and Fine-Tuned Chat Models.*
6. Swami, S. et al. (2018). *A Corpus of English-Hindi Code-Mixed Tweets for Sarcasm Detection.* arXiv:1805.11869.

Refer to the final report in the `docs/` directory for the full discussion and bibliographic details.

## Acknowledgment

This was a **collaborative group project** by Akshada Rane, Gokulakrishnan Ravi, and Jaideep Manupati. The repository presents the study's three experimental paradigms while clearly identifying which implementation code was available in the supplied materials.
