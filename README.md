# Hausa Sentiment Analysis with AfroXLM-R

A three class sentiment classifier for Hausa tweets, built by fine tuning a
multilingual Transformer and deployed as a public web application.

## Project links

| Resource | Link |
| --- | --- |
| Live web application | https://hausa-sentiment.streamlit.app |
| Fine tuned model | https://huggingface.co/hasbyUmutoniwabo/hausa-sentiment-afroxlmr |
| Demo video | https://www.youtube.com/watch?v=au36-z5TRVc |
| Report | https://docs.google.com/document/d/1qWrIUKEQ7Dc_9bG5BTVXNGyEbK3R0V_xF972-efCPpc/edit?usp=sharing |

## The problem

Most sentiment analysis tools only work for languages with large amounts of
labelled text. Hausa is one of the most widely spoken languages in Nigeria and
across West Africa, but it has very few working NLP tools.

Anyone who needs to understand what people are saying online in Hausa currently
has two bad options: translate everything into English first, which loses
meaning and adds translation errors, or read it all by hand, which does not
scale. This project removes both problems by reading Hausa directly.

Likely users are media monitoring teams, public health and government
communication units, customer support teams at Nigerian companies, and
researchers studying public opinion in Northern Nigeria.

## Dataset

The Hausa subset of **AfriSenti**, a benchmark of African language tweets
labelled by native speakers.

| Split | Tweets |
| --- | --- |
| Train | 14,172 |
| Validation | 2,677 |
| Test | 5,303 |
| Total | 22,152 |

Labels are positive, negative and neutral, and the training set is close to
balanced at 34.7 / 33.1 / 32.3 percent. Tweets average 14 words.

Source: https://github.com/afrisenti-semeval/afrisent-semeval-2023

The repository also ships a Hausa sentiment lexicon of 1,215 positive and 1,001
negative words, used here to build the simplest baseline.

### Preprocessing

Four steps, each a deliberate choice:

1. URLs removed. They carry no sentiment and every one is unique.
2. Twitter handles replaced with a placeholder. Keeps the fact of a mention,
   drops the username, protects privacy.
3. Repeated whitespace and internal line breaks collapsed.
4. **Emojis kept.** They carry real sentiment in social media text. Removing
   them discards signal. This decision was tested, see Experiment 3.

Text is not lowercased and punctuation is not stripped, because the Transformer
tokenizer handles both itself.

### Known data problems

- **Inconsistent spelling.** The same word appears in several written forms,
  for example *wallahi* written as *wlh* or *wallah*.
- **Sparse vocabulary.** 34,359 unique word forms in training, only 9,514
  appearing twice or more. That is 72.3 percent of word types occurring once.
- **Single domain.** All tweets, heavily weighted towards politics, football
  and religion.
- **Offensive language.** The corpus contains real profanity and slurs. It was
  not filtered, because filtering would systematically bias the negative class.

## Models

Three baselines and one proposed approach.

**Baseline 1, lexicon rule.** Counts positive minus negative words from the
Hausa lexicon. No learning at all. This answers whether machine learning is
needed in the first place.

**Baseline 2, TF-IDF and logistic regression.** Character n grams of length 2
to 5, 50,000 features, fitted on training data only. Character level rather
than word level on purpose: because Hausa tweets are spelled inconsistently, a
word model treats *wallahi* and *wlh* as unrelated while a character model sees
shared sequences.

**Baseline 3, BiLSTM.** 200 dimensional embeddings learned from scratch, one
bidirectional LSTM layer with 128 hidden units per direction, max pooling
across time, dropout 0.4. 2,241,491 parameters. Covers the word embedding and
recurrent network material from the course.

**Proposed approach, fine tuned AfroXLM-R base.** A multilingual Transformer
already adapted to 17 African languages including Hausa. 278,045,955
parameters. Subword tokenization means unseen words are split into known pieces
instead of becoming unknown tokens, which is the structural advantage over the
BiLSTM.

Training configuration: learning rate 2e-5, AdamW with weight decay 0.01,
linear schedule with 10 percent warmup, batch size 16, 5 epochs, max length 64,
mixed precision.

The learning rate is 50 times smaller than the BiLSTM's, because AfroXLM-R
already knows useful things about Hausa and large steps would overwrite them.

## Experiments

| Run | Configuration | Split | Accuracy | Macro F1 |
| --- | --- | --- | --- | --- |
| 0 | Lexicon rule | Validation | 0.5622 | 0.5577 |
| 1 | TF-IDF, logistic regression | Validation | 0.7691 | 0.7705 |
| 2 | BiLSTM | Validation | 0.7243 | 0.7252 |
| 3 | AfroXLM-R, 3 epochs | Validation | 0.7878 | 0.7878 |
| 4 | AfroXLM-R, 5 epochs | Validation | 0.8028 | 0.8036 |
| 5 | AfroXLM-R, lr 3e-5 | Validation | 0.7972 | 0.7970 |
| 6 | AfroXLM-R, emojis removed | Validation | 0.7987 | 0.7988 |

### What each experiment showed

**Training longer helped.** Validation loss was still falling at epoch 3, so
the model had not converged. Extending to 5 epochs added 1.6 points.

**Higher learning rate was less stable.** 3e-5 scored 0.7 points lower, but the
more telling difference is behavioural: validation loss rose to 0.6882 against
0.6186, and scores moved up and down between epochs instead of improving
steadily.

**The emoji result is inconclusive, and this matters.** Removing emojis gave
0.7988 against 0.8036, a difference of 0.5 points. Before computing test
scores, the winning configuration was retrained from scratch and reached
0.7961, a 0.75 point gap from an identical earlier run. Run to run variance is
therefore larger than the effect being measured. Neither the emoji result nor
the learning rate result can be claimed as a real finding. Establishing a
difference this small would require several random seeds per configuration.

## Results

Test set, scored once after all decisions were final.

| Model | Accuracy | Macro F1 |
| --- | --- | --- |
| Lexicon rule | 0.5497 | 0.5452 |
| TF-IDF and logistic regression | 0.7466 | 0.7453 |
| **AfroXLM-R fine tuned** | **0.7788** | **0.7787** |

Per class, for the final model:

| Class | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| Negative | 0.731 | 0.794 | 0.762 | 1,759 |
| Neutral | 0.757 | 0.695 | 0.725 | 1,789 |
| Positive | 0.851 | 0.848 | 0.850 | 1,755 |

Test macro F1 is 2.5 points below validation. That drop is expected: everything
was tuned on validation data, so the test score is the honest estimate.

### Why the BiLSTM lost to a logistic regression

This is the most instructive result in the project. The BiLSTM has 2.24 million
parameters and scored 4.5 points *below* character n grams.

The cause is in the data. 72.3 percent of Hausa word forms in training occur
exactly once, so they are dropped from the vocabulary and become unknown
tokens. Many of them are alternative spellings of words the model has already
seen, but a word level model cannot connect them. Character n grams and subword
tokenization both sidestep this entirely.

The model also overfit from epoch 3: training loss fell from 0.5401 to 0.1612
while validation loss rose from 0.6406 to 0.9506.

## Error analysis

The final model got 1,173 of 5,303 test tweets wrong.

### Two hypotheses, tested

**Hypothesis 1: religious vocabulary misleads the model. Refuted.**

Every one of the eight most confident errors contained prayer language, and all
were predicted positive. The obvious conclusion was that the model reads
religious vocabulary as a positive signal.

Testing it against the full test set reversed the finding:

| Measure | With religious vocabulary | Without |
| --- | --- | --- |
| Accuracy | 0.857 | 0.741 |
| True positive share | 0.682 | |
| Predicted positive share | 0.713 | |

Accuracy is 11.6 points *higher* on those tweets, and the predicted
distribution tracks the true one within 3 points. Religious vocabulary in this
corpus genuinely is 68 percent positive, so the model learned a correct prior,
not a shortcut.

What survives is narrower. The failures are rare but expensive: the minority of
cases that contradict the prior, curses phrased as prayers, are predicted
wrongly with near total confidence. A confidence threshold cannot catch them.

The methodological lesson is that inspecting only the most confident errors
selects on the outcome you are trying to explain.

**Hypothesis 2: Hausa negation causes errors. Supported.**

Hausa negates a verb with a *ba ... ba* frame wrapped around it, which reverses
the meaning of a positive word like *son* (to like). Accuracy on the 921 tweets
containing negation markers is 0.754 against 0.784 without. A 3 point drop.
This compares two subgroups under one fixed model, so run variance does not
apply and the effect is real.

### The actual main weakness: the neutral class

| Predicted class | Mean confidence | Median |
| --- | --- | --- |
| Positive | 0.936 | 0.994 |
| Negative | 0.880 | 0.962 |
| Neutral | 0.817 | 0.863 |

Neutral has the lowest F1 at 0.725, the lowest recall at 0.695, the lowest
confidence, and is involved in 944 of 1,173 errors, which is 80.5 percent.

The reason is that neutral is not a sentiment but the absence of one, so it has
no characteristic vocabulary. Positive tweets have praise words, negative
tweets have complaints, and both give the model a clear signal. Neutral tweets
are defined only by lacking both.

### Other findings

**Length does not predict failure the way you would expect.** Accuracy is 0.804
for tweets of 1 to 5 words, dips to 0.757 at 11 to 20 words, then rises. Short
tweets are usually single clear reactions; medium ones mix topics and
code switch.

**The model is somewhat aware when it is wrong.** Correct predictions average
0.909 confidence, incorrect ones 0.774. A threshold could route uncertain cases
to a human, though it would miss the confident religious language errors.

## Deployment

A Streamlit application hosted on Streamlit Community Cloud. The user types
Hausa text and gets a predicted label with confidence scores for all three
classes. Six examples are preloaded, including two deliberate failure cases.

The fine tuned model lives on the Hugging Face Model Hub and is downloaded when
the application starts, then cached so it loads once rather than on every
interaction.

**The app uses exactly the same preprocessing function as training.** If it
cleaned text differently, the model would receive input in a format it never
learned on and accuracy would drop with no visible error.

The free tier sleeps after inactivity and takes about thirty seconds to wake.
That is normal, not a fault.

## Repository structure

```
.
├── streamlit_app.py       # the deployed web application
├── requirements.txt       # pinned versions
├── README.md
├── notebooks/
│   └── hausa_sentiment.ipynb   # full pipeline with saved outputs
└── figures/
    ├── label_distribution.png
    ├── confusion_matrix_test.png
    └── accuracy_by_length.png
```

The notebook runs end to end: data loading, preprocessing, all three baselines,
the Transformer fine tuning, every experiment, test evaluation, and the full
error analysis. Outputs are saved, so results can be read without rerunning.

## Reproducing this

### The web application

```bash
git clone https://github.com/hasby-umutoniwabo/ML_technique_I-Summative.git
cd ML_technique_I-Summative
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

The model downloads from Hugging Face on first run, roughly 1.1GB.

### Training from scratch

Open `notebooks/hausa_sentiment.ipynb` in Google Colab, set the runtime to a
T4 GPU, and run all cells. The dataset is cloned from GitHub inside the
notebook, so no manual download is needed. Full training takes about 40 minutes
including all experiments.

Random seeds are fixed at 42 for reproducibility, though GPU nondeterminism
still produces run to run variation of roughly 0.75 macro F1 points, as
documented above.

### Using the model directly

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

MODEL_ID = "hasbyUmutoniwabo/hausa-sentiment-afroxlmr"
tok = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)

enc = tok("allah ya karo shekaru masu albarka", return_tensors="pt")
with torch.no_grad():
    probs = torch.softmax(model(**enc).logits, dim=1)[0]

for label, p in zip(["negative", "neutral", "positive"], probs):
    print(f"{label}: {p:.3f}")
```

## Limitations

- Every configuration was trained once. Differences smaller than about one
  macro F1 point are not reliable.
- 14,172 training tweets is small. Rare constructions cannot be learned.
- Twitter only, mostly politics, football and religion. Do not assume it
  transfers to formal Hausa writing or other domains.
- No native speaker reviewed the model's outputs. Some counted errors may be
  genuinely ambiguous tweets.
- The claim that the model relies on vocabulary over structure is inferred from
  error patterns, not measured with attention or attribution methods.
- Only learning rate and epoch count were varied. Batch size, dropout, max
  length and the choice of pretrained model were fixed.

## Future work

- **Multiple seeds per configuration.** The single most valuable improvement,
  and it only costs compute time. It would turn the inconclusive experiments
  into defensible statements.
- **Target the negation and curse cases.** Add training examples pairing the
  same vocabulary with opposite meanings, so the model has to learn the
  structures that invert meaning.
- **Train across languages.** AfriSenti has 13 other African languages.
  Multilingual training is what AfroXLM-R was designed to enable.
- **Confidence based routing in deployment.** Return predictions above a
  threshold, flag the rest for human review.
- **Parameter efficient fine tuning.** Compare LoRA against full fine tuning
  for similar accuracy at much lower memory cost.
- **Native speaker evaluation.** Would reveal error types that comparison
  against existing labels cannot.

## What I used and what I built

**Used as they are:** the AfriSenti dataset and its Hausa lexicon, the
pretrained AfroXLM-R weights, and the libraries scikit-learn, PyTorch, Hugging
Face Transformers and Streamlit.

**Written for this project:** the preprocessing pipeline, the lexicon scoring
rule, the TF-IDF baseline, the BiLSTM architecture and training loop, the
Transformer fine tuning loop with mixed precision and best epoch selection, the
shared evaluation functions, all experiments, the full error analysis including
both hypothesis tests, and the deployed application.

## References

Alabi, J. O., Adelani, D. I., Mosbach, M., and Klakow, D. (2022). Adapting
pre-trained language models to African languages via multilingual adaptive
fine-tuning. *COLING 2022*, 4336 to 4349.
https://aclanthology.org/2022.coling-1.382

Conneau, A., et al. (2020). Unsupervised cross-lingual representation learning
at scale. *ACL 2020*, 8440 to 8451. https://aclanthology.org/2020.acl-main.747

Devlin, J., Chang, M. W., Lee, K., and Toutanova, K. (2019). BERT:
Pre-training of deep bidirectional transformers for language understanding.
*NAACL 2019*, 4171 to 4186. https://aclanthology.org/N19-1423

Hochreiter, S., and Schmidhuber, J. (1997). Long short-term memory.
*Neural Computation*, 9(8), 1735 to 1780.

Muhammad, S. H., et al. (2023). AfriSenti: A Twitter sentiment analysis
benchmark for African languages. *EMNLP 2023*, 13968 to 13981.
https://aclanthology.org/2023.emnlp-main.862

Muhammad, S. H., et al. (2023). SemEval-2023 Task 12: Sentiment analysis for
African languages. *SemEval 2023*, 2319 to 2337.
https://aclanthology.org/2023.semeval-1.315

Muhammad, S. H., et al. (2022). NaijaSenti: A Nigerian Twitter sentiment
corpus for multilingual sentiment analysis. *LREC 2022*, 590 to 602.
https://aclanthology.org/2022.lrec-1.63

Nekoto, W., et al. (2020). Participatory research for low-resourced machine
translation: A case study in African languages. *Findings of EMNLP 2020*,
2144 to 2160. https://aclanthology.org/2020.findings-emnlp.195

Ogueji, K., Zhu, Y., and Lin, J. (2021). Small data? No problem! Exploring the
viability of pretrained multilingual language models for low-resourced
languages. *MRL Workshop 2021*, 116 to 126.
https://aclanthology.org/2021.mrl-1.11

Vaswani, A., et al. (2017). Attention is all you need. *NeurIPS 30*.
https://arxiv.org/abs/1706.03762

Wolf, T., et al. (2020). Transformers: State-of-the-art natural language
processing. *EMNLP 2020 System Demonstrations*, 38 to 45.
https://aclanthology.org/2020.emnlp-demos.6

## Licence and acknowledgements

The AfriSenti dataset is released for research use by its authors and the
Masakhane community. AfroXLM-R is released under Apache 2.0 by the authors of
Alabi et al. (2022). This repository's own code is available for academic use.
