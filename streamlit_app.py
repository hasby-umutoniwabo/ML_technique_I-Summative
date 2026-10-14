import re
import torch
import torch.nn.functional as F
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_ID = "hasbyUmutoniwabo/hausa-sentiment-afroxlmr"
LABELS = ["negative", "neutral", "positive"]

st.set_page_config(page_title="Hausa Sentiment Analysis")


# Loads the model once instead of on every interaction
@st.cache_resource
def load_model():
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
    model.eval()
    return tok, model


# Identical to the cleaning used in training. A mismatch here would
# quietly lower accuracy without raising any error.
def clean_tweet(text):
    text = str(text)
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"@\w+", "@user", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def predict(text, tok, model):
    enc = tok(clean_tweet(text), truncation=True, max_length=64, return_tensors="pt")
    with torch.no_grad():
        probs = F.softmax(model(**enc).logits, dim=1)[0]
    return {LABELS[i]: float(probs[i]) for i in range(3)}


st.title("Hausa Sentiment Analysis")
st.write("AfroXLM-R fine tuned on the Hausa subset of AfriSenti. Test set macro F1 0.779.")

tok, model = load_model()

EXAMPLES = {
    "Positive": "allah ya karo shekaru masu albarka my princess",
    "Negative": "nigeria is a joke ko sun bayyanan ma aikin banza ne masu cin kudi",
    "Neutral": "toh ya kake gani",
    "Failure: negation": "Ba na son wannan sam",
    "Failure: curse as prayer": "allah ka kayar dashi a jihar nashi",
}

choice = st.selectbox("Load an example", ["Type my own"] + list(EXAMPLES.keys()))
default = "" if choice == "Type my own" else EXAMPLES[choice]

text = st.text_area("Hausa text", value=default, height=120)

if st.button("Analyse") and text.strip():
    scores = predict(text, tok, model)
    top = max(scores, key=scores.get)
    st.subheader(f"Prediction: {top}")
    for label in LABELS:
        st.write(f"{label}: {scores[label]:.1%}")
        st.progress(scores[label])

st.caption(
    "The last two examples are known failures, kept deliberately. "
    "Hausa negation and curses phrased as prayers are both misread."
)