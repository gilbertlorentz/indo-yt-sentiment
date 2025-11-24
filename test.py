from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

MODEL_NAME = "w11wo/indonesian-roberta-base-sentiment-classifier"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

def analyze_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    probs = torch.softmax(logits, dim=1)
    sentiment_id = torch.argmax(probs).item()

    label = model.config.id2label[sentiment_id]
    confidence = probs[0][sentiment_id].item()

    return {"sentiment": label, "confidence": confidence}


if __name__ == "__main__":
    print(analyze_sentiment("Aplikasinya bagus banget!"))
