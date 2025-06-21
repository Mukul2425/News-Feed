import pandas as pd
import spacy
from transformers import pipeline

# Load models
nlp = spacy.load("en_core_web_sm")
sentiment_analyzer = pipeline("sentiment-analysis")

def extract_entities(text):
    doc = nlp(text)
    return [ent.text for ent in doc.ents if ent.label_ in ["ORG", "PRODUCT"]]

def analyze_sentiment(text):
    result = sentiment_analyzer(text)[0]
    return result['label'], result['score']

def process_headlines(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    results = []

    for _, row in df.iterrows():
        headline = row['headline']
        entities = extract_entities(headline)
        sentiment_label, sentiment_score = analyze_sentiment(headline)

        results.append({
            "headline": headline,
            "source": row.get("source", ""),
            "date": row.get("date", ""),
            "entities": ", ".join(entities),
            "sentiment": sentiment_label,
            "score": round(sentiment_score, 2)
        })

    output_df = pd.DataFrame(results)
    output_df.to_csv(output_csv, index=False)
    print(f"[✓] Processed headlines saved to {output_csv}")

# Example usage
if __name__ == "__main__":
    process_headlines("data/raw_headlines.csv", "data/processed_output.csv")
