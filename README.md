# VIN Annual Report NLP Topic Analytics Dashboard

This project analyzes Vingroup annual reports from 2019 to 2024 using Vietnamese NLP and topic modeling. It includes a BERTopic pipeline and an interactive Streamlit dashboard for exploring business themes over time.

## Features

- PDF extraction and OCR fallback
- Vietnamese text cleaning, segmentation, tokenization, and stopword filtering
- LDA benchmarking with coherence score
- BERTopic modeling with multilingual sentence embeddings, UMAP, HDBSCAN, and c-TF-IDF
- Manual business topic taxonomy
- Streamlit dashboard for topic trend analysis, evidence exploration, and keyword search

## Project Structure

```text
NLP_VIN.ipynb
topic_dashboard/
  app.py
  requirements.txt
  dashboard_artifacts/
    segments.parquet
    topic_keywords.json
    topic_info.csv
    manual_topic_map.json
```
## How to Run
- cd topic_dashboard
- pip install -r requirements.txt
- streamlit run app.py
