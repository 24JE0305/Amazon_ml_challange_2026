# Business Entity Resolution Pipeline

## Overview
This pipeline matches business records from a deduplicated reference source (Source 1) against two noisy sources (Source 2 and Source 3) using TF-IDF character n-gram blocking and cosine similarity matching. 

## Environment Setup
The environment requires standard data science libraries as pinned in `requirements.txt`.
1. Create a virtual environment: `python -m venv venv`
2. Activate it: `.\venv\Scripts\activate` (Windows) or `source venv/bin/activate` (Mac/Linux)
3. Install dependencies: `pip install -r requirements.txt`

## Reproduction Instructions
To reproduce the output from the raw dataset, run the main pipeline script from the root of the project:

```bash
python code/business_entity_resolution/src/main.py