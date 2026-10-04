# Credit card fraud detection

Machine learning project for detecting fraudulent credit card transactions.

## Requirements

- Docker Compose
- Python
- Make

## Setup

Clone the repository and set up the environment:

```bash
git clone https://github.com/a-mx/fraud-detection
cd fraud-detection
cp .env.example .env
```
Create virtual environment

```bash
python -m venv .venv
.\.venv\Scripts\activate #Windows
source .venv/bin/activate #Linux
```
Install required packages
```bash
pip install uv
uv pip install -r requirements.txt
```
Launch container
```bash
make up
```
Train models:

```bash
make train MODEL=xgb
```

Verify API health:

```bash
make health
```
