# 📊 Consumer Financial Complaints Analysis & Classification

An end-to-end data analysis, NLP exploration, and classification project analyzing consumer complaints submitted to the **Consumer Financial Protection Bureau (CFPB)**.

---

## 📌 Project Overview
The CFPB maintains a nationwide database of financial complaints filed by consumers against banks, lenders, credit card issuers, and credit bureaus. This project focuses on:
- Identifying high-frequency complaint categories, issues, and company trends.
- Analyzing consumer narratives using Natural Language Processing (NLP).
- Developing predictive models to categorize complaints or forecast consumer disputes.

---

## 📂 Project Structure

```bash
consumer-complaint-project/
├── Consumer_Complaints_Analysis.ipynb  # Primary Jupyter notebook containing analysis & models
├── requirements.txt                    # Project dependencies
├── .gitignore                          # Excludes large raw data, cache, and checkpoints
├── README.md                           # Project documentation
└── data/
    ├── consumer_complaints.csv         # Full CFPB dataset (~175MB, kept locally)
    └── sample_consumer_complaints.csv  # Lightweight sample dataset for quick testing
```

---

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have **Python 3.9+** and `pip` installed.

### 2. Setup Virtual Environment (Recommended)
```bash
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Running the Notebook
Open VS Code, Jupyter Lab, or Jupyter Notebook:
```bash
jupyter notebook Consumer_Complaints_Analysis.ipynb
```
Select the Python kernel and run the cells sequentially.

---

## 📑 Dataset Details
- **Source**: Consumer Financial Protection Bureau (CFPB)
- **Features Include**:
  - `date_received`: Date complaint was filed
  - `product` & `sub_product`: Financial product category (e.g., Mortgage, Credit Card, Student Loan)
  - `issue` & `sub_issue`: Nature of the issue
  - `consumer_complaint_narrative`: Detailed text description written by the consumer
  - `company`: Financial institution involved
  - `company_response_to_consumer`: Resolution status (e.g., Closed with explanation, Closed with monetary relief)
  - `timely_response`: Whether the company responded within the required window
  - `consumer_disputed?`: Whether the consumer contested the company's response

> **Note on Dataset Storage**: Due to GitHub's 100 MB file limit, the full 175 MB `consumer_complaints.csv` is excluded from git tracking via `.gitignore`. A sample file (`sample_consumer_complaints.csv`) is provided for testing.

---

## 👤 Author
- **GitHub**: [@anushajain3107](https://github.com/anushajain3107)
