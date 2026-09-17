# 📘 Consumer Complaint Intelligence System
## Complete 3-Stage Master Interview Notes & Technical Concept Guide

> **Author:** Anusha Jain  
> **Repository:** `anushajain3107/consumer-complaints-analysis`  
> **Project Scope:** End-to-End Classification (86% F1), Hybrid Retrieval (BM25 + FAISS + RRF + Cross-Encoder), and Evidence-Grounded RAG.

---

# 📑 TABLE OF CONTENTS
1. [Topic 1: Exploratory Data Analysis & Data Quality Audits](#topic-1-exploratory-data-analysis--data-quality-audits)
2. [Topic 2: Text Preprocessing & Strategic Category Consolidation](#topic-2-text-preprocessing--strategic-category-consolidation)
3. [Topic 3: Feature Engineering & Machine Learning Classification](#topic-3-feature-engineering--machine-learning-classification)
4. [Topic 4: Sparse Lexical Retrieval (BM25)](#topic-4-sparse-lexical-retrieval-bm25)
5. [Topic 5: Dense Semantic Search (Sentence-Transformers + FAISS)](#topic-5-dense-semantic-search-sentence-transformers--faiss)
6. [Topic 6: Hybrid Search & Reciprocal Rank Fusion (RRF)](#topic-6-hybrid-search--reciprocal-rank-fusion-rrf)
7. [Topic 7: Cross-Encoder Re-Ranking & Deep Diagnostics](#topic-7-cross-encoder-re-ranking--deep-diagnostics)

---

# Topic 1: Exploratory Data Analysis & Data Quality Audits

### 🧠 Stage 1: Deep Theory & Intuition
* **Data Volume & Source:** Over 208,000 real-world consumer complaints from the Consumer Financial Protection Bureau (CFPB).
* **Missing Narratives Audit:** Over 65% of raw CFPB complaints lack free-text narratives (consumers only ticked categories). For NLP text classification, non-null narrative filtering is mandatory.
* **Class Imbalance:** Significant skew between high-frequency categories (Credit Reporting, Debt Collection) and smaller categories. Class imbalance ratio exceeds $10:1$.
* **Text Length Distribution:** Narrative word counts are heavily right-skewed. 95% of complaints contain $\le 250$ words, meaning setting maximum sequence lengths to 256 or 384 tokens prevents out-of-memory errors without context loss.

### 💻 Stage 2: Code & Line-by-Line Breakdown
```python
# Narrative Length Statistics & Skew Inspection
df_text['word_count'] = df_text['Consumer Complaint'].apply(lambda x: len(str(x).split()))
stats = df_text['word_count'].describe(percentiles=[0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
```
* `str(x).split()`: Breaks text on whitespaces to count word tokens.
* `percentiles=[..., 0.95, 0.99]`: Calculates exact tail cutoffs to establish sequence truncation bounds for transformer and vector models.

### 🎯 Stage 3: Top Interview Questions & Model Answers
* **Q1: Why can't standard Accuracy be used as the primary metric for this dataset?**
  * **Answer:** *"With an imbalance ratio exceeding 10:1 between majority and minority classes, a naive classifier predicting only the top majority classes would achieve high nominal accuracy while completely failing on minority classes. Therefore, Macro and Weighted F1-Scores are mandatory to evaluate balanced precision and recall across all categories."*
* **Q2: What does narrative length analysis reveal about modeling choices?**
  * **Answer:** *"The word count distribution is strongly right-skewed with a median of ~110 words and a 95th percentile under 250 words. This empirical evidence guides sequence truncation bounds (`max_length=256`), ensuring 95%+ of narrative context is captured without wasting GPU compute on unneeded padding."*

---

# Topic 2: Text Preprocessing & Strategic Category Consolidation

### 🧠 Stage 1: Deep Theory & Intuition
* **CFPB Anonymization Noise:** CFPB redacts Personal Identifiable Information (PII) using repeated Xs (`XXXX`, `XX/XX/XXXX`). Treating these as literal vocabulary words pollutes TF-IDF feature space with useless high-frequency tokens.
* **The Nomenclature Shift (The Big Discovery):** CFPB historically renamed several product categories around 2017:
  - *Credit card or prepaid card* $\leftrightarrow$ *Credit card*
  - *Checking or savings account* $\leftrightarrow$ *Bank account or service*
  - *Credit reporting, credit repair services...* $\leftrightarrow$ *Credit reporting*
  These pairs share 100% identical vocabulary. Consolidating them into **7 non-overlapping classes** eliminates artificial ambiguity and drove model accuracy from **73.3% $\to$ 85.3% (+12%)**.
* **Stratified Splitting:** Splitting data into Train (80%), Val (10%), and Test (10%) using target stratification preserves exact category proportions across all partitions.

### 💻 Stage 2: Code & Line-by-Line Breakdown
```python
# 1. Regex Text Sanitization
def clean_text(text):
    if not isinstance(text, str): return ""
    text = text.lower()                                    # 1. Lowercase
    text = re.sub(r'https?://\S+|www\.\S+', '', text)      # 2. Strip URLs
    text = re.sub(r'\b[x]{2,}\b|\bxx[/\-]\S+', '', text)   # 3. Strip CFPB XXXX
    text = re.sub(r'[^a-z\s]', ' ', text)                  # 4. Strip non-letters
    return re.sub(r'\s+', ' ', text).strip()               # 5. Clean whitespaces

# 2. Consolidation Mapping
df_clean['Product_Cleaned'] = df_clean['Product'].replace({
    'Credit card or prepaid card': 'Credit card',
    'Checking or savings account': 'Bank account or service',
    'Credit reporting, credit repair services, or other personal consumer reports': 'Credit reporting'
})
```
* `re.sub(r'\b[x]{2,}\b', ...)`: Removes words composed of 2 or more 'x' characters (CFPB redacted account numbers).
* `.replace(dict)`: Replaces matching keys while keeping unmentioned categories untouched.
* `LabelEncoder()`: Converts string targets into integer labels ($0$ to $6$).

### 🎯 Stage 3: Top Interview Questions & Model Answers
* **Q1: How did you handle CFPB category renaming in your pipeline?**
  * **Answer:** *"Through confusion matrix error analysis, I discovered massive cross-prediction between historical alias pairs like 'Credit card' and 'Credit card or prepaid card'. These represented a 2017 CFPB portal taxonomy update rather than distinct financial products. Consolidating these into 7 clean categories eliminated label ambiguity and increased test accuracy from 73.3% to 85.3%."*
* **Q2: Why is Stratified Splitting critical over random shuffling?**
  * **Answer:** *"In imbalanced multi-class datasets, naive random splitting risks under-representing or completely starving minority classes in validation and test splits. Stratification guarantees that the class distribution in train, val, and test splits matches the population distribution exactly."*

---

# Topic 3: Feature Engineering & Machine Learning Classification

### 🧠 Stage 1: Deep Theory & Intuition
* **TF-IDF with Sublinear TF:**
  - Standard TF awards linear importance ($TF = 20 \implies 20\times$ score).
  - Sublinear TF replaces raw frequency with $1 + \log(TF)$ when $TF > 0$. This models diminishing returns—repeating a word 20 times indicates emphasis, not 20x information content.
* **Why Linear Support Vector Classifier (LinearSVC) Beats Logistic Regression:**
  - High-dimensional text spaces (10,000 sparse TF-IDF features) are linearly separable.
  - While Logistic Regression minimizes log-loss across all points, SVM finds the **Maximum Margin Hyperplane**—maximizing geometric distance to support vectors, yielding higher robustness against boundary noise.
* **Overfitting & Generalization Diagnostics:**
  - A model is healthy if the generalization gap ($Train - Test$) is small ($< 5\%$).
  - Our results: **Train 90.08% vs. Test 86.03% (Gap: 4.05%)**, with Validation (85.93%) and Test (86.03%) matching within $0.1\%$.

### 💻 Stage 2: Code & Line-by-Line Breakdown
```python
# 1. TF-IDF Feature Extraction
tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), stop_words='english', sublinear_tf=True)
X_train_vec = tfidf.fit_transform(train_df['text'])
X_test_vec  = tfidf.transform(test_df['text'])       # Transform ONLY (Prevents Data Leakage)

# 2. LinearSVC Training
svm_model = LinearSVC(C=0.5, class_weight='balanced', random_state=42, max_iter=2000)
svm_model.fit(X_train_vec, y_train)

# 3. Model Explainability: Extracting Top Features
feature_names = np.array(tfidf.get_feature_names_out())
top_words = feature_names[np.argsort(svm_model.coef_[class_idx])[::-1][:6]]
```
* `sublinear_tf=True`: Uses $1 + \log(TF)$ scaling.
* `fit_transform` on train vs `transform` on test: Guarantees zero test data contamination (no data leakage).
* `coef_`: Exposes learned feature weights, enabling Explainable AI (XAI) verification.

### 🎯 Stage 3: Top Interview Questions & Model Answers
* **Q1: How do you mathematically explain Sublinear TF?**
  * **Answer:** *"Sublinear TF applies logarithmic scaling ($1 + \log(TF)$) to term frequency. A term appearing 20 times in a complaint does not carry 20 times the information of a term appearing once; it reflects ranting or repetitive reporting. Sublinear scaling dampens term saturation and prevents repetitive words from overpowering other critical features."*
* **Q2: How do you prove your classification model isn't memorizing training data?**
  * **Answer:** *"We evaluated empirical generalization across Train (90.08%), Validation (85.93%), and Test (86.03%). The generalization gap between train and test is just 4.05% (< 5%), and validation and test match within 0.1%, proving robust generalization without overfitting."*

---

# Topic 4: Sparse Lexical Retrieval (BM25)

### 🧠 Stage 1: Deep Theory & Intuition
* **What is BM25?** Best Matching 25 is the gold-standard probabilistic ranking function for keyword search (used in Elasticsearch and Lucene).
* **The BM25 Mathematical Formula:**
  $$\text{Score}(D, Q) = \sum_{q \in Q} \text{IDF}(q) \cdot \frac{f(q, D) \cdot (k_1 + 1)}{f(q, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)}$$
* **The $k_1$ Parameter (Term Saturation - Default 1.5):** Controls how quickly repeated words saturate. Unlike TF-IDF where score grows indefinitely, BM25 term frequency asymptotically approaches an upper limit ($k_1 + 1$).
* **The $b$ Parameter (Document Length Normalization - Default 0.75):**
  - $b=0$: Document length is ignored (long documents win unfairly).
  - $b=1$: Document length is heavily penalized.
  - $b=0.75$: The empirical sweet spot—penalizes bloated complaints while retaining detailed, informative narratives.
* **Summation Over Query Words ($\sum_{q \in Q}$):** BM25 calculates scores *only* for the words present in the query and sums them into **one single scalar score per document**.

### 💻 Stage 2: Code & Line-by-Line Breakdown
```python
# 1. Corpus Tokenization
tokenized_corpus = [doc.lower().split() for doc in corpus_texts]

# 2. Build Okapi BM25 Index
bm25 = BM25Okapi(tokenized_corpus)

# 3. Query Scoring & Ranking
scores = bm25.get_scores(query.lower().split())
top_indices = np.argsort(scores)[::-1][:top_k]
```
* `BM25Okapi(tokenized_corpus)`: Computes IDF for all unique terms and stores average document length across the 10,000 corpus.
* `get_scores()`: Evaluates query terms across all 10,000 complaints, returning a 1D numpy array of 10,000 BM25 scores in $< 0.4$ seconds.

### 🎯 Stage 3: Top Interview Questions & Model Answers
* **Q1: Why use BM25 over standard TF-IDF in modern retrieval pipelines?**
  * **Answer:** *"Standard TF-IDF suffers from document length bias (longer documents accumulate higher scores) and lacks term frequency saturation. BM25 solves both via the $b$ parameter (penalizing bloated text) and $k_1$ parameter (asymptotically capping repetitive keyword impact)."*
* **Q2: What is the exact mathematical role of $b=0.75$?**
  * **Answer:** *"$b$ controls the strength of document length normalization between 0 and 1. At $b=0$, length penalty is disabled. At $b=1$, score is divided purely by relative length. A value of $b=0.75$ provides 75% length normalization with 25% baseline tolerance, effectively neutralizing verbose noise without penalizing comprehensive reports."*

---

# Topic 5: Dense Semantic Search (Sentence-Transformers + FAISS)

### 🧠 Stage 1: Deep Theory & Intuition
* **Why Dense Retrieval?** BM25 fails when users use synonyms or colloquial phrasing (e.g., searching *"bank sold my property unlawfully"* returns zero BM25 matches if documents use *"foreclosure"*).
* **Embedding Model (`all-MiniLM-L6-v2`):**
  - Converts text into a continuous **384-dimensional dense vector space**.
  - Sits at the optimal speed-to-quality Pareto frontier: 5x faster than heavy models, ~80MB footprint, runs sub-second on CPU, yet retains 95%+ of large transformer retrieval quality.
* **FAISS (`IndexFlatIP`):**
  - Facebook AI Similarity Search. Written in C++ for maximum throughput.
  - `Flat`: Performs exact, uncompressed brute-force search (100% recall, zero approximation error).
  - `IP` (Inner Product): Computes dot product.
* **The Normalization Trick (Inner Product = Cosine Similarity):**
  - Cosine formula: $\frac{A \cdot B}{\|A\| \cdot \|B\|}$.
  - When vectors are L2-normalized ($\|A\|=1, \|B\|=1$), the denominator equals $1$.
  - Therefore: $\text{Cosine Similarity} \equiv \text{Inner Product}$! This eliminates expensive square root norm calculations at query time.

### 💻 Stage 2: Code & Line-by-Line Breakdown
```python
# 1. Encode with L2-Normalization
corpus_embeddings = embedding_model.encode(
    corpus_texts, batch_size=64, normalize_embeddings=True
)

# 2. FAISS Index Construction
embedding_dim = corpus_embeddings.shape[1]    # 384
faiss_index = faiss.IndexFlatIP(embedding_dim) # Flat Inner Product
faiss_index.add(corpus_embeddings.astype('float32'))

# 3. Microsecond Search
query_vec = embedding_model.encode([query], normalize_embeddings=True).astype('float32')
scores, indices = faiss_index.search(query_vec, top_k)
```
* `embedding_dim=384`: Pre-allocates contiguous C++ memory tables with exact 384-float row strides.
* `IndexFlatIP`: Computes SIMD-accelerated dot products in microseconds.

### 🎯 Stage 3: Top Interview Questions & Model Answers
* **Q1: Why did you choose `all-MiniLM-L6-v2` over heavier models like `all-mpnet-base-v2`?**
  * **Answer:** *"With a 384-dimensional vector size and ~80MB footprint, `all-MiniLM-L6-v2` delivers sub-second encoding on commodity CPU hardware and consumes 4x less memory in FAISS compared to 1536-d models, while retaining over 95% of the retrieval accuracy on the Massive Text Embedding Benchmark (MTEB)."*
* **Q2: Why use `IndexFlatIP` instead of `IndexFlatL2` in FAISS?**
  * **Answer:** *"By L2-normalizing all vectors during embedding generation, the Cosine Similarity mathematically reduces to the Inner Product ($A \cdot B$). `IndexFlatIP` computes dot products directly without calculating vector norms at runtime, maximizing query throughput."*

---

# Topic 6: Hybrid Search & Reciprocal Rank Fusion (RRF)

### 🧠 Stage 1: Deep Theory & Intuition
* **The Score Incompatibility Problem:**
  - BM25 scores are unbounded positive numbers ($0$ to $50+$) scaling with query length.
  - FAISS scores are bounded cosine values ($0.0$ to $1.0$).
  - Direct addition ($\text{BM25} + \text{FAISS}$) is impossible because BM25 completely obliterates FAISS. Min-max normalization is brittle because score ranges fluctuate across queries.
* **The RRF Solution (Cormack et al., 2009):**
  - RRF is completely **Score-Agnostic**! It discards raw scores and fuses documents purely based on their **Ordinal Ranks** (1st, 2nd, 3rd place).
  - Formula:
    $$\text{RRF\_Score}(d) = \sum_{m \in \{\text{BM25}, \text{Dense}\}} \frac{1}{k + r_m(d)}$$
* **The $k=60$ Smoothing Constant:**
  - If $k=0$, the jump from Rank 1 ($1/1 = 1.0$) to Rank 2 ($1/2 = 0.5$) is a brutal 50% drop! A single system's #1 pick would crush any combined candidates.
  - With $k=60$, Rank 1 ($1/61 \approx 0.01639$) and Rank 2 ($1/62 \approx 0.01613$) differ by only $0.00026$. This smooth decay allows documents performing consistently well across both systems to outrank a single model's noisy outlier.

### 💻 Stage 2: Code & Line-by-Line Breakdown
```python
def hybrid_search_rrf(query, top_k=5, candidate_pool=20, k_constant=60):
    bm25_candidates = search_bm25(query, top_k=candidate_pool)
    faiss_candidates = search_faiss(query, top_k=candidate_pool)
    
    rrf_scores = {}
    for item in bm25_candidates:
        idx, rank = item['corpus_idx'], item['rank']
        rrf_scores[idx] = rrf_scores.get(idx, 0.0) + (1.0 / (k_constant + rank))
        
    for item in faiss_candidates:
        idx, rank = item['corpus_idx'], item['rank']
        rrf_scores[idx] = rrf_scores.get(idx, 0.0) + (1.0 / (k_constant + rank))
        
    sorted_indices = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:top_k]
    return sorted_indices
```
* `candidate_pool=20`: Pulls top 20 from both channels (up to 40 candidate pool).
* `1.0 / (k_constant + rank)`: Accumulates rank-smoothed reciprocal scores. Documents appearing in both lists receive additive scores.

### 🎯 Stage 3: Top Interview Questions & Model Answers
* **Q1: Why is RRF superior to linear weighted score combination ($\alpha \cdot S_1 + (1-\alpha) \cdot S_2$)?**
  * **Answer:** *"Linear score combination requires delicate hyperparameter tuning ($\alpha$) and breaks when query length shifts BM25 scale. RRF is scale-invariant; it operates strictly on ordinal ranks, providing robust fusion across diverse query types without score calibration."*
* **Q2: What happens if $k$ is set to 0 in RRF?**
  * **Answer:** *"Without $k$, the penalty curve is too steep ($1.0 \to 0.5 \to 0.33$), meaning Rank 1 in one channel completely overpowers Rank 2 in both channels. Setting $k=60$ acts as a rank smoother, ensuring balanced, additive fusion."*

---

# Topic 7: Cross-Encoder Re-Ranking & Deep Diagnostics

### 🧠 Stage 1: Deep Theory & Intuition
* **Bi-Encoder vs. Cross-Encoder Architecture:**
  - **Bi-Encoder (FAISS):** Encodes Query and Document independently. Fast ($O(N)$), but cannot perform joint token-to-token cross-attention. Misses subtle differences (e.g. confusing *Credit Card* and *Debit Card*).
  - **Cross-Encoder (`ms-marco-MiniLM-L-6-v2`):** Feeds concatenated `[Query, Document]` into all transformer layers simultaneously. Full self-attention across every word pair yields state-of-the-art semantic precision.
* **The Two-Stage Retrieval Funnel:**
  - Stage 1 (High Recall): BM25 + FAISS + RRF filters 10,000 documents to Top 10 candidates in $< 50$ ms.
  - Stage 2 (High Precision): Cross-Encoder scores only those Top 10 candidates in $< 100$ ms.
* **The Intent vs. Entity Trade-Off (Diagnostic Finding):**
  - When querying *"unauthorized credit card fees disputed but denied"*, the Cross-Encoder scored a *Debit Card* narrative highest ($7.47$).
  - **Reason:** Trained on MS MARCO passage search, the model prioritized the **procedural grievance trajectory** (*unauthorized charge $\to$ dispute $\to$ denied*) over the payment instrument entity (*credit* vs *debit* card).
* **Cascading Error Propagation & Mitigation:**
  - Applying hard pre-filters based on upstream classifier predictions risks permanent recall loss if the classifier makes an error.
  - In production, mitigate via **Confidence Thresholding** (only filter if $> 90\%$ confident), **Top-2 Class Expansion**, or **Verified Account Metadata**.

### 💻 Stage 2: Code & Line-by-Line Breakdown
```python
# 1. Load Cross-Encoder
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# 2. Pairwise Joint Scoring
pairs = [[query, item['text']] for item in candidates]
cross_scores = reranker.predict(pairs)

# 3. Final Re-ranking
for idx, item in enumerate(candidates):
    item['cross_score'] = float(cross_scores[idx])
reranked = sorted(candidates, key=lambda x: x['cross_score'], reverse=True)[:final_top_k]
```
* `[query, item['text']]`: Concatenates query and document into paired input tensors.
* `reranker.predict(pairs)`: Computes full token-level cross-attention logits.

### 🎯 Stage 3: Top Interview Questions & Model Answers
* **Q1: Why not run a Cross-Encoder over the entire database directly?**
  * **Answer:** *"Cross-encoders require joint evaluation of every `[Query, Document]` pair, resulting in $O(N)$ transformer forward passes at runtime. For a 10,000+ document corpus, latency would exceed several seconds. The Two-Stage Funnel solves this: fast Bi-encoders/BM25 retrieve top candidates in milliseconds, and the Cross-Encoder re-ranks only the top 10-20, achieving SOTA accuracy within sub-second latency."*
* **Q2: What is Cascading Error in an ML pipeline, and how do you prevent it?**
  * **Answer:** *"Cascading error occurs when an upstream classifier's mistake permanently starves downstream systems (e.g. hard-filtering by a misclassified category prevents BM25/FAISS from ever finding the true category). We prevent this using confidence thresholding (only filter if confidence $> 90\%$), soft candidate expansion, or verified user account metadata."*

---
*(End of Master Study Guide)*
