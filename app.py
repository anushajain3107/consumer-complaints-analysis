"""
Consumer Complaint Intelligence - Enterprise Streamlit Dashboard
Interactive UI for ML Classification, Hybrid Precedent Search, and Evidence-Grounded RAG Triage.
"""

import streamlit as st
import requests
import json
import time

# =====================================================================
# PAGE CONFIGURATION & STYLING
# =====================================================================
st.set_page_config(
    page_title="CFPB Complaint Intelligence",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Financial-Grade Modern CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    .badge-approved {
        background-color: #DCFCE7;
        color: #15803D;
        font-weight: 700;
        padding: 6px 14px;
        border-radius: 20px;
        border: 1px solid #86EFAC;
        display: inline-block;
    }
    .badge-flagged {
        background-color: #FEE2E2;
        color: #B91C1C;
        font-weight: 700;
        padding: 6px 14px;
        border-radius: 20px;
        border: 1px solid #FCA5A5;
        display: inline-block;
    }
    .precedent-card {
        background-color: #FFFFFF;
        border-left: 4px solid #3B82F6;
        border-radius: 6px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        padding: 14px 18px;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)

API_BASE_URL = "http://127.0.0.1:8000"

# =====================================================================
# SIDEBAR: SYSTEM HEALTH & ARCHITECTURE
# =====================================================================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/bank-building.png", width=64)
    st.markdown("### **System Status**")
    
    # Check FastAPI Backend Health
    backend_online = False
    try:
        health_res = requests.get(f"{API_BASE_URL}/health", timeout=2)
        if health_res.status_code == 200:
            health_data = health_res.json()
            backend_online = True
            st.success("🟢 **Backend API Online**")
            st.caption(f"**Host:** `{API_BASE_URL}`")
            st.caption(f"**Knowledge Base:** `{health_data.get('knowledge_base_size', 0):,}` complaints")
            st.caption(f"**Models Cached:** `{len(health_data.get('models_loaded', []))}` in RAM")
    except Exception:
        st.error("🔴 **Backend API Offline**")
        st.warning("FastAPI server nahi chal raha hai.\n\nTerminal mein run karein:\n`py -3.11 api.py`")

    st.markdown("---")
    st.markdown("### **Pipeline Specifications**")
    st.markdown("""
    - **Classifier:** LinearSVC (F1: **86.03%**)
    - **Lexical Search:** BM25Okapi ($k_1=1.5, b=0.75$)
    - **Vector Search:** `all-MiniLM-L6-v2` + FAISS IndexFlatIP
    - **Fusion:** Reciprocal Rank Fusion ($k=60$)
    - **Re-ranker:** Cross-Encoder `ms-marco-MiniLM-L-6-v2`
    - **Guardrails:** PII Sanitization + Citation Faithfulness
    """)
    st.markdown("---")
    st.caption("CFPB Financial Regulatory Intelligence v1.0")

# =====================================================================
# MAIN HEADER
# =====================================================================
st.markdown('<div class="main-header">⚖️ Consumer Complaint Intelligence System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Production AI Platform for Automated Regulatory Triage, Hybrid Precedent Retrieval, and Evidence-Grounded Synthesis</div>', unsafe_allow_html=True)

# 3 Main Tabs
tab1, tab2, tab3 = st.tabs([
    "⚡ 1. Live Classification",
    "🔍 2. Hybrid Precedent Search",
    "🛡️ 3. Evidence-Grounded RAG Triage"
])

# =====================================================================
# TAB 1: LIVE COMPLAINT CLASSIFIER
# =====================================================================
with tab1:
    st.markdown("### **Machine Learning Complaint Classifier**")
    st.write("Champion Sublinear TF-IDF + LinearSVC model running sub-millisecond inference across 7 consolidated CFPB categories.")

    col1, col2 = st.columns([2, 1])

    with col1:
        sample_complaints = {
            "Select a pre-loaded sample...": "",
            "Credit Card Fraud / Dispute": "I discovered three unauthorized transactions totaling $450 on my credit card statement from an unknown vendor. I immediately contacted customer care to dispute the charges, but they refused to issue a provisional credit.",
            "Debt Collection Harassment": "A collection agency has been repeatedly calling my workplace and threatening legal garnishment without providing any written debt validation letter as required under FDCPA regulations.",
            "Mortgage Escrow Miscalculation": "The mortgage servicing company failed to disburse property taxes from my escrow account, resulting in local tax penalties and threats of a tax lien on my primary residence.",
            "Student Loan Repayment Servicing": "I submitted my documentation for the Income-Driven Repayment (IDR) plan over four months ago. The servicer has placed my account into improper administrative forbearance and charged capitalization interest."
        }
        
        selected_sample = st.selectbox("💡 Quick Fill with Precedent Sample:", list(sample_complaints.keys()))
        default_text = sample_complaints[selected_sample] if selected_sample != "Select a pre-loaded sample..." else ""

        narrative_input = st.text_area(
            "Consumer Complaint Narrative:",
            value=default_text,
            height=160,
            placeholder="Type or paste a consumer grievance narrative here..."
        )

        classify_btn = st.button("🚀 Classify Narrative", type="primary", use_container_width=True)

    with col2:
        st.markdown("#### **Inference Architecture**")
        st.info("""
        - **Vocabulary:** 10,000 sublinear TF-IDF unigrams/bigrams
        - **Model:** LinearSVC (Max-margin hyperplane)
        - **Confidence Engine:** Softmax-calibrated decision margins
        - **Target Latency:** < 25ms
        """)

    if classify_btn:
        if not narrative_input.strip():
            st.warning("⚠️ Please enter a complaint narrative to classify.")
        elif not backend_online:
            st.error("❌ FastAPI backend is offline! Please start `api.py` first.")
        else:
            with st.spinner("Classifying complaint..."):
                start_time = time.time()
                try:
                    res = requests.post(f"{API_BASE_URL}/classify", json={"text": narrative_input}, timeout=5)
                    elapsed_ms = (time.time() - start_time) * 1000
                    
                    if res.status_code == 200:
                        data = res.json()
                        st.success(f"Classification completed in **{elapsed_ms:.1f} ms**!")
                        
                        m1, m2, m3 = st.columns(3)
                        with m1:
                            st.metric(label="Predicted CFPB Category", value=data["predicted_category"])
                        with m2:
                            confidence_pct = data["confidence_score"] * 100
                            st.metric(label="Confidence Score", value=f"{confidence_pct:.2f}%")
                        with m3:
                            st.metric(label="Inference Latency", value=f"{elapsed_ms:.1f} ms")
                            
                        st.progress(data["confidence_score"], text=f"Model Certainty: {confidence_pct:.1f}%")
                    else:
                        st.error(f"Error {res.status_code}: {res.text}")
                except Exception as e:
                    st.error(f"Request failed: {str(e)}")

# =====================================================================
# TAB 2: HYBRID PRECEDENT SEARCH
# =====================================================================
with tab2:
    st.markdown("### **2-Stage Hybrid Legal Precedent Search Engine**")
    st.write("Combines Lexical Precision (BM25) with Dense Semantic Embeddings (FAISS) via Reciprocal Rank Fusion ($k=60$) and Cross-Encoder Re-Ranking.")

    col_q1, col_q2 = st.columns([3, 1])
    with col_q1:
        sample_queries = {
            "Select sample search query...": "",
            "Debt Collection: Phone harassment without validation": "debt collector calling workplace repeatedly threatening legal action without debt validation letter",
            "Credit Card: Unauthorized fraud transactions": "unauthorized fraudulent transactions charged on credit card customer service refused refund",
            "Mortgage: Escrow tax payment deficiency": "mortgage servicer failed to pay property taxes from escrow account causing tax penalties",
            "Credit Reporting: Inaccurate derogatory marks": "credit bureau refusing to remove outdated inaccurate collection account after dispute"
        }
        selected_query = st.selectbox("💡 Sample Precedent Query:", list(sample_queries.keys()))
        default_query_text = sample_queries[selected_query] if selected_query != "Select sample search query..." else ""
        
        search_query = st.text_input(
            "Legal / Regulatory Search Query:",
            value=default_query_text,
            placeholder="e.g. unauthorized transactions on checking account overdraft fees"
        )
    with col_q2:
        top_k = st.slider("Top Precedents (K):", min_value=1, max_value=8, value=3)

    search_btn = st.button("🔎 Retrieve Precedents", type="primary", use_container_width=True)

    if search_btn:
        if not search_query.strip():
            st.warning("⚠️ Please enter a search query.")
        elif not backend_online:
            st.error("❌ FastAPI backend is offline! Please start `api.py` first.")
        else:
            with st.spinner("Executing 2-Stage Hybrid Retrieval & Cross-Encoder Re-ranking..."):
                start_time = time.time()
                try:
                    res = requests.post(f"{API_BASE_URL}/retrieve", json={"query": search_query, "top_k": top_k}, timeout=10)
                    elapsed_ms = (time.time() - start_time) * 1000
                    
                    if res.status_code == 200:
                        data = res.json()
                        st.success(f"Retrieved **{data['total_retrieved']}** historical precedents in **{elapsed_ms:.1f} ms**!")
                        
                        for prec in data["precedents"]:
                            score_color = "#15803D" if prec["relevance_score"] > 0 else "#B91C1C"
                            st.markdown(f"""
                            <div class="precedent-card">
                                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                                    <span style="font-weight: 700; color: #1E3A8A; font-size: 1.1rem;">{prec['precedent_id']} — {prec['product']}</span>
                                    <span style="background-color: #EFF6FF; color: {score_color}; font-weight: 700; padding: 4px 10px; border-radius: 12px; font-size: 0.85rem;">
                                        Cross-Encoder Score: {prec['relevance_score']:.4f}
                                    </span>
                                </div>
                                <p style="color: #374151; font-size: 0.95rem; line-height: 1.5; margin: 0;">
                                    "{prec['text_snippet']}"
                                </p>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.error(f"Error {res.status_code}: {res.text}")
                except Exception as e:
                    st.error(f"Request failed: {str(e)}")

# =====================================================================
# TAB 3: EVIDENCE-GROUNDED RAG TRIAGE & GUARDRAILS
# =====================================================================
with tab3:
    st.markdown("### **Evidence-Grounded RAG Assistant & Compliance Guardrails**")
    st.write("Full Enterprise Pipeline: Automated PII Scrubbing $\\to$ Precedent Retrieval $\\to$ Structured Dynamic Synthesis $\\to$ Hallucination Verification.")

    sample_rag_inputs = {
        "Select a sample grievance with PII...": "",
        "Credit Card Billing Dispute (with SSN & Account #)": "My name is John Doe and my SSN is 123-45-6789. I noticed an unauthorized charge of $850 on my Citibank credit card ending in 453289012345. I called phone number 800-555-0199 but customer service refused to reverse it or honor my billing dispute.",
        "Debt Collection Harassment (with Phone & Account)": "A debt collector from ABC Recovery has been calling my mobile 555-234-5678 four times a day regarding an old medical bill account 9876543210. They threatened to notify my employer and seize my wages without any debt validation notice.",
        "Mortgage Loan Modification (with SSN)": "I applied for a loan modification under statutory guidelines. My social security number is 987-65-4321 and my loan number is 1122334455. The servicer initiated foreclosure proceedings while my modification application was still under active review."
    }
    
    selected_rag = st.selectbox("💡 Sample Grievance (Contains Raw PII):", list(sample_rag_inputs.keys()))
    default_rag_text = sample_rag_inputs[selected_rag] if selected_rag != "Select a sample grievance with PII..." else ""

    rag_input_text = st.text_area(
        "Raw Consumer Grievance:",
        value=default_rag_text,
        height=140,
        placeholder="Enter un-redacted consumer grievance text (our PII engine will automatically scrub sensitive identifiers)..."
    )

    col_triage1, col_triage2 = st.columns([1, 1])
    with col_triage1:
        triage_k = st.slider("Evidence Precedents to Retrieve:", min_value=1, max_value=5, value=3)
    with col_triage2:
        st.write("")
        st.write("")
        triage_btn = st.button("🛡️ Execute Regulatory Triage & Audit", type="primary", use_container_width=True)

    if triage_btn:
        if not rag_input_text.strip():
            st.warning("⚠️ Please provide a consumer grievance narrative.")
        elif not backend_online:
            st.error("❌ FastAPI backend is offline! Please start `api.py` first.")
        else:
            with st.spinner("Sanitizing PII, Retrieving Grounded Precedents, Synthesizing Report & Verifying Citations..."):
                start_time = time.time()
                try:
                    payload = {"complaint_text": rag_input_text, "top_k": triage_k}
                    res = requests.post(f"{API_BASE_URL}/rag-triage", json=payload, timeout=15)
                    elapsed_ms = (time.time() - start_time) * 1000
                    
                    if res.status_code == 200:
                        data = res.json()
                        st.success(f"Triage Pipeline Completed in **{elapsed_ms:.1f} ms**!")
                        
                        # --- Section A: PII Audit ---
                        st.markdown("#### **1. Automated PII Sanitization Audit**")
                        pii = data["pii_redactions_made"]
                        c_ssn, c_acc, c_phone = st.columns(3)
                        with c_ssn:
                            st.metric("SSNs Redacted", pii.get("ssn_redacted", 0))
                        with c_acc:
                            st.metric("Account Numbers Redacted", pii.get("account_numbers_redacted", 0))
                        with c_phone:
                            st.metric("Phone Numbers Redacted", pii.get("phone_numbers_redacted", 0))
                            
                        with st.expander("👁️ View Sanitized Prompt Transmitted to Engine"):
                            st.code(data["sanitized_input"], language="text")

                        # --- Section B: Generated Triage Report ---
                        st.markdown("#### **2. Evidence-Grounded Triage Report**")
                        st.markdown(data["generated_triage_report"])

                        # --- Section C: Guardrail Certificate ---
                        st.markdown("#### **3. Programmatic Citation & Hallucination Guardrail Certificate**")
                        status_class = "badge-approved" if data["is_compliant"] else "badge-flagged"
                        status_text = data["guardrail_status"]
                        
                        st.markdown(f'<div class="{status_class}">🛡️ {status_text}</div>', unsafe_allow_html=True)
                        st.write("")
                        
                        g1, g2 = st.columns(2)
                        with g1:
                            st.metric("Faithfulness Score", f"{data['faithfulness_score']:.1f}%")
                        with g2:
                            st.metric("Regulatory Hallucinations", "0 Found" if data["is_compliant"] else "Flagged Unverified Precedents")
                            
                        st.progress(min(data["faithfulness_score"] / 100.0, 1.0))

                    else:
                        st.error(f"Error {res.status_code}: {res.text}")
                except Exception as e:
                    st.error(f"Request failed: {str(e)}")
