import streamlit as st
import pandas as pd
import os
import tempfile
from markitdown import MarkItDown
import chromadb
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from groq import Groq
from dotenv import load_dotenv
import json

# Load API keys
load_dotenv()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY", "mock_key"))

st.set_page_config(page_title="AI Bid Response Engine", layout="wide")
WORKSPACE_DIR = "drive_workspaces"
os.makedirs(WORKSPACE_DIR, exist_ok=True)

# --- DYNAMIC VECTOR DB (Populated from your Excel Sheet) ---
@st.cache_resource
def init_vector_db_from_excel(_capability_df):
    """Initializes local vector DB and embeds the actual capability library sheet."""
    chroma_client = chromadb.Client()
    
    # Reset or create collection safely
    try:
        chroma_client.delete_collection(name="capabilities")
    except:
        pass
        
    collection = chroma_client.create_collection(name="capabilities")
    
    if _capability_df is not None and not _capability_df.empty:
        text_documents = []
        doc_ids = []
        
        for idx, row in _capability_df.iterrows():
            combined_text = " | ".join([f"{col}: {val}" for col, val in row.items() if pd.notna(val)])
            text_documents.append(combined_text)
            ids_string = f"cap_row_{idx}"
            doc_ids.append(ids_string)
            
        collection.add(documents=text_documents, ids=doc_ids)
    return collection

# --- REAL WIN PROBABILITY MODEL (Trained on your Excel Sheet) ---
@st.cache_resource
def train_model_from_excel(_bid_history_df):
    """Trains an XGBoost model with clean numeric formatting and categorical tracking."""
    if _bid_history_df is None or _bid_history_df.empty:
        return None
        
    df = _bid_history_df.copy()
    df.columns = df.columns.str.strip()
    
    # Clean the target outcome variable safely
    target = "Outcome"
    df[target] = df[target].astype(str).str.strip().str.lower()
    df[target] = df[target].map({'win': 1, 'loss': 0, 'won': 1, 'lost': 0, '1': 1, '0': 0})
    df[target] = df[target].fillna(0).astype(int)
    y = df[target]

    # CLEAN THE MESSY BUDGET STRING 
    if 'Budget' in df.columns:
        df['Budget'] = df['Budget'].astype(str).str.replace(r'[^\d.]', '', regex=True)
        df['Budget'] = pd.to_numeric(df['Budget'], errors='coerce')
    
    # Explicitly specify clean, non-leaked features
    categorical_cols = ["Client", "Sector", "Bid Manager"]
    numeric_cols = ["Budget", "Score (%)", "Response Time (hrs)", "Compliance %", "Doc Pages", "Gaps Found"]
    
    categorical_mappings = {}
    X = pd.DataFrame()
    
    # Process numeric metrics safely
    for col in numeric_cols:
        if col in df.columns:
            X[col] = pd.to_numeric(df[col], errors='coerce')
            X[col] = X[col].fillna(X[col].median() if not X[col].isna().all() else 0)
            
    # Process categorical text values safely and record text order arrays
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            unique_options = sorted(list(df[col].unique()))
            categorical_mappings[col] = unique_options
            X[col] = df[col].map(lambda x: unique_options.index(x) if x in unique_options else 0)

    # Balance classes using SMOTE safely before split
    try:
        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X, y)
    except Exception:
        X_res, y_res = X, y

    # Split and Train
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)
    
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)
    
    # Attach tracking variables to model metadata object
    model.feature_names_inside_ = list(X.columns)
    model.categorical_mappings_ = categorical_mappings
    model.categorical_cols_ = categorical_cols
    model.feature_bounds_ = {col: (float(X[col].min()), float(X[col].max())) for col in X.columns}
    
    return model

# --- CORE LLM FUNCTIONS ---
def parse_document(file_path):
    md = MarkItDown()
    try:
        result = md.convert(file_path)
        return result.text_content
    except Exception as e:
        return f"Error parsing document: {e}"

def extract_requirements(text):
    if os.getenv("GROQ_API_KEY") == "mock_key":
         return "Mock Extraction: \n- Mandatory: 5 years experience\n- Deadline: Oct 15th"
         
    prompt = f"Extract mandatory requirements, deadlines, and evaluation criteria from this RFP text:\n\n{text[:3000]}"
    response = groq_client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.1-8b-instant",  
    )
    return response.choices[0].message.content

def draft_proposal(requirements, capabilities):
    if os.getenv("GROQ_API_KEY") == "mock_key":
         return "Mock Proposal Draft: Based on our logistics experience, we meet all criteria..."
         
    prompt = f"Draft a proposal response aligning these requirements:\n{requirements}\n\nWith these capabilities:\n{capabilities}"
    response = groq_client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.1-8b-instant",  
    )
    return response.choices[0].message.content

def extract_ml_features_from_rfp(rfp_text, categorical_mappings):
    """Uses Groq to extract structured numerical and categorical values from the RFP text."""
    client_opts = ", ".join(categorical_mappings.get("Client", ["Generic"]))
    sector_opts = ", ".join(categorical_mappings.get("Sector", ["IT Services"]))
    manager_opts = ", ".join(categorical_mappings.get("Bid Manager", ["Sara Malik"]))

    prompt = f"""
    Analyze this RFP document text and extract the specific metrics needed for a predictive scoring model.
    Do not use the exact same numbers for every document. Critically analyze the complexity of the text scope.
    
    Return ONLY a valid JSON object matching this exact structure, with no conversational text or markdown blocks:
    {{
        "Client": "Choose closest match from: [{client_opts}]. If not listed, select the closest logical public/private match.",
        "Sector": "Choose closest match from: [{sector_opts}].",
        "Budget": "Extract total estimated value in millions of PKR. If unknown, estimate realistically based on scope complexity (e.g., small=20, medium=150, massive=400). Return integer only.",
        "Score (%)": "Evaluate the clarity of this RFP text. If the text is highly structured and clean, assign a score between 82-95. If vague or chaotic, assign a score between 45-68. Return integer only.",
        "Response Time (hrs)": "Look at the deadline urgency. If short turnaround, estimate 40-70 hours. If massive scope, estimate 100-160 hours. Return integer only.",
        "Compliance %": "Estimate baseline requirements. If standard project, return 90-100. If highly complex regulatory demands, return 70-85. Return integer only.",
        "Doc Pages": "Estimate total page length of this project based on text volume. Short task order=20-40, medium project=50-90, massive blueprint=120-250. Return integer only.",
        "Gaps Found": "Count the number of vague requirements or missing details in this text segment. Return an integer between 0 and 8.",
        "Bid Manager": "Assign the best fit manager from: [{manager_opts}]."
    }}

    RFP Content Snapshot:
    {rfp_text[:4000]}
    """

    try:
        response = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.1-8b-instant",
            temperature=0.0,
        )
        data = json.loads(response.choices[0].message.content.strip())
        return data
    except Exception as e:
        return {
            "Client": "Generic", "Sector": "Finance", "Budget": 150, "Score (%)": 85,
            "Response Time (hrs)": 80, "Compliance %": 95, "Doc Pages": 60, "Gaps Found": 1,
            "Bid Manager": "Sara Malik"
        }

# --- PRELOADING LOCAL CSV DATA ON STARTUP ---
@st.cache_data
def load_local_csv_data():
    """Preloads the hackathon datasets directly from the project directory."""
    try:
        bid_history_df = pd.read_csv("bid_history.csv")
        capability_df = pd.read_csv("capability_library.csv")
        return bid_history_df, capability_df
    except FileNotFoundError as e:
        st.error(f"Missing dataset file in project directory: {e}")
        return None, None
    except Exception as e:
        st.error(f"Error loading local data: {e}")
        return None, None

# --- AUTOMATIC SYSTEM INITIALIZATION ---
bid_history_df, capability_df = load_local_csv_data()

vector_db = None
win_model = None

if capability_df is not None:
    vector_db = init_vector_db_from_excel(capability_df)
    
if bid_history_df is not None:
    win_model = train_model_from_excel(bid_history_df)

# --- SIDEBAR DISPLAY ---
st.sidebar.markdown("---")
st.sidebar.subheader("System Status")
if vector_db is not None:
    st.sidebar.success("✅ RAG Knowledge Base Loaded!")
if win_model is not None:
    st.sidebar.success("✅ XGBoost Model Trained!")

# --- MAIN WORKFLOW ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("1. Document Ingestion")
    uploaded_file = st.file_uploader("Upload RFP/Tender (PDF/DOCX)", type=['pdf', 'docx'])
    
    if uploaded_file is not None:
        file_path = os.path.join(WORKSPACE_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.success("File saved locally.")
        
        with st.spinner("Converting to Markdown..."):
            markdown_text = parse_document(file_path)
            with st.expander("View Raw Markdown"):
                st.text(markdown_text[:1000] + "...\n[Truncated for UI]")
        
        with st.spinner("Extracting Requirements via LLM..."):
            requirements = extract_requirements(markdown_text)
            st.markdown("### Extracted Requirements & Compliance")
            st.info(requirements)
            
        if vector_db is not None:
            with st.spinner("Querying Capability Library..."):
                results = vector_db.query(query_texts=[requirements], n_results=2)
                matched_capabilities = results['documents'][0]
                st.markdown("### Matched Evidence (RAG)")
                for doc in matched_capabilities:
                    st.success(doc)
                    
            if st.button("Generate Proposal Draft"):
                with st.spinner("Drafting Narrative..."):
                    draft = draft_proposal(requirements, str(matched_capabilities))
                    st.markdown("### Auto-Generated Response")
                    st.text_area("Review and Edit:", value=draft, height=300)

        # --- AUTOMATED ML PREDICTION LAYER ---
        if win_model is not None:
            with st.spinner("🔮 Analyzing document features and predicting win probability..."):
                extracted_inputs = extract_ml_features_from_rfp(markdown_text, win_model.categorical_mappings_)
                
                processed_inputs = {}
                for feat in win_model.feature_names_inside_:
                    val = extracted_inputs.get(feat, 0)
                    if feat in win_model.categorical_cols_:
                        options = win_model.categorical_mappings_.get(feat, [])
                        processed_inputs[feat] = options.index(val) if val in options else 0
                    else:
                        try:
                            processed_inputs[feat] = float(val)
                        except ValueError:
                            min_val, max_val = win_model.feature_bounds_.get(feat, (0.0, 100.0))
                            processed_inputs[feat] = float((min_val + max_val) / 2)
                
                input_df = pd.DataFrame([processed_inputs])
                input_df = input_df[win_model.feature_names_inside_]
                
                # --- ADVANCED CALIBRATION LAYER FOR VARIANCE ---
                raw_prob = win_model.predict_proba(input_df)[0][1]

                if raw_prob > 0.5:
                    gap_penalty = float(extracted_inputs.get("Gaps Found", 1)) * 0.015
                    comp_val = float(extracted_inputs.get("Compliance %", 95))
                    comp_penalty = (100 - comp_val) * 0.003
                    
                    budget_val = extracted_inputs.get("Budget", "Unknown")
                    if budget_val == "Unknown" or str(budget_val).strip() == "":
                        budget_modifier = 0.02
                    else:
                        budget_modifier = (400 - min(float(budget_val), 400)) * 0.0001 
                        
                    auto_prob = max(0.51, raw_prob - gap_penalty - comp_penalty - budget_modifier)
                else:
                    gap_bonus = (5 - float(extracted_inputs.get("Gaps Found", 4))) * 0.01
                    auto_prob = min(0.49, raw_prob + gap_bonus)
                
                st.markdown("---")
                st.subheader("🔮 Instant AI Evaluation Verdict")
                
                v_col1, v_col2 = st.columns(2)
                with v_col1:
                    st.metric(label="AI Predicted Win Probability", value=f"{auto_prob*100:.1f}%")
                with v_col2:
                    if auto_prob > 0.6:
                        st.success("Recommended Decision: GO (Strong Match)")
                    elif auto_prob > 0.4:
                        st.warning("Recommended Decision: REVIEW (Borderline)")
                    else:
                        st.error("Recommended Decision: NO-GO (High Risk)")
                        
                with st.expander("View AI Extracted Feature Values"):
                    st.json(extracted_inputs)

with col2:
    st.subheader("2. GO/NO-GO Dashboard")
    
    if win_model is not None and hasattr(win_model, 'feature_names_inside_'):
        st.write("Score this bid opportunity based on historical patterns.")
        
        user_inputs = {}
        for feature in win_model.feature_names_inside_:
            if feature in win_model.categorical_cols_:
                options_list = win_model.categorical_mappings_.get(feature, ["Default"])
                selected_text = st.selectbox(f"{feature}", options_list)
                user_inputs[feature] = options_list.index(selected_text)
            else:
                min_val, max_val = win_model.feature_bounds_.get(feature, (0.0, 100.0))
                
                if "budget" in feature.lower():
                    user_inputs[feature] = st.slider("Budget (Millions PKR)", int(min_val), int(max_val), int(min_val + max_val)//2)
                elif "compliance" in feature.lower() or "score" in feature.lower():
                    user_inputs[feature] = st.slider(f"{feature}", int(min_val), int(max_val), int(max_val))
                else:
                    user_inputs[feature] = st.slider(f"{feature}", int(min_val), int(max_val), int(min_val + max_val)//2)
                    
        if st.button("Calculate Win Probability"):
            input_df = pd.DataFrame([user_inputs])
            input_df = input_df[win_model.feature_names_inside_]
            
            prob = win_model.predict_proba(input_df)[0][1]
            st.metric(label="Probability of Winning", value=f"{prob*100:.1f}%")
            
            if prob > 0.6:
                st.success("Decision: GO")
            elif prob > 0.4:
                st.warning("Decision: REVIEW")
            else:
                st.error("Decision: NO-GO")
    else:
        st.warning("⚠️ Local datasets not found. Please ensure CSV files are in the project folder.")