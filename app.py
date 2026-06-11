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

# Load API keys
load_dotenv()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY", "mock_key"))

st.set_page_config(page_title="AI Bid Response Engine", layout="wide")
WORKSPACE_DIR = "drive_workspaces"
os.makedirs(WORKSPACE_DIR, exist_ok=True)

# --- PRELOADING LOCAL CSV DATA ON STARTUP ---

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
        # Combine row columns into a singular text string for semantic search matching
        text_documents = []
        doc_ids = []
        
        for idx, row in _capability_df.iterrows():
            # Merges all row context into a block of text the LLM can easily read
            combined_text = " | ".join([f"{col}: {val}" for col, val in row.items() if pd.notna(val)])
            text_documents.append(combined_text)
            ids_string = f"cap_row_{idx}"
            doc_ids.append(ids_string)
            
        collection.add(documents=text_documents, ids=doc_ids)
    return collection

# --- REAL WIN PROBABILITY MODEL (Trained on your Excel Sheet) ---
@st.cache_resource
def train_model_from_excel(_bid_history_df):
    """Trains an XGBoost model on your actual historical bid history sheet with auto-encoding."""
    if _bid_history_df is None or _bid_history_df.empty:
        return None
        
    df = _bid_history_df.copy()
    
    # 1. Identify and extract the target column
    target_col = [col for col in df.columns if 'outcome' in col.lower() or 'win' in col.lower() or 'status' in col.lower()]
    if target_col:
        target = target_col[0]
    else:
        target = df.columns[-1] 
        
    # Standardize target column to binary numeric values 1 and 0
    if df[target].dtype == 'object':
        df[target] = df[target].astype(str).str.lower().map({'win': 1, 'loss': 0, 'won': 1, 'lost': 0, '1': 1, '0': 0})
        df[target] = df[target].fillna(0).astype(int)

    y = df[target]
    X_raw = df.drop(columns=[target])
    
    # 2. AUTOMATIC CATEGORICAL TO NUMERIC ENCODING LAYER
    X = pd.DataFrame()
    for col in X_raw.columns:
        # Skip completely unique text identifiers like "Bid ID" or "Client Name" to avoid noise
        if "id" in col.lower() or "name" in col.lower() or "summary" in col.lower():
            continue
            
        # If it's already numeric, keep it
        if pd.api.types.is_numeric_dtype(X_raw[col]):
            X[col] = X_raw[col].fillna(X_raw[col].median() if not X_raw[col].isna().all() else 0)
        
        # If it's text data (e.g., "High", "Medium", "Low" or "IT", "Construction"), encode it
        elif pd.api.types.is_object_dtype(X_raw[col]):
            # Convert text categories to distinct numerical integers (0, 1, 2...)
            X[col] = X_raw[col].astype('category').cat.codes
            
    if X.empty:
        st.error("XGBoost error: Could not extract features from 'The bid history' sheet.")
        return None

    # 3. Handle class imbalance natively using SMOTE
    try:
        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X, y)
    except Exception:
        X_res, y_res = X, y
        
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)
    
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)
    
    # Save feature names for UI generation
    model.feature_names_inside_ = list(X.columns)
    
    # Save the original min/max value bounds to calibrate UI sliders intelligently
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

# --- PRELOADING LOCAL CSV DATA ON STARTUP ---
@st.cache_data
def load_local_csv_data():
    """Preloads the hackathon datasets directly from the project directory."""
    try:
        # Looking for the files locally in the repository folder
        bid_history_df = pd.read_csv("bid_history.csv")
        capability_df = pd.read_csv("capability_library.csv")
        return bid_history_df, capability_df
    except FileNotFoundError as e:
        st.error(f"Missing dataset file in project directory: {e}")
        return None, None
    except Exception as e:
        st.error(f"Error loading local data: {e}")
        return None, None

@st.cache_resource
def train_model_preloaded(df):
    """Trains an XGBoost model by explicitly targeting the core win/loss outcome field."""
    if df is None or df.empty:
        return None
        
    df = df.copy()
    
    # 1. Clean up column whitespace and casing
    df.columns = df.columns.str.strip()
    
    # --- FIXED: SPECIFIC EXPLICIT TARGETING ---
    # We look for common win/loss target labels. Change 'Outcome' to your exact column name if different.
    potential_targets = ['Outcome', 'Win/Loss', 'Status', 'Win', 'Result']
    target = None
    
    for t in potential_targets:
        match = [col for col in df.columns if t.lower() in col.lower()]
        if match:
            target = match[0]
            break
            
    # If no keywords match, explicitly look for a column that contains binary data, or default safely
    if not target:
        target = df.columns[-1] # Fallback to last column if all else fails
        
    # Remove any stray rows where the data row itself accidently duplicates the header name string
    df = df[df[target] != 'Submission Date']
    df = df[df[target] != target]
    
    # Safely convert textual outcomes ('Win', 'Loss') to clean binary numbers
    df[target] = df[target].astype(str).str.strip().str.lower()
    df[target] = df[target].map({
        'win': 1, 'loss': 0, 'won': 1, 'lost': 0, 
        '1': 1, '0': 0, '1.0': 1, '0.0': 0, 
        'pass': 1, 'fail': 0
    })
    
    # Handle any empty cells by filling them as 0 (Loss)
    df[target] = df[target].fillna(0).astype(int)
    # ------------------------------------------

    # Define the exact features required by the win-probability heuristics
    # Lowercasing search to guarantee column match regardless of CSV casing
    df.columns = df.columns.str.lower()
    target_lower = target.lower()
    
    feature_cols = ['budget_alignment', 'competitor_presence', 'past_domain_win_rate']
    actual_features = []
    
    for feat in feature_cols:
        match = [col for col in df.columns if feat[:5] in col and col != target_lower]
        if match:
            actual_features.append(match[0])
            
    if not actual_features:
        # Fallback: select the first 3 numeric columns that aren't our target
        actual_features = [col for col in df.select_dtypes(include=['number']).columns if col != target_lower][:3]
        
    if not actual_features:
        st.error("XGBoost error: System could not identify clean numeric feature columns in 'bid_history.csv' to train on.")
        return None
        
    X = df[actual_features].fillna(0)
    y = df[target_lower]
    
    # Balance classes using SMOTE
    try:
        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X, y)
    except Exception:
        X_res, y_res = X, y
        
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)
    
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)
    
    model.feature_names_inside_ = actual_features
    model.feature_bounds_ = {}
    for col in X.columns:
        min_v = float(X[col].min())
        max_v = float(X[col].max())
        if min_v == max_v:
            min_v, max_v = (0.0, 1.0) if min_v <= 1.0 else (0.0, 100.0)
        model.feature_bounds_[col] = (min_v, max_v)
        
    return model

# --- AUTOMATIC SYSTEM INITIALIZATION ---
# This runs instantly when the page loads up—no upload blocks required!
bid_history_df, capability_df = load_local_csv_data()

vector_db = None
win_model = None

if capability_df is not None:
    vector_db = init_vector_db_from_excel(capability_df)
    
if bid_history_df is not None:
    win_model = train_model_preloaded(bid_history_df)

# --- SIDEBAR DISPLAY ---
# Removed the file uploaders completely to keep the user view uncluttered
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

with col2:
    st.subheader("2. GO/NO-GO Dashboard")
    
    if win_model is not None:
        st.write("Score this bid opportunity based on historical patterns.")
        
        user_inputs = {}
        for feature in win_model.feature_names_inside_:
            min_val, max_val = win_model.feature_bounds_.get(feature, (0.0, 100.0))
            
            # Form clean labels for the UI
            clean_label = feature.replace('_', ' ').title()
            
            if max_val <= 1.0:
                user_inputs[feature] = st.slider(clean_label, float(min_val), float(max_val), float(min_val + max_val)/2)
            else:
                user_inputs[feature] = st.slider(clean_label, int(min_val), int(max_val), int(min_val + max_val)//2)
                
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