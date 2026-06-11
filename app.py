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

# --- NEW EXCEL DATA LOADING LAYER ---
@st.cache_data
def load_excel_data(file_path):
    """Loads both sheets from your hackathon workbook."""
    try:
        bid_history_df = pd.read_excel(file_path, sheet_name="PS1 – Bid History")
        capability_df = pd.read_excel(file_path, sheet_name="PS1 – Capability Library")
        return bid_history_df, capability_df
    except Exception as e:
        st.error(f"Error loading Excel file sheets: {e}")
        return None, None

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
    """Trains an XGBoost model on your actual historical bid history sheet."""
    if _bid_history_df is None or _bid_history_df.empty:
        return None
        
    df = _bid_history_df.copy()
    
    # Target column extraction (e.g., 'outcome', 'Status', 'Win/Loss' - adjust string to match your column precisely)
    # Finding the outcome column dynamically or fallback to the last column
    target_col = [col for col in df.columns if 'outcome' in col.lower() or 'win' in col.lower()]
    if target_col:
        target = target_col[0]
    else:
        target = df.columns[-1] 
        
    # Standardizing target column to binary numeric values 1 and 0
    if df[target].dtype == 'object':
        df[target] = df[target].astype(str).str.lower().map({'win': 1, 'loss': 0, 'won': 1, 'lost': 0, '1': 1, '0': 0})
        df[target] = df[target].fillna(0).astype(int)

    # Feature selection: isolate only numeric features for XGBoost compatibility
    X = df.drop(columns=[target]).select_dtypes(include=['number'])
    y = df[target]
    
    if X.empty:
        st.error("XGBoost error: No numeric feature columns found in 'The bid history' sheet to train on.")
        return None

    # Handle class imbalance natively using SMOTE (Same architecture as customer churn prediction)
    try:
        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X, y)
    except Exception:
        # Fallback if dataset size is too small for default SMOTE neighbors
        X_res, y_res = X, y
        
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)
    
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)
    
    # Store the expected feature sequence names to validate against user sliders later
    model.feature_names_inside_ = list(X.columns)
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

# --- UI & STATE MANAGEMENT ---
if 'workspaces' not in st.session_state:
    st.session_state.workspaces = ["Default RFP"]
if 'current_workspace' not in st.session_state:
    st.session_state.current_workspace = "Default RFP"

st.sidebar.title("RFP Workspaces")
new_workspace = st.sidebar.text_input("New Workspace Name")
if st.sidebar.button("Create Workspace") and new_workspace:
    st.session_state.workspaces.append(new_workspace)
    st.session_state.current_workspace = new_workspace

st.session_state.current_workspace = st.sidebar.radio("Select Workspace:", st.session_state.workspaces)

st.title(f"🚀 Bid Engine: {st.session_state.current_workspace}")

# --- GLOBAL DATA INGESTION SIDEBAR ---
st.sidebar.markdown("---")
st.sidebar.subheader("System Data Source")
excel_file = st.sidebar.file_uploader("Upload Hackathon Dataset (.xlsx)", type=['xlsx'])

bid_history_df, capability_df = None, None
vector_db, win_model = None, None

if excel_file is not None:
    bid_history_df, capability_df = load_excel_data(excel_file)
    if capability_df is not None:
        vector_db = init_vector_db_from_excel(capability_df)
        st.sidebar.success("✅ RAG Knowledge Base Loaded!")
    if bid_history_df is not None:
        win_model = train_model_from_excel(bid_history_df)
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
        else:
            st.warning("⚠️ Please upload the Hackathon Dataset (.xlsx) via the sidebar to enable RAG features.")

with col2:
    st.subheader("2. GO/NO-GO Dashboard")
    
    if win_model is not None and hasattr(win_model, 'feature_names_inside_'):
        st.write("Score this bid opportunity based on trained historical features.")
        
        # Dynamically build entry forms matching whatever numeric columns are in your sheet
        user_inputs = {}
        for feature in win_model.feature_names_inside_:
            # Detects percentages to scale sliders logically
            if "rate" in feature.lower() or "prob" in feature.lower():
                user_inputs[feature] = st.slider(f"{feature}", 0.0, 1.0, 0.5)
            else:
                user_inputs[feature] = st.number_input(f"Enter {feature}", value=50)
                
        if st.button("Calculate Win Probability"):
            input_df = pd.DataFrame([user_inputs])
            # Ensure sequence sorting maps perfectly to the trained matrix array
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
        st.warning("⚠️ Upload Hackathon Dataset (.xlsx) via sidebar to enable ML Scoring.")