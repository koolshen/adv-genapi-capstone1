"""
Ford Bronco Support Chat System - Using LlamaIndex (adapted from Audi app)
All modules consolidated into a single file
"""
import streamlit as st
import os
import asyncio
import re
import nest_asyncio
from pathlib import Path
from dotenv import load_dotenv
from trello import TrelloClient

# Apply nest_asyncio to allow nested event loops in Streamlit
nest_asyncio.apply()

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage, Settings
from llama_index.core.tools import FunctionTool
from llama_index.core.agent import ReActAgent
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.llms import ChatMessage
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

# Load .env file if it exists (for local development)
# On Hugging Face Spaces, use environment variables/secrets
load_dotenv()

# ============================================================================
# CONFIGURATION
# ============================================================================

# Company Information
COMPANY_NAME = os.getenv("COMPANY_NAME", "TechSupport Inc.")
COMPANY_PHONE = os.getenv("COMPANY_PHONE", "+1-555-0123")
COMPANY_EMAIL = os.getenv("COMPANY_EMAIL", "support@techsupport.com")

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4-turbo-preview")
EMBEDDING_MODEL = "text-embedding-3-small"

# Document Configuration
# Use the specified documents folder path
DOCUMENTS_PATH = r"C:\Users\root\Documents\repos\ai\capstone3\documents"
if not os.path.exists(DOCUMENTS_PATH):
    # Fallback to relative path if absolute doesn't exist
    DOCUMENTS_PATH = os.getenv("DOCUMENTS_PATH", "./documents")
STORAGE_PATH = "./storage"

# Trello Configuration (for ticket creation)
TRELLO_API_KEY = os.getenv("TRELLO_API_KEY", "")
TRELLO_TOKEN = os.getenv("TRELLO_TOKEN", "")
TRELLO_BOARD_ID = os.getenv("TRELLO_BOARD_ID", "")

# ============================================================================
# STREAMLIT APPLICATION
# ============================================================================

# Page configuration
st.set_page_config(
    page_title="Ford Bronco Support Chat",
    page_icon="💬",
    layout="wide"
)

# Check for OpenAI API key
if not OPENAI_API_KEY:
    st.error("""
    ⚠️ **OpenAI API Key Not Found**
    
    Please set your OpenAI API key:
    
    **For Hugging Face Spaces:**
    1. Go to Settings → Secrets
    2. Add a secret named `OPENAI_API_KEY`
    3. Enter your OpenAI API key as the value
    4. Restart the Space
    
    **For Local Development:**
    1. Create a `.env` file in the project root
    2. Add: `OPENAI_API_KEY=your_key_here`
    
    Get your API key at: https://platform.openai.com/api-keys
    """)
    st.stop()

# Setup OpenAI LLM and Embeddings
try:
    # Lower temperature for more deterministic, document-focused responses
    llm = OpenAI(model=OPENAI_MODEL, api_key=OPENAI_API_KEY, temperature=0.1)
    embed_model = OpenAIEmbedding(model=EMBEDDING_MODEL, api_key=OPENAI_API_KEY)
    
    Settings.llm = llm
    Settings.embed_model = embed_model
except Exception as e:
    st.error(f"Error initializing OpenAI: {str(e)}")
    st.stop()

# ============================================================================
# DATA INGESTION (RAG) - Cached globally
# ============================================================================

@st.cache_resource
def get_index():
    """Load and return index from storage or create new one."""
    storage_dir = STORAGE_PATH
    
    # Ensure documents directory exists
    documents_path = Path(DOCUMENTS_PATH)
    if not documents_path.exists():
        documents_path.mkdir(parents=True, exist_ok=True)
        st.warning(f"⚠️ Documents directory not found. Created: {DOCUMENTS_PATH}")
        st.info("Please add PDF or text files to the documents folder and refresh the page.")
        return None
    
    # Check if documents exist
    pdf_files = list(documents_path.glob("*.pdf"))
    txt_files = list(documents_path.glob("*.txt"))
    
    if not pdf_files and not txt_files:
        st.warning(f"⚠️ No documents found in {DOCUMENTS_PATH}")
        st.info("Please add PDF or text files to the documents folder.")
        return None
    
    # Exclude large-manual files
    excluded_files = {'large-manual.pdf', 'large-manual.txt'}
    pdf_files = [f for f in pdf_files if f.name.lower() not in excluded_files]
    txt_files = [f for f in txt_files if f.name.lower() not in excluded_files]
    
    if not pdf_files and not txt_files:
        st.warning("⚠️ No valid documents found (excluding large-manual files)")
        return None
    
    # Load from storage if exists
    if os.path.exists(storage_dir) and os.listdir(storage_dir):
        try:
            storage_context = StorageContext.from_defaults(persist_dir=storage_dir)
            index = load_index_from_storage(storage_context)
            st.success(f"✅ Loaded index from storage ({len(pdf_files + txt_files)} documents)")
            return index
        except Exception as e:
            st.warning(f"⚠️ Could not load from storage: {str(e)}. Creating new index...")
    
    # Create new index
    try:
        with st.spinner("📄 Processing documents and creating embeddings (this may take a few minutes)..."):
            # SimpleDirectoryReader automatically extracts page metadata from PDFs
            documents = SimpleDirectoryReader(DOCUMENTS_PATH).load_data()
            
            # Filter out excluded files and ensure PDFs only
            filtered_docs = []
            for doc in documents:
                file_name = doc.metadata.get("file_name", "")
                # Only include PDF files (exclude text files and excluded PDFs)
                if file_name.lower().endswith('.pdf') and file_name.lower() not in excluded_files:
                    filtered_docs.append(doc)
            
            if not filtered_docs:
                st.error("No valid documents to process")
                return None
            
            index = VectorStoreIndex.from_documents(filtered_docs)
            index.storage_context.persist(persist_dir=storage_dir)
            st.success(f"✅ Created new index with {len(filtered_docs)} documents")
            return index
    except Exception as e:
        st.error(f"❌ Error creating index: {str(e)}")
        return None

# Initialize index and query engine
index = get_index()

if index is None:
    st.stop()

# Create query engine with similarity search and source metadata
rag_engine = index.as_query_engine(
    similarity_top_k=5,
    response_mode="compact"  # Ensures source nodes are included
)

# ============================================================================
# TRELLO INTEGRATION
# ============================================================================

def create_ticket(user_name: str, user_email: str, summary: str, description: str) -> str:
    """Create a support ticket in Trello."""
    if not all([user_name, user_email, summary, description]):
        return "Error: Missing required fields. Please provide name, email, summary, and description."
    
    try:
        if not TRELLO_API_KEY or not TRELLO_TOKEN or not TRELLO_BOARD_ID:
            # Log locally if Trello not configured
            print(f"Support Ticket (Local - Trello not configured):")
            print(f"  Name: {user_name}")
            print(f"  Email: {user_email}")
            print(f"  Summary: {summary}")
            print(f"  Description: {description}")
            return f"✅ Support ticket logged successfully!\n\n**Ticket Details:**\n- Name: {user_name}\n- Email: {user_email}\n- Summary: {summary}\n- Description: {description}\n\n*Note: Trello integration is not configured, so the ticket was logged locally.*"
        
        client = TrelloClient(
            api_key=TRELLO_API_KEY,
            token=TRELLO_TOKEN
        )
        board = client.get_board(TRELLO_BOARD_ID)
        first_list = board.list_lists()[0]
        card = first_list.add_card(
            name=summary,
            desc=f"Name: {user_name}\nEmail: {user_email}\n\n{description}"
        )
        ticket_url = card.url
        
        # Store ticket URL in session state for guaranteed display
        if 'last_ticket_url' not in st.session_state:
            st.session_state.last_ticket_url = None
        st.session_state.last_ticket_url = ticket_url
        
        # Return response with clear, extractable URL format
        return f"✅ Support ticket created successfully in Trello!\n\n**Ticket Details:**\n- Name: {user_name}\n- Email: {user_email}\n- Summary: {summary}\n- Description: {description}\n\n🔗 **TRELLO TICKET LINK:** {ticket_url}\n\n[Click here to open the ticket in Trello]({ticket_url})\n\nTICKET_URL: {ticket_url}"
    except Exception as e:
        # Log locally on error
        print(f"Support Ticket (Local - Trello error):")
        print(f"  Name: {user_name}")
        print(f"  Email: {user_email}")
        print(f"  Summary: {summary}")
        print(f"  Description: {description}")
        print(f"  Error: {str(e)}")
        return f"✅ Support ticket logged locally!\n\n**Ticket Details:**\n- Name: {user_name}\n- Email: {user_email}\n- Summary: {summary}\n- Description: {description}\n\n*Note: Trello integration encountered an error, so the ticket was logged locally.*"

# ============================================================================
# AGENT TOOLS
# ============================================================================

def search_knowledge_base(query: str) -> str:
    """Search ONLY the PDF documents for technical information. Returns answer with sources."""
    try:
        # First, check if the query is clearly unrelated to vehicle documentation
        query_lower = query.lower().strip()
        
        # Patterns that indicate unrelated questions (math, general knowledge, programming, etc.)
        unrelated_patterns = [
            r'\d+\s*[+\-*/=]\s*\d+',  # Simple math: "2 + 2", "5-3", "2+2=?", etc. (removed ^ to match anywhere)
            r'^\d+\s*[+\-*/]\s*\d+\s*[=?]',  # Math with equals/question: "2 + 2 =?", "5-3=?"
            r'what time is',  # Time questions
            r'what is the weather',  # Weather
            r'who is',  # General knowledge about people
            r'what is the capital of',  # Geography
            r'how many days in',  # Calendar questions
            r'what year is',  # Current events
            r'show me.*method',  # Programming: "show me method"
            r'show me.*how to',  # Programming: "show me how to"
            r'how to.*code',  # Programming questions
            r'how to.*program',  # Programming questions
            r'example.*code',  # Code examples
            r'\.net|\.NET',  # .NET framework
            r'c#|c\+\+',  # Programming languages
            r'subtract|substract.*integer',  # Programming operations (with typo)
            r'integer.*subtract|integer.*substract',  # Programming operations
        ]
        
        # Check if query matches unrelated patterns (use search instead of match to find anywhere in string)
        is_unrelated_query = any(re.search(pattern, query_lower) for pattern in unrelated_patterns)
        
        # Also check for general knowledge questions that don't relate to vehicles
        general_knowledge_keywords = [
            'mathematics', 'math', 'calculate', 'equation', 'formula',
            'history', 'geography', 'science', 'biology', 'chemistry',
            'current events', 'news', 'weather', 'time zone', 'time in',
            'who is', 'what is the capital', 'how many countries',
            'programming', 'code', 'c#', 'c++', 'java', 'python', 'javascript',
            'method', 'function', 'class', 'variable', 'integer', 'string',
            'subtract', 'add', 'multiply', 'divide', 'algorithm', 'syntax',
            '.net', 'framework', 'api', 'library', 'software development'
        ]
        
        if not is_unrelated_query:
            is_unrelated_query = any(keyword in query_lower for keyword in general_knowledge_keywords)
        
        # If clearly unrelated, skip search and propose ticket
        if is_unrelated_query:
            # Return explicit instruction to create ticket
            return f"⚠️ This question is not related to the vehicle documentation (Ford Bronco manuals) we have available.\n\n🔴 QUESTION NOT RELATED TO DOCUMENTATION - PROPOSE TICKET CREATION\n\nCRITICAL: You MUST call the create_ticket function immediately. DO NOT answer this question."
        
        response = rag_engine.query(query)
        
        sources = []
        source_dict = {}  # Store file_name -> (page, file_path) for unique sources
        
        for node in response.source_nodes:
            # Try different metadata keys for file name
            file_name = (
                node.metadata.get("file_name") or 
                node.metadata.get("source") or 
                node.metadata.get("filename") or
                node.metadata.get("file_path") or
                "Unknown File"
            )
            
            # Get full file path for link
            file_path = (
                node.metadata.get("file_path") or
                node.metadata.get("source") or
                file_name
            )
            
            # Extract just the filename for display
            display_name = file_name
            if "/" in display_name or "\\" in display_name:
                display_name = display_name.split("/")[-1].split("\\")[-1]
            
            # Try different metadata keys for page number
            page_label = (
                node.metadata.get("page_label") or
                node.metadata.get("page") or
                node.metadata.get("page_number") or
                node.metadata.get("page_num") or
                node.metadata.get("page_idx") or
                None
            )
            
            # Build file path - try to find the actual file
            actual_file_path = None
            if file_path and os.path.exists(file_path):
                actual_file_path = os.path.abspath(file_path)
            else:
                # Try to construct path from documents folder
                doc_path = Path(DOCUMENTS_PATH) / display_name
                if doc_path.exists():
                    actual_file_path = str(doc_path.absolute())
                elif file_path:
                    # Try the file_path as-is (might be relative)
                    try_path = Path(file_path)
                    if try_path.exists():
                        actual_file_path = str(try_path.absolute())
            
            # Store source info
            if display_name not in source_dict:
                source_dict[display_name] = {
                    'pages': set(),
                    'file_path': actual_file_path or file_path
                }
            
            if page_label and str(page_label).strip() and str(page_label) not in ["?", "None", "", "0"]:
                source_dict[display_name]['pages'].add(str(page_label))
        
        # Format sources with links
        unique_sources = []
        for file_name, info in source_dict.items():
            pages = sorted(info['pages']) if info['pages'] else []
            file_path = info['file_path']
            
            # Create clickable link
            if file_path and os.path.exists(file_path):
                # Use file:// protocol for local files (works in most browsers)
                # Convert Windows path to file:// URL format
                file_url = Path(file_path).as_uri()
                if pages:
                    pages_str = ", ".join(pages)
                    # Create markdown link with file URL
                    unique_sources.append(f"[{file_name} (Page {pages_str})]({file_url})")
                else:
                    unique_sources.append(f"[{file_name}]({file_url})")
            else:
                # Fallback without link if file not found
                if pages:
                    pages_str = ", ".join(pages)
                    unique_sources.append(f"{file_name} (Page {pages_str})")
                else:
                    unique_sources.append(file_name)
        
        sources_str = "\n- ".join(unique_sources) if unique_sources else "No sources found."
        
        answer_text = str(response)
        answer_lower = answer_text.lower()
        
        # Check if the answer indicates no relevant information found
        no_info_indicators = [
            "i don't know", "i cannot", "i'm not sure", "i don't have",
            "unable to find", "no information", "not available", "not found",
            "cannot find", "does not contain", "doesn't contain", "no relevant",
            "i apologize", "i'm sorry", "unfortunately", "no specific information",
            "not in the documentation", "not in the manual", "not found in"
        ]
        
        has_no_info = any(indicator in answer_lower for indicator in no_info_indicators)
        has_sources = unique_sources and sources_str != "No sources found."
        
        # Check if answer is actually relevant to the query
        # If query is about math/general knowledge but answer mentions vehicle parts, it's likely a false positive
        query_words = set(query_lower.split())
        
        # Vehicle-related keywords that should be in the answer if it's relevant
        vehicle_keywords = {
            'vehicle', 'car', 'truck', 'ford', 'bronco', 'engine', 'transmission',
            'brake', 'tire', 'wheel', 'door', 'seat', 'belt', 'safety', 'manual',
            'maintenance', 'service', 'repair', 'specification', 'model', 'year',
            'driving', 'operation', 'feature', 'system', 'component', 'part',
            'vehicle', 'owner', 'manual', 'guide', 'instruction'
        }
        
        # Check if answer contains vehicle-related terms (indicating relevance)
        answer_has_vehicle_terms = any(keyword in answer_lower for keyword in vehicle_keywords)
        
        # Check if query is clearly non-vehicle related
        query_is_non_vehicle = any(
            term in query_lower for term in [
                'math', 'calculate', 'equation', 'formula', 'time', 'weather',
                'capital', 'country', 'history', 'who is', 'what is the',
                '=', '+', '-', '*', '/', 'plus', 'minus', 'times', 'divided'
            ]
        )
        
        # If query is non-vehicle but answer doesn't have vehicle terms, it's unrelated
        # OR if we have sources but the answer doesn't make sense for the query
        is_false_positive = (
            (query_is_non_vehicle and not answer_has_vehicle_terms) or
            (has_sources and has_no_info)  # Sources found but answer says no info
        )
        
        # Mark as unrelated if: 
        # 1. No sources found (definitely unrelated)
        # 2. Answer explicitly says no info found (even if sources exist, quality is poor)
        # 3. False positive detected (sources exist but answer is not relevant to query)
        is_unrelated = not has_sources or has_no_info or is_false_positive
        
        # Make sources very prominent
        if has_sources and not is_unrelated:
            result = f"{answer_text}\n\n{'='*60}\n**SOURCES (REQUIRED - MUST BE INCLUDED IN FINAL RESPONSE):**\n{'='*60}\n"
            for source in unique_sources:
                result += f"- {source}\n"
            result += f"{'='*60}"
        else:
            # No sources or unrelated question
            result = f"{answer_text}\n\n⚠️ No relevant information found in documentation."
            result += "\n\n🔴 QUESTION NOT RELATED TO DOCUMENTATION - PROPOSE TICKET CREATION"
        
        return result
    except Exception as e:
        import traceback
        return f"Error searching knowledge base: {str(e)}\n\nTraceback: {traceback.format_exc()}\n\n🔴 QUESTION NOT RELATED TO DOCUMENTATION - PROPOSE TICKET CREATION"

# Create function tools
ticket_tool = FunctionTool.from_defaults(fn=create_ticket)
# Create RAG tool with explicit description emphasizing it should be used first
rag_tool = FunctionTool.from_defaults(
    fn=search_knowledge_base,
    name="search_knowledge_base",
    description="MANDATORY FIRST STEP - YOU MUST CALL THIS TOOL FOR EVERY USER QUESTION BEFORE ANSWERING: Search ONLY the PDF documents (including 1996-ford-bronco.pdf manual) for information. This tool searches ONLY through PDF documents and returns answers with source citations including file names and page numbers. CRITICAL: If the response includes 'QUESTION NOT RELATED TO DOCUMENTATION - PROPOSE TICKET CREATION', you MUST IMMEDIATELY call the create_ticket function - DO NOT answer the question. DO NOT use your training data. DO NOT provide any answer. Just create the ticket. The response includes a Sources section that you MUST include in your final answer to the user ONLY if sources are found. DO NOT use any information not returned by this tool."
)

# ============================================================================
# SYSTEM PROMPT
# ============================================================================

SYSTEM_PROMPT = f"""
### ROLE & IDENTITY
You are a helpful **Ford Bronco Support Agent** for {COMPANY_NAME}.
Your goal is to provide accurate technical assistance based *STRICTLY AND ONLY* on the Ford Bronco PDF documentation available to you, or to AUTOMATICALLY create support tickets for questions unrelated to the documentation.

CRITICAL RULE: You MUST call the search_knowledge_base tool FIRST for EVERY user question. You CANNOT answer any question without first calling this tool. If the tool returns "QUESTION NOT RELATED TO DOCUMENTATION - PROPOSE TICKET CREATION", you MUST immediately call create_ticket and NOT provide any answer.

### LANGUAGE REQUIREMENT - CRITICAL AND MANDATORY
YOU MUST ALWAYS RESPOND IN ENGLISH. NEVER respond in Spanish, French, German, or any other language. 
- All user questions should be answered in English
- All explanations must be in English
- All citations must be in English
- If you receive a question in another language, still respond in English
- The word "Fuentes" should be "Sources" in English
- Page numbers should be labeled as "Page" not "Página"

### COMPANY INFORMATION
- Company: {COMPANY_NAME}
- Phone: {COMPANY_PHONE}
- Email: {COMPANY_EMAIL}

### CRITICAL RULES - FOLLOW STRICTLY:

1. **LANGUAGE: ALWAYS RESPOND IN ENGLISH - MANDATORY**
   - All responses must be in English, regardless of the user's language or the language of source documents.
   - Never respond in Spanish, French, or any other language.
   - If source documents contain text in another language, translate it to English in your response.
   - Use "Sources:" not "Fuentes:", "Page" not "Página", "Answer" not "Respuesta".

2. **DATA SOURCE: ONLY USE PDF DOCUMENTS - NO EXCEPTIONS**
   - You MUST ONLY use information from the PDF documents provided through the `search_knowledge_base` tool.
   - DO NOT use your training data, general knowledge, or any information not found in the PDF documents.
   - DO NOT provide information about models, years, or specifications that are not explicitly mentioned in the search results.
   - If the search results mention "1996-ford-bronco.pdf", you can ONLY provide information about the 1996 Ford Bronco, not other years or models.

3. **MANDATORY: SEARCH FIRST (RAG) - NO EXCEPTIONS:**
   - For EVERY user question, you MUST FIRST call the `search_knowledge_base` tool before providing any answer.
   - DO NOT ask for clarification, DO NOT guess, DO NOT use your training data.
   - IMMEDIATELY use `search_knowledge_base` tool with the user's question as-is.
   - The tool will search ONLY the PDF documents and return relevant information with source citations.
   - ONLY use the information returned by the search tool. Do not add information from your training data.
   - The tool automatically provides source citations (File Name + Page). 
   - **CRITICAL: You MUST include the Sources section from the search results in your final response to the user.**
   - Always cite the document name and page number when using information (e.g., "According to 1996-ford-bronco.pdf, page 13...").
   - The Sources section from the search tool response MUST be included verbatim in your answer.
   - **IF the search tool response contains "QUESTION NOT RELATED TO DOCUMENTATION - PROPOSE TICKET CREATION", you MUST propose creating a support ticket instead of answering.**

4. **HANDLE UNRELATED QUESTIONS - MANDATORY AND STRICT - NO EXCEPTIONS:**
   - You MUST ALWAYS call `search_knowledge_base` tool FIRST for every user question. NEVER answer without calling this tool first.
   - If the `search_knowledge_base` tool returns "QUESTION NOT RELATED TO DOCUMENTATION - PROPOSE TICKET CREATION" or indicates no relevant information, you MUST IMMEDIATELY call the `create_ticket` function.
   - DO NOT ANSWER THE QUESTION. DO NOT make up an answer. DO NOT use your training data. DO NOT provide code examples. DO NOT provide programming help. DO NOT answer math questions. DO NOT ask for clarification. DO NOT just propose - CREATE THE TICKET IMMEDIATELY.
   - If you see "QUESTION NOT RELATED TO DOCUMENTATION - PROPOSE TICKET CREATION" in the search results, your ONLY action is to call create_ticket. Do nothing else.
   - Questions that are NOT related to Ford Bronco documentation include (but not limited to):
     * Math questions (2+2=?, calculations, equations)
     * Programming questions (C#, .NET, code examples, methods, functions, syntax)
     * General knowledge (time, weather, geography, history)
     * Any question not about Ford Bronco vehicles, manuals, or documentation
   - IMMEDIATELY call the `create_ticket` function with:
     * Title/Summary: Extract a meaningful title from the user's question (e.g., "Question: [user's question]" or "Support Request: [topic]")
     * Description: The full user question and note: "This question is not related to the available Ford Bronco vehicle documentation."
     * User Name: Use "User" as default if not provided in conversation or sidebar
     * User Email: Use "user@example.com" as default if not provided in conversation or sidebar
   - For ANY question that doesn't relate to the Ford Bronco manual or other PDF documents, AUTOMATICALLY create a ticket without answering.
   - After creating the ticket, inform the user: "I've created a support ticket for your question since it's not related to the available Ford Bronco documentation. Here's the link to your ticket: [link]"

5. **TICKET CREATION PROTOCOL:**
   - If the user agrees to open a ticket (or explicitly asks for one), you must collect the following 4 pieces of information:
     1. **User Name** (check if provided in sidebar)
     2. **User Email** (check if provided in sidebar)
     3. **Summary** (Short title extracted from the user's question)
     4. **Description** (Include the user's question and note that it wasn't found in the documentation)
   - Do NOT call the `create_ticket` tool until you have ALL 4 values. Ask for missing details one by one if needed.

6. **TONE:**
   - Professional, concise, and helpful.
   - Always respond in English.
   - Always cite sources with document name and page number when available.
"""

# ============================================================================
# UI LOGIC
# ============================================================================

st.title("💬 Ford Bronco Support Chat")
st.markdown("Ask me anything! I'll search through the documentation to help you.")

# Sidebar
with st.sidebar:
    st.header("User Information")
    st.session_state.user_name = st.text_input("Your Name", value=st.session_state.get("user_name", ""))
    st.session_state.user_email = st.text_input("Your Email", value=st.session_state.get("user_email", ""))
    
    st.divider()
    
    st.subheader("About")
    st.info("""
    This Ford Bronco Support Chat system can:
    - Answer questions from documentation
    - Create support tickets when needed
    - Cite sources with page numbers
    - Maintain conversation history
    """)
    
    if TRELLO_API_KEY and TRELLO_TOKEN and TRELLO_BOARD_ID:
        st.success("✅ Trello Board Connected")
    else:
        st.info("ℹ️ Tickets logged locally (Trello not configured)")

# Initialize messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Type your question here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            
            # Helper to run async agent
            async def run_agent_interaction(user_prompt, history):
                # Build chat history for memory
                chat_history = [
                    ChatMessage(role=m["role"], content=m["content"])
                    for m in history
                ]
                memory = ChatMemoryBuffer.from_defaults(chat_history=chat_history)

                # Initialize ReActAgent with explicit language instruction
                enhanced_prompt = SYSTEM_PROMPT + "\n\nCRITICAL: You MUST respond in ENGLISH ONLY. Never respond in Spanish, French, or any other language. All responses must be in English."
                
                agent = ReActAgent(
                    tools=[rag_tool, ticket_tool],  # RAG tool first to prioritize searching
                    llm=llm,
                    memory=memory,
                    timeout=120,
                    system_prompt=enhanced_prompt,
                    verbose=True
                )
                
                # Add user info to context if available
                context = ""
                if st.session_state.get("user_name"):
                    context += f"\nUser Name: {st.session_state.user_name}"
                if st.session_state.get("user_email"):
                    context += f"\nUser Email: {st.session_state.user_email}"
                
                if context:
                    user_prompt = f"{user_prompt}\n\n{context}"
                
                # Add explicit language instruction to user prompt
                user_prompt = f"{user_prompt}\n\nIMPORTANT: Respond in ENGLISH ONLY. Do not use Spanish or any other language."
                
                return await agent.run(user_msg=user_prompt)
            
            # Execution with loop handling
            try:
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                
                response = loop.run_until_complete(
                    run_agent_interaction(prompt, st.session_state.messages[:-1])
                )
                
                # Handle response - it might be a Response object or string
                if hasattr(response, 'response'):
                    response_text = str(response.response)
                else:
                    response_text = str(response)
                
                st.markdown(response_text)
                
                # Always check for and display Trello ticket link
                ticket_url = None
                
                # First, check if we have a stored ticket URL from this interaction
                if st.session_state.get('last_ticket_url'):
                    ticket_url = st.session_state.last_ticket_url
                    # Clear it after use
                    st.session_state.last_ticket_url = None
                
                # Also try to extract URL from response text (multiple methods)
                if not ticket_url:
                    # Method 1: Look for TICKET_URL: prefix (added by create_ticket function)
                    ticket_url_match = re.search(r'TICKET_URL:\s*(https://trello\.com/[^\s\n]+)', response_text)
                    if ticket_url_match:
                        ticket_url = ticket_url_match.group(1)
                    else:
                        # Method 2: Extract any Trello URL
                        url_pattern = r'https://trello\.com/[^\s\)\n]+'
                        urls = re.findall(url_pattern, response_text)
                        if urls:
                            ticket_url = urls[0]
                
                # Also check if ticket was created (look for keywords)
                ticket_created = (
                    "ticket created" in response_text.lower() or 
                    "support ticket" in response_text.lower() or
                    "trello" in response_text.lower()
                )
                
                # Display ticket link prominently if found
                if ticket_url:
                    st.markdown("---")
                    st.markdown(f"### 🔗 **Trello Ticket Link**")
                    st.markdown(f"[**Click here to view your ticket in Trello**]({ticket_url})")
                    st.markdown(f"**Direct URL:** `{ticket_url}`")
                    # Update response text to include the link
                    if ticket_url not in response_text:
                        response_text = f"{response_text}\n\n---\n### 🔗 **Trello Ticket Link**\n[**Click here to view your ticket in Trello**]({ticket_url})\n\n**Direct URL:** `{ticket_url}`"
                elif ticket_created and not ticket_url:
                    # Ticket was created but URL not found - try to get from session state one more time
                    if st.session_state.get('last_ticket_url'):
                        ticket_url = st.session_state.last_ticket_url
                        st.markdown("---")
                        st.markdown(f"### 🔗 **Trello Ticket Link**")
                        st.markdown(f"[**Click here to view your ticket in Trello**]({ticket_url})")
                        st.markdown(f"**Direct URL:** `{ticket_url}`")
                        response_text = f"{response_text}\n\n---\n### 🔗 **Trello Ticket Link**\n[**Click here to view your ticket in Trello**]({ticket_url})\n\n**Direct URL:** `{ticket_url}`"
                        st.session_state.last_ticket_url = None
                
                st.session_state.messages.append({"role": "assistant", "content": response_text})

            except Exception as e:
                error_msg = f"I apologize, but I encountered an error: {str(e)}. Please try again or create a support ticket."
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
