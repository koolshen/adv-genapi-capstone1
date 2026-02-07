# Ford Bronco Support Chat System

A conversational AI support system for Ford Bronco vehicles, powered by LlamaIndex and Streamlit. This application provides intelligent document-based Q&A and automated ticket creation for customer support.

**🌐 Published on [Hugging Face Spaces](https://huggingface.co/spaces)**

## Features

### 🤖 AI-Powered Document Search (RAG)
- **Retrieval-Augmented Generation** using LlamaIndex to search through Ford Bronco PDF documentation
- Processes multiple PDF manuals (1996-2024 Ford Bronco models)
- Creates and persists vector embeddings for fast, accurate retrieval
- Automatically filters and excludes large manual files during processing

### 📚 Knowledge Base Query Engine
- Searches through PDF documents to answer technical questions
- Provides **source citations** with file names and page numbers
- Clickable links to source documents with page references
- Similarity-based search with top-k retrieval (top 5 results)

### 🎫 Automated Support Ticket Creation
- **Trello Integration** for creating support tickets
- Automatically detects questions unrelated to documentation
- Creates tickets for math, programming, general knowledge, or off-topic questions
- Falls back to local logging if Trello is not configured
- Displays ticket links prominently in the chat interface

### 💬 Conversational Interface
- **Streamlit-based chat UI** with conversation history
- Maintains context across multiple messages using ChatMemoryBuffer
- User information collection (name and email) via sidebar
- Real-time response streaming with loading indicators

### 🧠 Intelligent Agent System
- **ReAct Agent** (Reasoning + Acting) for structured problem-solving
- Mandatory knowledge base search before answering any question
- Strict enforcement to use only documentation sources (no training data)
- Automatic ticket creation for unrelated queries

### 🌍 Language Support
- **English-only responses** enforced at system level
- Handles multilingual queries but responds in English
- Translates source document content to English when needed

### ⚙️ Configuration & Setup
- Environment variable support for API keys and configuration
- Hugging Face Spaces integration with secrets management
- Configurable company information (name, phone, email)
- Document path configuration with fallback options

## Technology Stack

- **Streamlit** - Web interface
- **LlamaIndex** - RAG framework and vector store
- **OpenAI** - LLM (GPT-4) and embeddings (text-embedding-3-small)
- **Trello API** - Support ticket management
- **Python** - Backend logic

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set environment variables:
   - `OPENAI_API_KEY` - Your OpenAI API key
   - `TRELLO_API_KEY` (optional) - For ticket creation
   - `TRELLO_TOKEN` (optional) - Trello authentication token
   - `TRELLO_BOARD_ID` (optional) - Trello board ID

3. Add PDF documents to the `documents/` folder

4. Run the application:
   ```bash
   streamlit run app.py
   ```

## Usage

1. Enter your name and email in the sidebar (optional, for ticket creation)
2. Ask questions about Ford Bronco vehicles, manuals, or documentation
3. The system will search through PDF documents and provide answers with source citations
4. For unrelated questions, a support ticket will be automatically created

## Document Processing

The system processes PDF files from the `documents/` directory and creates a persistent vector index stored in the `storage/` folder. The index is automatically loaded on startup if it exists, or created fresh if documents are added or modified.
