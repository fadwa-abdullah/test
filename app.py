from fastapi import FastAPI
from pydantic import BaseModel
from rag_pipeline import make_qa_chain, update_index
from config import rag_config 
app = FastAPI(title="Demand to Value API")

class QueryRequest(BaseModel):
    query: str
    
@app.get("/")
def root():
    return {"message": "Demand to value API is running"}

@app.post("/ask")
def ask_question(request: QueryRequest):
    qa_chain = make_qa_chain(
        skip_update=True  # Use updated global config
    )
    
    response = qa_chain.invoke({"input": request.query})
    return {
        "answer": response["output_text"], 
        "model_used": rag_config.model_name,
        "model_type": rag_config.model_type,
        "collection": rag_config.document_collection_name
    }

@app.post("/update-index")
def refresh_index():
    update_index()
    return {"status": "Index updated successfully!"}

@app.get("/config")
def get_current_config():
    """Get current global configuration"""
    from rag_pipeline import rag_config
    return {
        "model_name": rag_config.model_name,
        "model_type": rag_config.model_type,
        "system_prompt": rag_config.system_prompt,
        "document_collection": rag_config.document_collection_name
    }
""" {
    "model_name": "gpt-4o-mini",
    "model_type": "openAI",
    "system_prompt": "Answer the question based only on the following context:",
    "local_embedding_name": "sentence-transformers/all-MiniLM-L6-v2",
    "openai_api_key": "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",
    "document_collection_name": "documents",
    "document_collection_path": "documents"
  } """
@app.post("/config")
def update_global_config(config_update: dict):
    """Update global configuration"""
    from rag_pipeline import rag_config
    rag_config.update(**config_update)
    return {"status": "Global configuration updated", "new_config": {
        "model_name": rag_config.model_name,
        "model_type": rag_config.model_type,
        "system_prompt": rag_config.system_prompt,
        "document_collection_name": rag_config.document_collection_name,
        "document_collection_path": rag_config.document_collection_path,
         "local_embedding_name": rag_config.local_embedding_name,
         "openai_api_key": rag_config.openai_api_key
    }}

@app.post("/ask-with-global-config")
def ask_with_global_config(query: str):
    """Ask question using global configuration"""
    qa_chain = make_qa_chain(skip_update=True)  # Uses global rag_config
    response = qa_chain.invoke({"input": query})
    
    from rag_pipeline import rag_config
    return {
        "answer": response["output_text"],
        "config_used": {
            "model_name": rag_config.model_name,
            "model_type": rag_config.model_type,
            "system_prompt": rag_config.system_prompt,
            "document_collection": rag_config.document_collection_name
        }
    }

#todo: add error handling, logging, etc. , model name param to make_qa_chain, cashing ,collections,enhance system propmpt,scoring: 
# remove streamlit dependency from this file

# local testing, for embeddings and index update, appsettings , pakage versions, retest when we change embedding type 