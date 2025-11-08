# Configuration class for dynamic settings
class RAGConfig:
    def __init__(self):
        self.model_name = "" # e.g., "gpt-4o-mini", "local-model"
        self.model_type = "" # e.g., " openAI,  huggingface ,gemini, local"
        self.system_prompt = "" # e.g., "Answer the question based only on the following context:"
        self.local_embedding_name = "" # e.g., "sentence-transformers/all-MiniLM-L6-v2"
        self.openai_api_key = "" # e.g., "sk-..."
        self.document_collection_name = "" # e.g., "my_documents"
        self.document_collection_path = "" # e.g., "/path/to/documents"
    def update(self, **kwargs):
        """Update configuration parameters"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

# Global config instance
rag_config = RAGConfig()
# ================================