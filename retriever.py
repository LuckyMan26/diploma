from sentence_transformers import SentenceTransformer
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import chromadb

class Retriever():
    def __init__(self):
        self.model_name = "BAAI/bge-base-en-v1.5"
        self.model = SentenceTransformer(self.model_name, trust_remote_code=True, cache_folder="/dataset/.cache/").to('cuda')
        self.client = chromadb.PersistentClient(path='/dataset/vector_database_3')
        self.collection = self.client.get_or_create_collection("image_examples", metadata={"hnsw:space": "cosine"})
    def retrieve(self, batch_prompt: list[str]):
        query_vectors = self.model.encode(batch_prompt).tolist()
        query_result = self.collection.query(
            query_embeddings=query_vectors,
            n_results=1,)
        return query_result
