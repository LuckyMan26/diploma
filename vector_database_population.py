__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import chromadb
from utils import parse_record,search_for_image
import tensorflow as tf
from sentence_transformers import SentenceTransformer
from datasets import Dataset
import json
import modal
app = modal.App("vector_database_population")

vllm_image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "datasets==3.4.1",    
        "torch",
        "pillow",
        "transformers==4.46.0",
        "tensorflow",
        "scipy",
        "openai",
        "pysqlite3-binary",
        "chromadb",
        "sentence-transformers",
        "tf-keras"
       
    )
)


vol = modal.Volume.from_name("dataset")

def parse_tfrecord_file(ds, filename, batch_size=16):
    """Parses a TFRecord file, retrieves image and prompt data, and computes LLM scores."""
    llm_score_results = []
    human_scores = []

    raw_dataset = tf.data.TFRecordDataset(filename).batch(batch_size)
    for idx, batch_records in enumerate(raw_dataset):
        batch_filenames, batch_images, batch_prompts, batch_human_scores = [], [], [], []

        for raw_record in batch_records:
            filename, uid, human_score = parse_record(raw_record)
            batch_filenames.append(filename)
            batch_human_scores.append(human_score)

            # Fetch image and prompt
            image, prompt = search_for_image(ds=ds, uid=uid)
            batch_images.append(image)
            batch_prompts.append(prompt)

def insert_record(images,captions,human_scores : list[dict], uuids, collection : chromadb.Collection, model:SentenceTransformer):

    
    samples = [f"Ground truth prompt: {caption}\n\
Ground truth Alignment Score: {human_score['alignment_score']},\
Ground truth Artifact score: {human_score['artifact_score']},\
Ground truth Aesthetics Score: {human_score['aesthetics_score']}" for caption,human_score in zip(captions,human_scores)]
    query_vectors = model.encode(captions).tolist()

    collection.add(
        embeddings=query_vectors,
        documents=samples, 
        metadatas=[{'image': str(image)} for image in (images)],
        ids=uuids
    )



@app.function(volumes={"/dataset": vol}, gpu="L4", image=vllm_image,  timeout=60 * 60*2,)
def create_vector_database():
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
    import chromadb
    import tensorflow as tf
    from datasets import load_from_disk
  
    from torch.utils.data import DataLoader
    from sentence_transformers import SentenceTransformer
    filename = "/dataset/dataset/train.tfrecord"
    colection_name = "image_examples"
    client = chromadb.PersistentClient(path='/dataset/vector_database_3')
    raw_dataset = tf.data.TFRecordDataset(filename).batch(16)
    print(f"len dataset: {len(list(raw_dataset))}")
    collection = client.get_or_create_collection(colection_name, metadata={"hnsw:space": "cosine"})

    model_name = "BAAI/bge-base-en-v1.5"
    model = SentenceTransformer(model_name, trust_remote_code=True, cache_folder="/dataset/.cache/").to('cuda')
    dataset = load_from_disk(dataset_path="/dataset/train")
    dataset = dataset.remove_columns("created_at")
    batch_size = 16
    import time
    beg = time.time()
    json_list = [dict(example) for example in dataset]
    print(f"Time for dataset convert: {time.time()-beg}")
   
    for idx, batch_records in enumerate(raw_dataset):
        uuids, batch_human_scores,batch_images,batch_captions =[], [], [], []
        for raw_record in batch_records:

            filename, uid, human_score = parse_record(raw_record=raw_record)
            
            batch_human_scores.append(human_score)
            #print(f"UID: {uid}")
            # Fetch image and prompt
            import time
            beg = time.time()
            image, prompt = search_for_image(ds=json_list, uid=uid)
            print(f"Time for image search: {time.time()-beg}")

            if image == None or prompt == None:
                continue
            #print(prompt)
            uuids.append(uid)
            batch_images.append(image)
            batch_captions.append(prompt)
        insert_record(images=batch_images, captions=batch_captions, human_scores = batch_human_scores, uuids=uuids, collection=collection, model=model)
        print(f"idx: {idx}")