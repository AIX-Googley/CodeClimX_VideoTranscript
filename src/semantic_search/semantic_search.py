from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from splade.models.transformer_rep import Splade
from transformers import AutoTokenizer
import torch
from dotenv import load_dotenv
import os


def convert_seconds_to_hms(seconds):
    # 시간, 분, 초로 변환
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    
    # hh:mm:ss 포맷으로 변환
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def dense_embedding(model, text):
    embedded_text = model.encode(text).tolist()
    return embedded_text

def sparse_enbedding(tokenizer, model, text):
    tokens = tokenizer(text, return_tensors='pt')
    with torch.no_grad():
        sparse_emb = model(d_kwargs=tokens.to('cuda'))['d_rep'].squeeze()
    sparse_emb.shape
    indice = sparse_emb.nonzero().squeeze().cuda().tolist()
    value = sparse_emb[indice].cuda().tolist()
    return indice, value



def hybrid_score_norm(dense, sparse, alpha: float):
    if alpha < 0 or alpha > 1:
        raise ValueError("Alpha must be between 0 and 1")
    hs = {
        'indices': sparse['indices'],
        'values':  [v * (1 - alpha) for v in sparse['values']]
    }
    return [v * alpha for v in dense], hs


async def query_pincone_hybrid(text):
    dense_vector = dense_embedding(dense_model, text)
    indice, value = sparse_enbedding(tokenizer, sparse_model, text)
    sparse_vector={
            'indices': indice,
            'values':  value
        }
    
    hdense, hsparse = hybrid_score_norm(dense_vector, sparse_vector, alpha=0.5)
    index = pc.Index("final240325")
    result = index.query(
        top_k=5,
        vector=hdense,
        # vector=dense_vector,
        sparse_vector=hsparse,
        namespace='sparse_multilingual',
        include_metadata=True,
    )
    return result



# sparse model
sparse_model_name = 'naver/splade-cocondenser-ensembledistil'
sparse_model = Splade(sparse_model_name, agg='max')
sparse_model.to('cuda')
sparse_model.eval()
tokenizer = AutoTokenizer.from_pretrained(sparse_model_name)

# dense_model
dense_model = SentenceTransformer('distiluse-base-multilingual-cased-v1', device='cuda')

Pinecone_KEY = os.environ.get("PINECONE_API_KEY")
pc = Pinecone(api_key=Pinecone_KEY)