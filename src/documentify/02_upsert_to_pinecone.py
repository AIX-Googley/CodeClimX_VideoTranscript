import os
import json
from sentence_transformers import SentenceTransformer
from splade.models.transformer_rep import Splade
from transformers import AutoTokenizer
import torch
from pinecone import Pinecone
from dotenv import load_dotenv


# 정리된 문서로부터 timestamp와 텍스트 추출
def read_and_combine_texts(directory_path):
    timestamps = []
    paragraphs = []
    cnt = 0
    for filename in sorted(os.listdir(directory_path)):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path) and file_path.endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as file:
                print(cnt)
                data = json.load(file)
                for item in data:
                    timestamps.append(item['timestamp'])
                    paragraphs.append(item['paragraph'])
        cnt += 1
    return timestamps, paragraphs


# hh:mm:ss를 second로 변환
def convert_times_to_seconds(time_list):
    seconds_list = []
    for time_str in time_list:
        # 시간, 분, 초를 분리
        hours, minutes, seconds = map(int, time_str.split(':'))
        # 전체 시간을 초로 환산
        total_seconds = hours * 3600 + minutes * 60 + seconds
        seconds_list.append(total_seconds)
    return seconds_list


# dnese 임베딩
def dense_embedding(model, texts):
    dense_embeddings = []
    for i in texts:
        embedded_text = model.encode(i)
        dense_embeddings.append(embedded_text)
    return dense_embeddings

# splade 임베딩
def sparse_enbedding(tokenizer, model, texts):
    indices = []
    values = []
    for i in texts:
        tokens = tokenizer(i, return_tensors='pt')

        with torch.no_grad():
            sparse_emb = model(
                d_kwargs=tokens.to('cuda')
            )['d_rep'].squeeze()
        sparse_emb.shape
        
        indice = sparse_emb.nonzero().squeeze().cuda().tolist()
        value = sparse_emb[indice].cuda().tolist()
        indices.append(indice)
        values.append(value)
    return indices, values

# sparse model
sparse_model_name = 'naver/splade-cocondenser-ensembledistil'
sparse_model = Splade(sparse_model_name, agg='max')
sparse_model.to('cuda')
sparse_model.eval()
tokenizer = AutoTokenizer.from_pretrained(sparse_model_name)


# Pinecone 접속절정
Pinecone_KEY = os.environ.get("PINECONE_API_KEY")
pc = Pinecone(api_key=Pinecone_KEY)


# pinecone에 저장될 비디오명, url, id, namespace, index 설정
video_name = 'Machine Learning in 2024 – Beginners Course'
url='https://youtu.be/bmmQA8A-yUA?si=mdMQhke4u5Z0fES-'
id = "ML2024"
index_namespace = 'sparse_multilingual'
index_name = 'final240325'


directory_path = "static/srt/" + video_name + "/summarization_english"
dense_model = SentenceTransformer('distiluse-base-multilingual-cased-v1', device='cuda')

timestamps, paragraphs = read_and_combine_texts(directory_path)

seconds = convert_times_to_seconds(timestamps)

dense_embeddings = dense_embedding(dense_model, paragraphs)


sparse_indices, sparse_values = sparse_enbedding(tokenizer, sparse_model, paragraphs)



vectors = []
cnt = 1
for emb, text, second, sparse_indice, sparse_value in zip(dense_embeddings, paragraphs, seconds, sparse_indices, sparse_values):
    vector = {"id": id + "_" + str(cnt),
              "values" : emb,
              "sparse_values" : {'indices': sparse_indice, 'values': sparse_value},
              "metadata" : {"text": text, "videoName": video_name, "url": url, "second": second}
              }
    vectors.append(vector)
    cnt += 1
index = pc.Index(index_name)
index.upsert(vectors=vectors, namespace=index_namespace)
    

