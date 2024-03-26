from fastapi import FastAPI
from src.semantic_search import semantic_search
from src.translate import gpt_translate
import json
app = FastAPI()


@app.get("/docsearch")
async def process_text(text: str):
    eng_json = await gpt_translate.translate_to_english_gpt4(text)
    
    eng_text = json.loads(eng_json)["english_text"]
    print(eng_text)
    response_pincone = await semantic_search.query_pincone_hybrid(eng_text)
    print(response_pincone)
    result = []
    for i in response_pincone["matches"]:
        res_dic = i["metadata"]
        kor_json = await gpt_translate.translate_to_korean_gpt4(res_dic["text"])
      
        res_dic["text"] = json.loads(kor_json)["korean_text"]
        result.append(res_dic)
    return result

# uvicorn main:app --host 0.0.0.0 --port 9900
