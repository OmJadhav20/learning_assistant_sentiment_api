from fastapi import FastAPI,HTTPException
from pydantic import BaseModel
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

nltk.download('vader_lexicon')

app=FastAPI()
sia=SentimentIntensityAnalyzer()

class SentimentRequest(BaseModel):
    text :list[str]


@app.post("/analyze")
async def analyze_sentiment(request: SentimentRequest):
    if not request.text:
        raise HTTPException(status_code=400,detail='Text field is empty')
    

    results=[]
    for text in request.text:
        scores=sia.polarity_scores(text)
        results.append(scores['compound'])
    results_avg=sum(results)/len(results)
    final_result=0
    if results_avg<0:
        final_result=-1
    if results_avg>0:
        final_result=1

    return{
        "status":"success",
        "result":final_result,
    }