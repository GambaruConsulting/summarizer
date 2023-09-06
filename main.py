from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
import evaluate
import re


rouge = evaluate.load('rouge')


def splitSentence(sentence):
    sentences = []
    periods = [m.start() for m in re.finditer('\.\s', sentence)]
    abbreviations = [m.end() for m in re.finditer('(?:[A-Z][a-zA-Z]{0,}\.){1,}', sentence)]
    sentence_ends = periods
    for index in range(len(sentence_ends)):
        for item in abbreviations:
            if sentence_ends[index] + 1 == item:
                sentence_ends[index] = 0
    if sentence_ends.count(0) > 0:
        while 0 in sentence_ends: sentence_ends.remove(0)
    if len(sentence_ends) > 0:
        sentence_ends.insert(0, -2)
        sentence_ends.append(len(sentence))
        for index in range(len(sentence_ends)-1):
            sentences.append(sentence[sentence_ends[index]+2:sentence_ends[index+1]+1])
        return sentences
    else:
        return [sentence]


summarizer = pipeline(
    "summarization",
    model=AutoModelForSeq2SeqLM.from_pretrained("sshleifer/distilbart-cnn-12-6"),
    tokenizer=AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")
)

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Msg(BaseModel):
    summary: str
    original: str


@app.get("/")
async def root():
    return {"message": "Hello World. Welcome to FastAPI!"}


@app.get("/path")
async def demo_get():
    return {"message": "This is /path endpoint, use a post request to transform the text to uppercase"}


@app.post("/path")
async def demo_post(inp: Msg):
    if len(inp.original) > 2048:
        summary = summarizer(inp.original[:2048])
    else:
        summary = summarizer(inp.original)
    results = rouge.compute(predictions=[inp.summary], references=[[summary[0]["summary_text"]]])
    return {
        "summary": summary[0]["summary_text"],
        "rouge": results['rougeL']
    }


@app.post("/dumbarry")
async def demo_post(inp: Msg):
    return {"bollocks": summarizer(inp.msg)[0]["summary_text"]}


@app.get("/path/{path_id}")
async def demo_get_path_id(path_id: int):
    return {"message": f"This is /path/{path_id} endpoint, use post request to retrieve result"}