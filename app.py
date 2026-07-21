from fastapi import FastAPI,HTTPException,Request
from pydantic import BaseModel
from src.pipeline.inference import Inference
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app:FastAPI):
    app.state.inf_obj = Inference()
    yield
    app.state.inf_obj.sql_conn.close()

app = FastAPI(lifespan=lifespan)

class log_ds(BaseModel):
    logs:str

@app.get('/')
def homepage():
    return {"message":"welcome to the home page of LogRecall"}

@app.post('/search')
def fun(io: log_ds, request:Request):
    if not io:
        raise HTTPException(status_code=400, detail="logs can't be empty")
    try:
        engine = request.app.state.inf_obj
        embedding = engine.get_embedding(io.logs)
        res = engine.get_results(embedding)
        return {"results":res}
    except Exception as e:
        raise HTTPException(status_code=500,detail=str(e))