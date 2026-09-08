from fastapi import FastAPI,Form
from pydantic import BaseModel
from starlette.responses import FileResponse
from crew_ai_network import crew

app = FastAPI()

class SupportRequest(BaseModel):
    message:str

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/support")
def customer_support_index():
    return FileResponse("index.html")

@app.post("/support")
def customer_spport(msg:str=Form(...)):
    result = crew.kickoff(
        inputs={
            "customer_message":msg
        }
    )

    return f"{result}"
