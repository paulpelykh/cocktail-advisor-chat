from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Dict, List
import uuid
from rag import detect_favorite_ingredients, process_query

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

sessions: Dict[str, List[str]] = {}

class ChatRequest(BaseModel):
    message: str
    session_id: str

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    session_id = request.session_id
    if session_id not in sessions:
        sessions[session_id] = []
    
    user_message = request.message
    detection_result = detect_favorite_ingredients(user_message)
    
    if detection_result['is_favorite']:
        sessions[session_id].extend(detection_result['ingredients'])
        return JSONResponse(content={"response": "Favorite ingredients saved!"})
    
    response = process_query(user_message, session_id)
    return JSONResponse(content={"response": response})

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)