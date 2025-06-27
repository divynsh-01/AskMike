from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from chatbot_core import generate_gemini_answer, load_existing_vector_store

app = FastAPI()

# Enable frontend access (e.g., React/Vite)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QuestionRequest(BaseModel):
    question: str

@app.post("/ask")
def ask_question(req: QuestionRequest):
    try:
        vectorstore = load_existing_vector_store()
        answer = generate_gemini_answer(req.question, vectorstore)
        return {"answer": answer}
    except Exception as e:
        return {"error": str(e)}
