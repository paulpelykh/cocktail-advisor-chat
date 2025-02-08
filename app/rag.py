from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import pickle
import os

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
embeddings = OpenAIEmbeddings()

response_schemas = [
    ResponseSchema(name="is_favorite", description="Whether the user is sharing favorite ingredients", type="boolean"),
    ResponseSchema(name="ingredients", description="List of ingredients mentioned", type="List[str]")
]
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions()

vector_store = FAISS.load_local("data/cocktails_faiss_index", embeddings)
with open('data/cocktails_data.pkl', 'rb') as f:
    cocktails_data = pickle.load(f)

def detect_favorite_ingredients(message: str):
    prompt = [
        SystemMessage(content="Detect if the user is sharing favorite ingredients. Respond with JSON."),
        HumanMessage(content=f"{message}\n{format_instructions}")
    ]
    response = llm(prompt)
    try:
        return output_parser.parse(response.content)
    except:
        return {"is_favorite": False, "ingredients": []}

def process_query(query: str, session_id: str) -> str:
    favorites = sessions.get(session_id, [])
    
    if "my favourite ingredients" in query.lower():
        return format_favorites_response(favorites)
    
    docs = vector_store.similarity_search(query, k=5)
    indices = [doc.metadata['index'] for doc in docs]
    retrieved = [cocktails_data[i] for i in indices]
    
    if "recommend" in query.lower() and favorites:
        filtered = filter_cocktails(retrieved, favorites)
        if filtered:
            retrieved = filtered
    
    context = "\n\n".join([format_cocktail(c) for c in retrieved])
    return generate_response(context, query)

def format_favorites_response(favorites):
    if not favorites:
        return "You haven't shared any favorites yet."
    return f"Your favorite ingredients: {', '.join(favorites)}"

def filter_cocktails(cocktails, favorites):
    return [c for c in cocktails if any(ing.lower() in [f.lower() for f in favorites] for ing in c['ingredients'])]

def format_cocktail(cocktail):
    return f"Name: {cocktail['name']}\nIngredients: {', '.join(cocktail['ingredients'])}\nInstructions: {cocktail['instructions']}"

def generate_response(context, query):
    prompt = f"""Answer using the context below. Be concise.
    
    Context:
    {context}
    
    Query: {query}
    Answer:"""
    response = llm([HumanMessage(content=prompt)])
    return response.content