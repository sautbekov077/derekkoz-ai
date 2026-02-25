import asyncio
import json
import os
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ddgs import DDGS
from bs4 import BeautifulSoup
import httpx
from groq import Groq
from dotenv import load_dotenv


load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware, 
    allow_origins=["*"], 
    allow_methods=["*"], 
    allow_headers=["*"]
)


groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

class ChatRequest(BaseModel):
    query: str

async def fetch_page(client, url):
    try:
        response = await client.get(url, timeout=2.0)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        for script in soup(["script", "style", "nav", "footer"]):
            script.extract()
        text = " ".join(soup.stripped_strings)
        return text[:1500]
    except Exception:
        return ""

async def get_search_context(query: str):
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=5))
    
    urls = [res['href'] for res in results]
    sources_metadata = []
    
    for res in results:
        domain = res['href'].split('/')[2]
        sources_metadata.append({
            "title": res['title'],
            "url": res['href'],
            "domain": domain,
            "favicon": f"https://www.google.com/s2/favicons?domain={domain}&sz=32"
        })

    async with httpx.AsyncClient() as client:
        tasks = [fetch_page(client, url) for url in urls]
        pages_content = await asyncio.gather(*tasks)

    context_text = ""
    for idx, content in enumerate(pages_content):
        if content:
            context_text += f"Дереккөз {idx+1} ({urls[idx]}):\n{content}\n\n"
            
    return sources_metadata, context_text


@app.get("/")
async def serve_frontend():
    with open("index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.get("/logo.png")
async def serve_logo():
    if os.path.exists("logo.png"):
        return FileResponse("logo.png")
    return HTMLResponse(status_code=404)

@app.get("/favicon.png")
async def serve_favicon():
    if os.path.exists("favicon.png"):
        return FileResponse("favicon.png")
    return HTMLResponse(status_code=404)

@app.post("/api/search")
async def search_endpoint(request: ChatRequest):
    async def generate():
        sources, context_text = await get_search_context(request.query)
        yield f"data: {json.dumps({'type': 'sources', 'data': sources})}\n\n"

        prompt = f"""
        Сен тез іздеу жасайтын ЖИөсің. Қолданушы қойған сұраққа интернеттегі дереккөздерге сүйене отырып жауап бересің.
        Сенің жауабың ҚАТАҢ түрде осындай бөлімдерден тұру керек (Markdown қолдан):
        
        ### 1. Қысқа жауап (максимум 4-5 сөйлем)
        ### 2. Негізгі фактілер (дереккөздерге сүйене отырып)
        ### 3. Дерекөздердегі айырмашылықтар (егер бар болса)
        ### 4. Қорытынды (сенің жеке ойың, егер сұраққа қатысты болса)

        Сұраныс: {request.query}
        
        Дереккөздер:
        {context_text}
        """
        
       
        stream = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="groq/compound", 
            stream=True,
        )
        
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield f"data: {json.dumps({'type': 'text', 'data': chunk.choices[0].delta.content})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
