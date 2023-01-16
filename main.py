from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def root():
    return {"message": 
            """Welcome, 
            this model will help to predict the income levels based on demographics and characterstics"""}