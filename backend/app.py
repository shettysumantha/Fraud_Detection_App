from fastapi import FastAPI

from backend.routes import router

app = FastAPI(title="Fraud Detection API", version="0.1")
app.include_router(router, prefix="/api")


@app.get("/")
def root() -> dict:
	return {"status": "ok", "api_root": "/api"}
