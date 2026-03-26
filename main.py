import base64
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse

from saliency import process_image

ALLOWED_TYPES = {"image/jpeg", "image/png"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

app = FastAPI(title="視線誘導シミュレーター")


@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = Path(__file__).parent / "index.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


def _serve_page(filename: str) -> HTMLResponse:
    html_path = Path(__file__).parent / filename
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.get("/privacy", response_class=HTMLResponse)
async def privacy():
    return _serve_page("privacy.html")


@app.get("/contact", response_class=HTMLResponse)
async def contact():
    return _serve_page("contact.html")


@app.get("/about", response_class=HTMLResponse)
async def about():
    return _serve_page("about.html")


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail="対応していないファイル形式です。JPG または PNG をアップロードしてください。",
        )

    data = await file.read()
    if len(data) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail="ファイルサイズが大きすぎます（上限 10MB）。",
        )

    try:
        result = process_image(data)
    except (ValueError, RuntimeError) as e:
        raise HTTPException(status_code=422, detail=str(e))

    score = result["score"]
    return {
        "original": base64.b64encode(result["original_png"]).decode(),
        "heatmap": base64.b64encode(result["heatmap_png"]).decode(),
        "zones": result["zones"],
        "score": {
            "score": score["score"],
            "type": score["type"],
            "area_ratio": score["area_ratio"],
            "advice": score["advice"],
        },
        "width": result["width"],
        "height": result["height"],
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
