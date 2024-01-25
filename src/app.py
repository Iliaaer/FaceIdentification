import threading
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from src.camera.router import get_frame, get_frame_face, cap
from src.camera.router import router as router_camera

task_camera = threading.Thread(target=get_frame)
task_camera_face = threading.Thread(target=get_frame_face)

app = FastAPI(
    title="Verification"
)

app.mount(
    "/static",
    StaticFiles(directory=Path(__file__).parent.parent.absolute() / "src/static"),
    name="static",
)

templates = Jinja2Templates(
    directory=Path(__file__).parent.parent.absolute() / "src/templates",
)

app.include_router(router_camera)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request},
    )


def close_camera():
    return cap.isOpened()
