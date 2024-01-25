import uvicorn
from src.app import app
from src.app import close_camera
from src.app import task_camera, task_camera_face

if __name__ == "__main__":
    task_camera.start()
    task_camera_face.start()

    uvicorn.run(app, port=8000)
    close_camera()

    task_camera.join()
    print("TASK camera STOP")

    task_camera_face.join()
    print("TASK camera face STOP")

    exit()
