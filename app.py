import os
import uuid
import json
import cv2
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for
from ultralytics import YOLO
from flask import Response
from flask import jsonify
from openpyxl import Workbook
from flask import send_file
from openpyxl import Workbook
from flask import send_file


app = Flask(__name__)

UPLOAD_IMAGE_DIR = "static/uploads"
VIDEO_DIR = "static/videos"
HISTORY_FILE = "history.json"

STREAM_VIDEO = "static/stream/dogs.mp4"
current_no_muzzle = 0
stream_active = False


os.makedirs(UPLOAD_IMAGE_DIR, exist_ok=True)
os.makedirs(VIDEO_DIR, exist_ok=True)

NO_MUZZLE_CLASS_ID = 1
model = YOLO("models/muzzle_best.pt")


def save_history(filename, count):
    record = {
        "file": filename,
        "dogs": count,
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = []

    data.append(record)

    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/image", methods=["POST"])
def image():
    file = request.files.get("image")
    if not file or file.filename == "":
        return redirect("/")

    path = os.path.join(UPLOAD_IMAGE_DIR, file.filename)
    file.save(path)

    results = model(path)[0]
    count = 0

    if results.boxes is not None:
        for box in results.boxes:
            if int(box.cls[0]) == NO_MUZZLE_CLASS_ID:
                count += 1

    result_img = results.plot()
    result_path = os.path.join(UPLOAD_IMAGE_DIR, "result.jpg")
    cv2.imwrite(result_path, result_img)

    save_history(file.filename, count)

    return render_template(
        "index.html",
        active="photo",
        image_ready=True,
        image_path="/static/uploads/result.jpg",
        dogs=count
    )


@app.route("/video", methods=["POST"])
def video():
    if "video" not in request.files:
        return redirect(url_for("index"))

    file = request.files["video"]
    if not file or file.filename == "":
        return redirect(url_for("index"))

    uid = uuid.uuid4().hex
    input_name = f"input_{uid}.mp4"
    output_name = f"result_{uid}.webm"

    input_path = os.path.join(app.root_path, VIDEO_DIR, input_name)
    output_path = os.path.join(app.root_path, VIDEO_DIR, output_name)

    file.save(input_path)
    cap = cv2.VideoCapture(input_path)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps > 120:
        fps = 25

    # Настройки (подкорректируй при необходимости)
    IMG_SIZE = 640        # как для фото
    CONF_THRES = 0.25     # порог доверия (как для фото)
    FRAME_STEP = 1        # 1 = детектим каждый кадр; >1 = пропуск кадров для скорости

    fourcc = cv2.VideoWriter_fourcc(*"VP80")   # webm/VP8 - надежно в браузерах
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    frame_idx = 0
    max_no_muzzle = 0
    last_annotated = None
    last_count = 0

    print(f"[VIDEO] start processing {input_path} -> {output_path}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # по желанию можно ускорять, обрабатывая каждый N-й кадр
        if FRAME_STEP > 1 and frame_idx % FRAME_STEP != 0:
            # повторно записываем последний аннотированный кадр (или текущий кадр, если нет аннотаций)
            if last_annotated is None:
                out.write(frame)
            else:
                out.write(last_annotated)
            continue

        # Детекция как для фото — модель сама вернёт results[0]
        results = model(frame, imgsz=IMG_SIZE, conf=CONF_THRES)[0]

        # подсчёт No_muzzle в текущем кадре
        current_no_muzzle = 0
        if results.boxes is not None:
            try:
                for box in results.boxes:
                    if int(box.cls[0]) == NO_MUZZLE_CLASS_ID:
                        current_no_muzzle += 1
            except Exception:
                # на случай неожиданных форматов
                current_no_muzzle = 0

        # рисуем аннотацию так же, как на фото
        try:
            annotated = results.plot()  # Ultralytics метод — возвращает изображение с аннотациями
            # убедимся, что тип uint8
            if annotated.dtype != "uint8":
                annotated = annotated.astype("uint8")
            last_annotated = annotated
            last_count = current_no_muzzle
            out.write(annotated)
        except Exception as e:
            # если plot() вдруг упал — fallback: запишем оригинальный кадр
            print(f"[VIDEO] plot() error on frame {frame_idx}: {e}")
            last_annotated = frame
            last_count = current_no_muzzle
            out.write(frame)

        if current_no_muzzle > max_no_muzzle:
            max_no_muzzle = current_no_muzzle

        # лог в консоль для отладки (не слишком часто)
        if frame_idx % 50 == 0:
            print(f"[VIDEO] frame {frame_idx} — no_muzzle_in_frame={current_no_muzzle}  max_so_far={max_no_muzzle}")

    cap.release()
    out.release()

    save_history(output_name, max_no_muzzle)

    print(f"[VIDEO] done. max_no_muzzle_in_any_frame = {max_no_muzzle}")

    return render_template(
        "index.html",
        video_ready=True,
        active="video",
        video_path=f"/static/videos/{output_name}?v={uid}",
        dogs=max_no_muzzle
    )



def generate_stream():
    global current_no_muzzle, stream_active

    cap = cv2.VideoCapture(STREAM_VIDEO)

    while stream_active:
        ret, frame = cap.read()

        if not ret:
            cap.release()
            cap = cv2.VideoCapture(STREAM_VIDEO)
            continue

        results = model(frame)[0]

        no_muzzle_count = 0
        if results.boxes is not None:
            for box in results.boxes:
                if int(box.cls[0]) == NO_MUZZLE_CLASS_ID:
                    no_muzzle_count += 1

        current_no_muzzle = no_muzzle_count

        annotated = results.plot()
        _, buffer = cv2.imencode(".jpg", annotated)

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + buffer.tobytes()
            + b"\r\n"
        )

    cap.release()
    print("[STREAM] stopped")



@app.route("/stream")
def stream():
    global stream_active
    stream_active = True

    return Response(
        generate_stream(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/stop_stream", methods=["POST"])
def stop_stream():
    global stream_active
    stream_active = False
    return "", 204

@app.route("/count")
def count():
    return jsonify({"dogs": current_no_muzzle})


@app.route("/stats")
def stats():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            history = json.load(f)
    else:
        history = []

    return render_template("stats.html", history=history)

@app.route("/export/excel")
def export_excel():
    if not os.path.exists(HISTORY_FILE):
        return "Нет данных", 400

    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        history = json.load(f)

    wb = Workbook()
    ws = wb.active
    ws.title = "История"

    ws.append(["Файл", "Собак без намордника", "Время"])

    for item in history:
        ws.append([item["file"], item["dogs"], item["time"]])

    path = "history.xlsx"
    wb.save(path)

    return send_file(path, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
