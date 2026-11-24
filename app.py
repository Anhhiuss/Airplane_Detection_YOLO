import streamlit as st
import cv2
import tempfile
import math
import time
import pandas as pd
from ultralytics import YOLO

def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea)

def filter_overlapping_boxes(boxes, iou_thresh=0.25):
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda x: x[4], reverse=True)
    keep = []
    for current in boxes:
        is_duplicate = False
        for kept in keep:
            if compute_iou(current[:4], kept[:4]) > iou_thresh:
                is_duplicate = True
                break
            cx = (current[0] + current[2]) / 2.0
            cy = (current[1] + current[3]) / 2.0
            if kept[0] <= cx <= kept[2] and kept[1] <= cy <= kept[3]:
                is_duplicate = True
                break
        if not is_duplicate:
            keep.append(current)
    return keep

class SmartReIDTracker:
    def __init__(self, max_distance=350, max_disappeared=150, min_hits=4):
        self.next_id = 1
        self.tracks = {}
        self.max_distance = max_distance
        self.max_disappeared = max_disappeared
        self.min_hits = min_hits

    def update(self, boxes):
        for obj_id, t in self.tracks.items():
            if t["disappeared"] <= 5:
                t["pred_cx"] = t["cx"] + t["vx"] * 0.6
                t["pred_cy"] = t["cy"] + t["vy"] * 0.6
            else:
                t["pred_cx"] = t["cx"]
                t["pred_cy"] = t["cy"]

        if len(boxes) == 0:
            for obj_id in list(self.tracks.keys()):
                self.tracks[obj_id]["disappeared"] += 1
                if self.tracks[obj_id]["disappeared"] > self.max_disappeared:
                    del self.tracks[obj_id]
            return []

        detections = []
        for (x1, y1, x2, y2, conf) in boxes:
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            detections.append((cx, cy, x1, y1, x2, y2, conf))

        used_track_ids = set()
        used_det_indices = set()
        results = []

        pairs = []
        for det_idx, (cx, cy, x1, y1, x2, y2, conf) in enumerate(detections):
            for obj_id, t in self.tracks.items():
                dist = math.hypot(cx - t["pred_cx"], cy - t["pred_cy"])
                if dist < self.max_distance:
                    priority_dist = dist * 0.5 if t["confirmed"] else dist
                    pairs.append((priority_dist, det_idx, obj_id))

        pairs.sort(key=lambda x: x[0])

        for _, det_idx, obj_id in pairs:
            if det_idx in used_det_indices or obj_id in used_track_ids:
                continue

            cx, cy, x1, y1, x2, y2, conf = detections[det_idx]
            t = self.tracks[obj_id]

            if t["disappeared"] <= 2:
                t["vx"] = 0.7 * t["vx"] + 0.3 * (cx - t["cx"])
                t["vy"] = 0.7 * t["vy"] + 0.3 * (cy - t["cy"])
            else:
                t["vx"], t["vy"] = 0.0, 0.0

            t["cx"], t["cy"] = cx, cy
            t["disappeared"] = 0
            t["hits"] += 1
            if t["hits"] >= self.min_hits:
                t["confirmed"] = True

            used_det_indices.add(det_idx)
            used_track_ids.add(obj_id)

            if t["confirmed"]:
                results.append((obj_id, x1, y1, x2, y2, conf))

        unmatched_dets = [i for i in range(len(detections)) if i not in used_det_indices]
        lost_confirmed_tracks = [
            oid for oid, t in self.tracks.items()
            if oid not in used_track_ids and t["confirmed"]
        ]

        if unmatched_dets and lost_confirmed_tracks:
            reid_pairs = []
            for det_idx in unmatched_dets:
                cx, cy = detections[det_idx][0], detections[det_idx][1]
                for oid in lost_confirmed_tracks:
                    dist = math.hypot(cx - self.tracks[oid]["cx"], cy - self.tracks[oid]["cy"])
                    reid_pairs.append((dist, det_idx, oid))
            reid_pairs.sort(key=lambda x: x[0])

            for _, det_idx, obj_id in reid_pairs:
                if det_idx in used_det_indices or obj_id in used_track_ids:
                    continue
                cx, cy, x1, y1, x2, y2, conf = detections[det_idx]
                t = self.tracks[obj_id]
                t["cx"], t["cy"] = cx, cy
                t["vx"], t["vy"] = 0.0, 0.0
                t["disappeared"] = 0
                t["hits"] += 1
                used_det_indices.add(det_idx)
                used_track_ids.add(obj_id)
                results.append((obj_id, x1, y1, x2, y2, conf))

        for det_idx, (cx, cy, x1, y1, x2, y2, conf) in enumerate(detections):
            if det_idx not in used_det_indices:
                obj_id = self.next_id
                self.next_id += 1
                self.tracks[obj_id] = {
                    "cx": cx, "cy": cy,
                    "pred_cx": cx, "pred_cy": cy,
                    "vx": 0.0, "vy": 0.0,
                    "disappeared": 0,
                    "hits": 1,
                    "confirmed": (self.min_hits <= 1)
                }
                if self.tracks[obj_id]["confirmed"]:
                    results.append((obj_id, x1, y1, x2, y2, conf))

        for obj_id in list(self.tracks.keys()):
            if obj_id not in used_track_ids:
                self.tracks[obj_id]["disappeared"] += 1
                if not self.tracks[obj_id]["confirmed"] and self.tracks[obj_id]["disappeared"] > 3:
                    del self.tracks[obj_id]
                elif self.tracks[obj_id]["disappeared"] > self.max_disappeared:
                    del self.tracks[obj_id]

        return results

# --- CẤU HÌNH TRANG WEB GỌN TRONG 1 MÀN HÌNH ---
st.set_page_config(page_title="Giám sát & Đếm Máy Bay", layout="wide")

st.markdown("""
    <style>
        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 0.5rem;
            padding-left: 2rem;
            padding-right: 2rem;
        }
        header[data-testid="stHeader"] {
            height: 2rem;
            background: transparent;
        }
        h3 {
            margin-top: 0 !important;
            padding-top: 0 !important;
            margin-bottom: 0.5rem !important;
        }
        [data-testid="stImage"] img {
            max-height: 78vh !important;
            width: auto !important;
            object-fit: contain;
            margin: 0 auto;
            display: block;
            border-radius: 8px;
        }
    </style>
""", unsafe_allow_html=True)

# --- THANH BÊN TRÁI (SIDEBAR) ---
with st.sidebar:
    st.markdown("### Dữ liệu & Điều khiển")
    uploaded_file = st.file_uploader("Chọn video máy bay", type=["mp4", "avi", "mov"])
    start_btn = st.button("Bắt đầu phân tích", type="primary", use_container_width=True)

    st.divider()
    st.markdown("### Cấu hình mô hình")
    model_choice = st.selectbox(
        "Phiên bản YOLO:",
        ("yolo11n.pt", "yolo11s.pt", "yolov8n.pt")
    )
    img_size = st.select_slider(
        "Độ phân giải (imgsz):",
        options=[640, 800, 960, 1280],
        value=960
    )
    frame_skip = st.slider("Bước nhảy frame:", 1, 5, 2)
    conf_threshold = st.slider("Ngưỡng tin cậy:", 0.10, 0.90, 0.25, 0.05)

@st.cache_resource
def load_model(model_name):
    return YOLO(model_name)

model = load_model(model_choice)

# --- MÀN HÌNH CHÍNH (1 KHUNG HÌNH DUY NHẤT) ---
st.markdown("### Hệ thống Phát hiện và Đếm Máy Bay theo Thời gian thực (YOLO)")

col_video, col_stats = st.columns([3.5, 1])

with col_video:
    frame_placeholder = st.empty()
    if uploaded_file is None:
        frame_placeholder.info("Vui lòng tải video máy bay ở thanh công cụ bên trái và nhấn 'Bắt đầu phân tích'.")

with col_stats:
    st.markdown("**Bảng thống kê**")
    metric_current = st.empty()
    metric_total = st.empty()
    metric_fps = st.empty()
    status_text = st.empty()
    export_placeholder = st.container()

    metric_current.metric(label="Đang trong khung hình", value="0 máy bay")
    metric_total.metric(label="Tổng số đã đếm", value="0 máy bay")
    metric_fps.metric(label="Tốc độ xử lý", value="0.0 FPS")

if uploaded_file is not None and start_btn:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    dynamic_max_dist = max(300, int(frame_width * 0.40))

    # Tự động nhân hệ số nhảy frame nếu video đầu vào là 60fps hoặc 120fps
    effective_skip = frame_skip * max(1, int(round(video_fps / 30.0)))

    # Chuẩn hóa kích thước xuất video và hiển thị để tránh nghẽn bộ nhớ
    target_h = 650
    scale = target_h / max(1, frame_height)
    target_w = int(frame_width * scale)

    out_temp = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    out_path = out_temp.name
    out_writer = cv2.VideoWriter(
        out_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        max(1.0, video_fps / effective_skip),
        (target_w, target_h)
    )

    tracker = SmartReIDTracker(
        max_distance=dynamic_max_dist,
        max_disappeared=150,
        min_hits=4
    )
    confirmed_ids = set()
    detection_logs = {}
    status_text.info(f"Đang phân tích (Video {int(video_fps)} FPS)...")

    frame_idx = 0
    prev_time = time.time()

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame_idx += 1
        if frame_idx % effective_skip != 0:
            continue

        current_sec = round(frame_idx / video_fps, 2)

        results = model.predict(
            frame,
            classes=[4],
            conf=conf_threshold,
            iou=0.45,
            imgsz=img_size,
            verbose=False
        )

        raw_boxes = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            raw_boxes.append((x1, y1, x2, y2, conf))

        clean_boxes = filter_overlapping_boxes(raw_boxes, iou_thresh=0.25)
        tracked_airplanes = tracker.update(clean_boxes)

        for (obj_id, x1, y1, x2, y2, conf) in tracked_airplanes:
            confirmed_ids.add(obj_id)
            display_num = sorted(list(confirmed_ids)).index(obj_id) + 1

            if display_num not in detection_logs:
                detection_logs[display_num] = {
                    "Đối tượng": f"Máy bay #{display_num}",
                    "Xuất hiện (giây)": current_sec,
                    "Cập nhật cuối (giây)": current_sec,
                    "Độ tin cậy Max": round(conf, 2)
                }
            else:
                detection_logs[display_num]["Cập nhật cuối (giây)"] = current_sec
                if round(conf, 2) > detection_logs[display_num]["Độ tin cậy Max"]:
                    detection_logs[display_num]["Độ tin cậy Max"] = round(conf, 2)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            label = f"May bay #{display_num} ({conf:.2f})"
            cv2.putText(frame, label, (x1, max(30, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        resized_frame = cv2.resize(frame, (target_w, target_h))
        out_writer.write(resized_frame)

        curr_time = time.time()
        fps_display = effective_skip / max(0.001, (curr_time - prev_time))
        prev_time = curr_time

        metric_current.metric(label="Đang trong khung hình", value=f"{len(tracked_airplanes)} máy bay")
        metric_total.metric(label="Tổng số đã đếm", value=f"{len(confirmed_ids)} máy bay")
        metric_fps.metric(label="Tốc độ xử lý", value=f"{fps_display:.1f} FPS")

        rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(rgb_frame, channels="RGB")

    cap.release()
    out_writer.release()
    status_text.success(f"Hoàn tất! Tổng: {len(confirmed_ids)} máy bay.")

    with export_placeholder:
        if detection_logs:
            df_logs = pd.DataFrame(list(detection_logs.values()))
            csv_data = df_logs.to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                label="Tải nhật ký (.CSV)",
                data=csv_data,
                file_name="bao_cao_dem_may_bay.csv",
                mime="text/csv",
                use_container_width=True
            )

        with open(out_path, "rb") as video_file:
            st.download_button(
                label="Tải video (.MP4)",
                data=video_file,
                file_name="ket_qua_nhan_dien_may_bay.mp4",
                mime="video/mp4",
                use_container_width=True
            )