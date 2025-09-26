import os
import sys
import argparse
import glob
import time

import cv2
import numpy as np
from ultralytics import YOLO

# -------------------------------
# Argomenti CLI
# -------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True,
                    help='Percorso al file modello YOLO (es. runs/detect/train/weights/best.pt)')
parser.add_argument('--source', required=True,
                    help=('Sorgente: file immagine ("test.jpg"), cartella ("imgs/"), '
                          'file video ("video.mp4"), indice camera numerico ("0"), '
                          'USB cam ("usb0"), Picamera ("picamera0"), '
                          'oppure alias per webcam Mac: "mac", "webcam", "maccam"'))
parser.add_argument('--thresh', type=float, default=0.5,
                    help='Soglia di confidenza minima per mostrare i rilevamenti (es. 0.4)')
parser.add_argument('--resolution', default=None,
                    help='Risoluzione WxH per la visualizzazione/registrazione (es. 1280x720). '
                         'Se omessa, usa quella della sorgente.')
parser.add_argument('--record', action='store_true',
                    help='Registra il video risultante in "demo1.avi". Richiede --resolution.')
parser.add_argument('--backend', choices=['auto', 'avf', 'qt'], default='auto',
                    help='Backend OpenCV per la webcam. Su macOS consiglia "avf" (AVFoundation).')
args = parser.parse_args()

# -------------------------------
# Parsing input utente
# -------------------------------
model_path = args.model
img_source = args.source
min_thresh = float(args.thresh)
user_res = args.resolution
record = args.record
backend = args.backend

# -------------------------------
# Verifica modello
# -------------------------------
if not os.path.exists(model_path):
    print('ERROR: Model path is invalid or model not found.')
    sys.exit(0)

# Carica modello
model = YOLO(model_path, task='detect')
labels = model.names

# -------------------------------
# Determina tipo di sorgente
# -------------------------------
img_ext_list = ['.jpg','.JPG','.jpeg','.JPEG','.png','.PNG','.bmp','.BMP']
vid_ext_list = ['.avi','.mov','.mp4','.mkv','.wmv']

source_type = None
usb_idx = None
picam_idx = None
device_idx = None

if os.path.isdir(img_source):
    source_type = 'folder'
elif os.path.isfile(img_source):
    _, ext = os.path.splitext(img_source)
    if ext in img_ext_list:
        source_type = 'image'
    elif ext in vid_ext_list:
        source_type = 'video'
    else:
        print(f'File extension {ext} is not supported.')
        sys.exit(0)
elif 'usb' in img_source:
    source_type = 'usb'
    usb_idx = int(img_source[3:])
elif 'picamera' in img_source:
    source_type = 'picamera'
    picam_idx = int(img_source[8:])
# Nuove opzioni per webcam Mac / indice numerico generico
elif img_source.isdigit():
    source_type = 'device'
    device_idx = int(img_source)
elif img_source.lower() in ['mac', 'webcam', 'maccam', 'defaultcam', 'integrated', 'isight']:
    source_type = 'device'
    device_idx = 0
else:
    print(f'Input {img_source} is invalid. Please try again.')
    sys.exit(0)

# -------------------------------
# Risoluzione richiesta
# -------------------------------
resize = False
resW = resH = None
if user_res:
    try:
        resW, resH = int(user_res.split('x')[0]), int(user_res.split('x')[1])
        resize = True
    except Exception:
        print('ERROR: --resolution deve essere nel formato WxH, ad es. 1280x720')
        sys.exit(0)

# -------------------------------
# Setup registrazione (se richiesto)
# -------------------------------
if record:
    if source_type not in ['video','usb','device','picamera']:
        print('Recording only works for video and camera sources. Please try again.')
        sys.exit(0)
    if not user_res:
        print('Please specify --resolution to record video.')
        sys.exit(0)
    record_name = 'demo1.avi'
    record_fps = 30
    recorder = cv2.VideoWriter(record_name, cv2.VideoWriter_fourcc(*'MJPG'), record_fps, (resW, resH))

# -------------------------------
# Inizializza la sorgente
# -------------------------------
if source_type == 'image':
    imgs_list = [img_source]

elif source_type == 'folder':
    imgs_list = []
    for file in glob.glob(os.path.join(img_source, '*')):
        _, file_ext = os.path.splitext(file)
        if file_ext in img_ext_list:
            imgs_list.append(file)
    imgs_list.sort()

elif source_type in ['video', 'usb', 'device']:
    if source_type == 'video':
        cap_arg = img_source
        cap = cv2.VideoCapture(cap_arg)
    else:
        # Scegli backend per macOS
        if sys.platform == 'darwin':
            if backend == 'avf' or backend == 'auto':
                cap = cv2.VideoCapture(device_idx if source_type == 'device' else usb_idx, cv2.CAP_AVFOUNDATION)
            elif backend == 'qt':
                cap = cv2.VideoCapture(device_idx if source_type == 'device' else usb_idx, cv2.CAP_QT)
            else:
                cap = cv2.VideoCapture(device_idx if source_type == 'device' else usb_idx)
        else:
            cap = cv2.VideoCapture(device_idx if source_type == 'device' else usb_idx)

    if user_res:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, resW)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resH)

    if not cap.isOpened():
        print('ERROR: Unable to open the camera/video source. '
              'Su macOS prova --backend avf e verifica i permessi della fotocamera (Privacy & Security).')
        sys.exit(0)

elif source_type == 'picamera':
    from picamera2 import Picamera2
    cap = Picamera2()
    # Se non fornita, imposta una risoluzione tipica
    if not user_res:
        resW, resH = 1280, 720
        resize = True
    cap.configure(cap.create_video_configuration(main={"format": 'RGB888', "size": (resW, resH)}))
    cap.start()

# -------------------------------
# Colori bbox e variabili stato
# -------------------------------
bbox_colors = [(164,120,87), (68,148,228), (93,97,209), (178,182,133), (88,159,106),
               (96,202,231), (159,124,168), (169,162,241), (98,118,150), (172,176,184)]
avg_frame_rate = 0.0
frame_rate_buffer = []
fps_avg_len = 200
img_count = 0

# -------------------------------
# Loop di inferenza
# -------------------------------
while True:
    t_start = time.perf_counter()

    # Carica frame
    if source_type in ['image', 'folder']:
        if img_count >= len(imgs_list):
            print('All images have been processed. Exiting program.')
            break
        img_filename = imgs_list[img_count]
        frame = cv2.imread(img_filename)
        img_count += 1

    elif source_type == 'video':
        ret, frame = cap.read()
        if not ret:
            print('Reached end of the video file. Exiting program.')
            break

    elif source_type in ['usb', 'device']:
        ret, frame = cap.read()
        if (frame is None) or (not ret):
            print('Unable to read frames from the camera. It may be disconnected or unavailable. Exiting.')
            break

    elif source_type == 'picamera':
        frame = cap.capture_array()
        if frame is None:
            print('Unable to read frames from the Picamera. Exiting.')
            break

    # Resize per display/record se richiesto
    if resize:
        frame = cv2.resize(frame, (resW, resH))

    # Inference
    results = model(frame, verbose=False)
    detections = results[0].boxes

    object_count = 0

    # Disegno bbox
    for i in range(len(detections)):
        xyxy_tensor = detections[i].xyxy.cpu()
        xyxy = xyxy_tensor.numpy().squeeze()
        xmin, ymin, xmax, ymax = xyxy.astype(int)

        classidx = int(detections[i].cls.item())
        classname = labels[classidx]
        conf = float(detections[i].conf.item())

        if conf >= min_thresh:
            color = bbox_colors[classidx % len(bbox_colors)]
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)

            label = f'{classname}: {int(conf*100)}%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_ymin = max(ymin, labelSize[1] + 10)
            cv2.rectangle(frame, (xmin, label_ymin - labelSize[1] - 10),
                          (xmin + labelSize[0], label_ymin + baseLine - 10), color, cv2.FILLED)
            cv2.putText(frame, label, (xmin, label_ymin - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            object_count += 1

    # FPS overlay per stream/video
    if source_type in ['video', 'usb', 'device', 'picamera']:
        cv2.putText(frame, f'FPS: {avg_frame_rate:0.2f}', (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 255, 255), 2)

    # Count overlay
    cv2.putText(frame, f'Number of objects: {object_count}', (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 255, 255), 2)

    cv2.imshow('YOLO detection results', frame)
    if record:
        recorder.write(frame)

    # Key handling
    if source_type in ['image', 'folder']:
        key = cv2.waitKey()
    else:
        key = cv2.waitKey(5)

    if key == ord('q') or key == ord('Q'):
        break
    elif key == ord('s') or key == ord('S'):
        cv2.waitKey()
    elif key == ord('p') or key == ord('P'):
        cv2.imwrite('capture.png', frame)

    # FPS calcolo medio
    t_stop = time.perf_counter()
    frame_rate_calc = float(1 / max(1e-6, (t_stop - t_start)))
    if len(frame_rate_buffer) >= fps_avg_len:
        frame_rate_buffer.pop(0)
    frame_rate_buffer.append(frame_rate_calc)
    avg_frame_rate = float(np.mean(frame_rate_buffer))

# -------------------------------
# Cleanup
# -------------------------------
print(f'Average pipeline FPS: {avg_frame_rate:.2f}')
if source_type in ['video', 'usb', 'device']:
    cap.release()
elif source_type == 'picamera':
    cap.stop()
if record:
    recorder.release()
cv2.destroyAllWindows()
