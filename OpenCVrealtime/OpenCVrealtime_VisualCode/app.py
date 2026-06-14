from flask import Flask, Response, render_template
import cv2
import numpy as np

app = Flask(__name__)

# โหลด model
CLASSES = ["BACKGROUND", "AEROPLANE", "BICYCLE", "BIRD", "BOAT",
           "BOTTLE", "BUS", "CAR", "CAT", "CHAIR", "COW", "DININGTABLE",
           "DOG", "HORSE", "MOTORBIKE", "PERSON", "POTTEDPLANT",
           "SHEEP", "SOFA", "TRAIN", "TVMONITOR"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
net = cv2.dnn.readNetFromCaffe("./MobileNetSSD/MobileNetSSD.prototxt",
                                "./MobileNetSSD/MobileNetSSD.caffemodel")

def generate():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                      0.007843, (300, 300), 127.5)
        net.setInput(blob)
        detections = net.forward()
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                idx = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x1, y1, x2, y2) = box.astype("int")
                label = f"{CLASSES[idx]}: {confidence:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), COLORS[idx], 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return '''
    <html>
    <head><title>Object Detection</title></head>
    <body style="background:#111; text-align:center; color:white;">
        <h1>🎥 Object Detection</h1>
        <img src="/video" style="width:800px; border:2px solid #fff;">
    </body>
    </html>
    '''

@app.route('/video')
def video():
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)