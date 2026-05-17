import torch
from train import get_model, preprocess
import cv2
import mediapipe as mp


model = get_model()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

model.load_state_dict(torch.load('emotion_transformer.pt'))

classes = ['Ahegao', 'Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']

def find_emotion(image):
    image = preprocess(image)
    out = model(image)
    _, pred = torch.max(out, dim = 1)
    pred = pred.squeeze(0)
    return classes[pred.tolist()]

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

face_detection = mp_face_detection.FaceDetection(
    model_selection=0,      
    min_detection_confidence=0.5,
    
    
)

cap = cv2.VideoCapture(0)

# Optional: Set resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_detection.process(rgb_frame)

    if results.detections:
        for detection in results.detections:
            # detection = results.detections[0]
            mp_drawing.draw_detection(frame, detection)

            # Get bounding box coordinates
            bbox = detection.location_data.relative_bounding_box
            h, w, _ = frame.shape

            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            bw = int(bbox.width * w)
            bh = int(bbox.height * h)

            
            confidence = int(detection.score[0] * 100)
            x = max(0, x)
            y = max(0, y)

            
            x2 = min(w, x + bw)
            y2 = min(h, y + bh)

            
            face_crop = frame[y:y2, x:x2]

            emotion =  find_emotion(face_crop)
            
            cv2.rectangle(
                frame,
                (x, y),
                (x + bw, y + bh),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Emotion :{emotion}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )
            

    cv2.imshow("MediaPipe Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

