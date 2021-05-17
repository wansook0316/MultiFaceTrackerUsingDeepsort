from retinaface import RetinaFace
import cv2

def build_model():
    face_detector = RetinaFace.build_model()
    return face_detector

def detect_face(face_detector, img, align = True):

    face = None
    img_region = [0, 0, img.shape[0], img.shape[1]]

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #retinaface expects RGB but OpenCV read BGR

    faces = RetinaFace.extract_faces(img_rgb, model = face_detector, align = align)

    if len(faces) > 0:
        face = faces[0][:, :, ::-1]

    return face, img_region
