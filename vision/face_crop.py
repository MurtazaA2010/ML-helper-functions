from PIL import Image
import face_recognition

def crop_face(img_path):
    img = face_recognition.load_image_file(img_path)
    face_loc = face_recognition.face_locations(img)
    if len(face_loc) == 0:
        return None 
    top, right, bottom, left = face_loc[0] 
    face_img = img[top:bottom, left:right]
    face_img = Image.fromarray(face_img)
    return face_img
