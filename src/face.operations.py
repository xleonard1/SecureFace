import uuid
from azure.cognitiveservices.vision.face import FaceClient
from msrest.authentication import CognitiveServicesCredentials
from azure.cognitiveservices.vision.face.models import TrainingStatusType, QualityForRecognition
from config import VISION_KEY, VISION_ENDPOINT
import os

PERSON_GROUP_ID = str(uuid.uuid4())
TARGET_PERSON_GROUP_ID = str(uuid.uuid4())




def initialize_face_client():
    credentials = CognitiveServicesCredentials(VISION_KEY)
    face_client = FaceClient(VISION_ENDPOINT, credentials)
    return face_client

def detect_faces(image_url):
    face_client = initialize_face_client()
    detected_faces = face_client.face.detect_with_url(
        url=image_url, 
        detection_model='detection_03', 
        recognition_model='recognition_04', 
        return_face_attributes=['qualityForRecognition'])
    return detected_faces

def train_person_group(person_group_id): 
    face_client = initialize_face_client()
     # Train the person group
    training_status = face_client.person_group.train(person_group_id)
    return training_status.status

def identify_faces(face_ids, person_group_id):
    face_client = initialize_face_client()
    identified_faces = face_client.face.identify(face_ids, person_group_id)
    print('Identifying faces in image')
    if not identified_faces:
        print('No Person Identified in the person group')
    for identifiedFace in identified_faces:
        if len(identifiedFace.candidates) > 0:
            print('Person is identified for face ID {} in image, with a confidence of {}.'.format(identifiedFace.face_id, identifiedFace.candidates[0].confidence))
            verify_result = face_client.face.verify_face_to_person(identifiedFace.face_id, identifiedFace.candidates[0].person_id, person_group_id)
            print('verification result: {}. confidence: {}'.format(verify_result.is_identical, verify_result.confidence))
        else:
            print('No person identified for face ID {} in image.'.format(identifiedFace.face_id))
    