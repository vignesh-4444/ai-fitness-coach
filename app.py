from flask import Flask, render_template, request, Response, jsonify,redirect, url_for
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import imutils
import time
import pygame


app = Flask(__name__)


# Inizializza pygame
pygame.init()

# Carica il suono
perfectSound = pygame.mixer.Sound('static/good.mp3')


""""
NOSE = 0
  LEFT_EYE_INNER = 1
  LEFT_EYE = 2
  LEFT_EYE_OUTER = 3
  RIGHT_EYE_INNER = 4
  RIGHT_EYE = 5
  RIGHT_EYE_OUTER = 6
  LEFT_EAR = 7
  RIGHT_EAR = 8
  MOUTH_LEFT = 9
  MOUTH_RIGHT = 10
  LEFT_SHOULDER = 11
  RIGHT_SHOULDER = 12
  LEFT_ELBOW = 13
  RIGHT_ELBOW = 14
  LEFT_WRIST = 15
  RIGHT_WRIST = 16
  LEFT_PINKY = 17
  RIGHT_PINKY = 18
  LEFT_INDEX = 19
  RIGHT_INDEX = 20
  LEFT_THUMB = 21
  RIGHT_THUMB = 22
  LEFT_HIP = 23
  RIGHT_HIP = 24
  LEFT_KNEE = 25
  RIGHT_KNEE = 26
  LEFT_ANKLE = 27
  RIGHT_ANKLE = 28
  LEFT_HEEL = 29
  RIGHT_HEEL = 30
  LEFT_FOOT_INDEX = 31
  RIGHT_FOOT_INDEX = 32
"""
diz = {"curl":0, "squat":0, "flessioni":0, "plank":0}
action = None
reps = 0
msg = ""
time_diff = None

METS_MAN = 4.5
METS_WOMAN = 3.5


@app.template_filter()
def firstHalfExercises(diz):
    half_length = (len(diz) // 2)
    return dict(list(diz.items())[0:half_length])


@app.template_filter()
def secondHalfExercises(diz):
    half_length = (len(diz) // 2)
    return dict(list(diz.items())[half_length:])


colors = [(245,117,16), (117,245,16), (16,117,245),(45,110,160)]
actions = np.array(['curl', 'squat','flessioni','null'])

@app.route('/get_data')
def get_data():
    return jsonify(diz,action,reps,msg,time_diff)


@app.route('/', methods=['GET', 'POST'])
def home():
    form_type = request.form.get('form_type')
    if form_type == 'modal':
        #form della modale
        email = request.form.get('email')
        age = request.form.get('age')
        height = request.form.get('height')
        weight = request.form.get('weight')
        gender = request.form.get('gender')

        return redirect(url_for('gym', email=email, age=age, height=height, weight=weight,gender=gender))
    elif form_type == 'footer': 
        #form del footer
        email = request.form.get('email')
        age = request.form.get('age')
        height = request.form.get('height')
        weight = request.form.get('weight')
        gender = request.form.get('gender')
        
        return redirect(url_for('gym', email=email, age=age, height=height, weight=weight,gender=gender))

    return render_template('home.html')
    


@app.route('/gym/',methods=['GET', 'POST'])
def gym():
    gender = request.args.get('gender')
    email = request.args.get('email')
    age = request.args.get('age')
    height = request.args.get('height')
    weight = request.args.get('weight')

    if request.method == 'POST':
        #return redirect("/")
        return redirect(url_for('stats', email=email, age=age, height=height, weight=weight,gender=gender))
    return render_template('gym.html', email=email, age=age, height=height, weight=weight, diz=diz,gender=gender)


@app.route('/gym/stats/',methods=['GET', 'POST'])
def stats():
    local_diz = diz.copy()
    print(local_diz)
    for key in diz:
        diz[key] = 0
    print(local_diz)
    email = request.args.get('email')
    age = request.args.get('age')
    height = request.args.get('height')
    weight = request.args.get('weight')
    gender = request.args.get('gender')
    kcal = 0.0
    rounded = 0
    rounded = int(local_diz["plank"]) 
    local_diz["plank"] = rounded 
    for x in local_diz.keys():
        kcal += compute_kcal(local_diz,gender,weight,x)
    print(kcal)
    

    return render_template('stats.html', email=email,age=age, height=height, weight=weight, gender=gender,kcal=round(kcal,2),local_diz=local_diz)


"""Return = risposta HTTP contenente un flusso di dati video in formato multipart/x-mixed-replace --> flusso di dati che verrà
aggiornato costantemente sostituendo ogni immagine precedente.
"""
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities
mp_pose = mp.solutions.pose
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results

def draw_styled_landmarks(image, results):
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)) 

     



def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    try:
        #print(len(results.pose_landmarks.landmark))
        if float(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_FOOT_INDEX].visibility) <= 0.001:
            #print("piede non visibile")
            pass
        else:
            #print("piede visibile ",results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_FOOT_INDEX]) 
            #print("corretto ",results.pose_landmarks.landmark)
            pass
    except:
        #print("errore ", results.pose_landmarks)
        pass
    return np.concatenate([pose])

""" 
# OLD VERSION
def check_visibility(results):
    try:
        for res in results.pose_landmarks.landmark:
            if float(res.visibility) <= 0.3:
                return False
    except:
        return False
    return True
"""

def check_visibility(results):
    """
    Il landmark 0 corrisponde al NASO (NOSE)
    Il landmark 11 corrisponde alla SPALLA SINISTRA (LEFT_SHOULDER)
    Il landmark 12 corrisponde alla SPALLA DESTRA (RIGHT_SHOULDER)
    Il landmark 13 corrisponde al GOMITO SINISTRO (LEFT_ELBOW)
    Il landmark 14 corrisponde al GOMITO DESTRO (RIGHT_ELBOW)
    Il landmark 15 corrisponde al POLSO SINISTRO (LEFT_WRIST)
    Il landmark 16 corrisponde al POLSO DESTRO (RIGHT_WRIST)
    Il landmark 23 corrisponde all'ANCA SINISTRA (LEFT_HIP)
    Il landmark 24 corrisponde all'ANCA DESTRA (RIGHT_HIP)
    Il landmark 25 corrisponde al GINOCCHIO SINISTRO (LEFT_KNEE)
    Il landmark 26 corrisponde al GINOCCHIO DESTRO (RIGHT_KNEE)
    """
    main_landmarks = [0, 11, 12, 13, 14, 15, 16, 23, 24] # Lista dei landmark principali
    try:
        for index in main_landmarks:
            landmark = results.pose_landmarks.landmark[index]
            if float(landmark.visibility) <= 0.3:
                return False
    except:
        return False
    return True

def classify_speed(time,action=None):
    if action == "curl":
        if time >= 0.8 and time < 1.4: 
            perfectSound.play()
            return "Perfect"
        elif time > 1.5:
            return "Good"
        elif time < 0.8:
            return "Too fast.."
    if action == "squat":
        if time >= 0.8 and time < 1.4: 
            perfectSound.play()
            return "Perfect"
        elif time > 1.5:
            return "Good"
        elif time < 0.8:
            return "Too fast.."
    if action == "flessioni":
        if time >= 0.8 and time < 1.4: 
            perfectSound.play()
            return "Perfect"
        elif time > 1.5:
            return "Good"
        elif time < 0.8:
            return "Too fast.."
    



model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,132)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

#LOAD WEIGHTS
model.load_weights('action.h5')

def calculate_angle(a,b,c): # teh tree points to create angles
    a = np.array(a) # We convert all to numpy arrays
    b = np.array(b)
    c = np.array(c)
    radiants = np.arctan2(c[1]-b[1],c[0]-b[0]) - np.arctan2(a[1]-b[1],a[0]-b[0])
    angle= np.abs(radiants*180.0/np.pi)
    if angle > 180:# convert angle between 0 and 180    
        angle = 360 - angle
    return angle 

""" METS X 3.5 X BW (kg) / 200 = KCAL/MIN. """
def compute_kcal(local_diz,sex,kg,exercize):
    if sex == "male":
        MET = METS_MAN
    elif sex == "other":
        MET = METS_MAN
    else:
        MET = METS_WOMAN
    kcal_1m = (MET * 3.5 * float(kg)) / 200 # equivalent to 25 exercise 
     ##25 : kcal_1m = 36 : y
    print(exercize)
    if exercize != "plank":
        return round((kcal_1m * local_diz[exercize])/25,1)
    else: return round((((kcal_1m * local_diz[exercize])/25)/2),1)

   

    
def compute_curl(landmarks) : 
    mp_pose = mp.solutions.pose
    l_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y ]
    l_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y ]
    l_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y ]
    r_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y ]
    r_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y ]
    r_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y ]
    left_angle = calculate_angle(l_shoulder,l_elbow,l_wrist)
    right_angle  = calculate_angle(r_shoulder,r_elbow,r_wrist)
    return left_angle, right_angle


def compute_squat(landmarks):
    mp_pose = mp.solutions.pose
    r_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y ]
    r_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y ]
    r_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y ]
    l_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y ]
    l_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y ]
    l_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y ]
    right_angle = calculate_angle(r_hip,r_knee,r_ankle)
    left_angle = calculate_angle(l_hip,l_knee,l_ankle)
    return right_angle,left_angle



def gen_frames():
    global reps
    global action
    global msg
    global time_diff
    last_down = {'curl': None, 'squat': None, 'flessioni': None, 'plank': None }
    
    sequence = []
    actions_state = ["null"]
    counter = 0
    stage = {}
    threshold = 0.95
    
    exercise_landmarks = {
        'curl': {'left': 17, 'right': 18, 'angle_threshold': 160},
        'squat': {'left': 25, 'right': 26, 'angle_threshold': 155},
        'flessioni': {'left': 17, 'right': 18, 'angle_threshold': 90},
        'plank': {'angle_threshold': 150}
    }

    show_webcam_only = True
    start_time = time.time()
    countdown = 10
 
    cap = cv2.VideoCapture(0)

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()

            if show_webcam_only and time.time() - start_time < countdown:
                ret, jpeg = cv2.imencode('.jpg', frame)
                frame = jpeg.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                continue
            else:
                show_webcam_only = False
            
            image, results = mediapipe_detection(frame, holistic)
            keypoints = extract_keypoints(results)
            
            sequence.append(keypoints)
            sequence = sequence[-30:]
            
            if check_visibility(results):
                if len(sequence) == 30:
                    res = model.predict(np.expand_dims(sequence, axis=0))[0]
                    if res[np.argmax(res)] > threshold:
                        action = actions[np.argmax(res)]
                        if action != 'null':
                            actions_state.append(action)
                            if action != actions_state[-2]:
                                counter = 0
                                stage[action] = None
                                
                        try:
                            if results.pose_landmarks:
                                landmarks = results.pose_landmarks.landmark
                                l_curlangle, r_curlangle = compute_curl(landmarks)
                                r_squatangle, l_squatangle = compute_squat(landmarks)
                                draw_styled_landmarks(image, results)
                                
                                if action in exercise_landmarks:
                                    exercise = exercise_landmarks[action]
                                    angle_threshold = exercise['angle_threshold']
                                    if 'left' in exercise and 'right' in exercise:
                                        left_landmark = exercise['left']
                                        right_landmark = exercise['right']
                                        if l_curlangle > angle_threshold or r_curlangle > angle_threshold:
                                            stage[action] = "down"
                                            if not last_down[action]:
                                                last_down[action] = time.time()
                                        if (l_curlangle < 40 or r_curlangle < 40) and stage[action] == "down":
                                            stage[action] = "up"
                                            if last_down[action] is not None:
                                                time_diff = round(time.time() - last_down[action], 4)
                                                msg = classify_speed(time_diff, action)
                                                last_down[action] = None
                                            counter += 1
                                            if action != 'plank':
                                                diz[action] += 1
                                    else:
                                        if l_curlangle < angle_threshold or r_curlangle < angle_threshold:
                                            stage[action] = "down"
                                            if not last_down[action]:
                                                last_down[action] = time.time()
                                        if l_curlangle > angle_threshold or r_curlangle > angle_threshold and stage[action] == "down":
                                            stage[action] = "up"
                                            if last_down[action] is not None:
                                                time_diff = round(time.time() - last_down[action], 4)
                                                msg = classify_speed(time_diff, action)
                                                last_down[action] = None
                                            counter += 1
                                            if action != 'plank':
                                                diz[action] += 1
                                    
                                    # Check for plank during prolonged stable position of push-ups
                                    if action == 'flessioni':
                                        listOfStrings = actions_state[-10:]
                                        if len(listOfStrings) > 0 and all(elem == 'flessioni' for elem in listOfStrings) and counter == 0:
                                            if reps == 0:
                                                clock = time.time()
                                            action = "plank"
                                            reps += 0.2
                                            reps = round(reps, 1)  # Arrotonda a 1 cifra decimale
                                            diz[action] += 0.2
                                            diz[action] = round(diz[action], 1)  # Arrotonda a 1 cifra decimale

                                    
                        except Exception as e:
                            pass

                    if action != 'plank':
                        reps = counter

            else:
                msg = "⚠️Please move away from the camera a bit"

            ret, jpeg = cv2.imencode('.jpg', image)
            frame = jpeg.tobytes()
            yield (b'--frame\r\n' 
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')




    
        


if __name__ == '__main__':
    app.run(debug=True)
