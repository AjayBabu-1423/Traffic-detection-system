from django.http import JsonResponse
from django.shortcuts import render, redirect
from . models import UserPersonalModel
from . forms import UserPersonalForm, UserRegisterForm
from django.contrib.auth import authenticate, login,logout
from django.contrib import messages
import numpy as np
import joblib
from django.core.files.storage import FileSystemStorage 

# import pyperclip
import random


# Create your views here.
def passhome(request):
    return render(request, 'passwordGenerator/passhome.html')


def Landing_1(request):
    return render(request, '1_Landing.html')

def Register_2(request):
    form = UserRegisterForm()
    if request.method =='POST':
        form = UserRegisterForm(request.POST)
        if form.is_valid():
            form.save()
            user = form.cleaned_data.get('username')
            messages.success(request, 'Account was successfully created. ' + user)
            return redirect('Login_3')

    context = {'form':form}
    return render(request, '2_Register.html', context)


def Login_3(request):
    if request.method =='POST':
        username = request.POST.get('username')
        password = request.POST.get('password')

        user = authenticate(username=username, password=password)

        if user is not None:
            login(request, user)
            return redirect('Home_4')
        else:
            messages.info(request, 'Username OR Password incorrect')

    context = {}
    return render(request,'3_Login.html', context)

def Home_4(request):
    return render(request, '4_Home.html')

def Teamates_5(request):
    return render(request,'5_Teamates.html')

def Domain_Result_6(request):
    return render(request,'6_Domain_Result.html')

def Problem_Statement_7(request):
    return render(request,'7_Problem_Statement.html')
    

def Per_Info_8(request):
    if request.method == 'POST':
        fieldss = ['firstname','lastname','age','address','phone','city','state','country']
        form = UserPersonalForm(request.POST)
        if form.is_valid():
            print('Saving data in Form')
            form.save()
        return render(request, '4_Home.html', {'form':form})
    else:
        print('Else working')
        form = UserPersonalForm(request.POST)    
        return render(request, '8_Per_Info.html', {'form':form})
    


    






import cv2
import cvzone
import math
import time
import tkinter as tk
from tkinter import StringVar
from ultralytics import YOLO


def Deploy_8(request):
    if request.method == 'POST':
        # Initialize the YOLO model
        model = YOLO('APP/Vehicles.pt')
        classnames = ['Ambulance', 'Misc Vehicle', 'Siren', 'Label']
        allowed_classes = ['Ambulance']
        # Initialize webcam
        cap = cv2.VideoCapture(0)

        # Tkinter GUI for Signal Display
        def start_gui():
            # Create Tkinter window
            root = tk.Tk()
            root.title("Traffic Signal Control System")
            root.geometry("400x200")

            # Label to display signal status
            signal_status = StringVar()
            signal_status.set("Red Light: No Ambulance Detected")

            # Signal label
            signal_label = tk.Label(root, textvariable=signal_status, bg="red", fg="white", font=("Helvetica", 16), width=30, height=5)
            signal_label.pack(pady=20)

            # Close Tkinter window gracefully
            def close_gui():
                root.destroy()
                cap.release()
                cv2.destroyAllWindows()

            # Close button
            close_button = tk.Button(root, text="Stop Detection", command=close_gui, font=("Helvetica", 14), bg="blue", fg="white")
            close_button.pack(pady=10)

            # Detection loop
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Resize frame for consistency
                frame = cv2.resize(frame, (640, 480))

                # Default red light
                cv2.rectangle(frame, (550, 150), (600, 300), (255, 255, 255), -1)  # Traffic light background
                cv2.circle(frame, (575, 200), 40, (0, 0, 255), -1)  # Red light

                # Perform YOLO detection
                results = model(frame, stream=True)
                for info in results:
                    boxes = info.boxes
                    for box in boxes:
                        confidence = math.ceil((box.conf[0] * 100)) / 100
                        detected_class_index = int(box.cls[0])

                        if confidence >= 0.65:
                            detected_class = classnames[detected_class_index]
                            if detected_class in allowed_classes:
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 5)
                                cvzone.putTextRect(frame, f'{detected_class} {confidence}%', [x1 + 8, y1 + 100],
                                                scale=1.5, thickness=2)

                                if detected_class == "Ambulance":
                                    signal_label.config(bg="green")
                                    signal_status.set("Green Light: Ambulance Detected")
                                    # Update to green light
                                    cv2.rectangle(frame, (550, 150), (600, 300), (255, 255, 255), -1)  # Traffic light background
                                    cv2.circle(frame, (575, 200), 40, (0, 255, 0), -1)
                                    break
                                else:
                                    signal_label.config(bg="red")
                                    signal_status.set("Red Light: No Ambulance Detected")

                cv2.imshow('Real-Time Detection', frame)

                if cv2.waitKey(1) & 0xFF == 27:  # Exit on 'ESC'
                    break

            cap.release()
            cv2.destroyAllWindows()
            root.destroy()

        # Start Tkinter GUI
        start_gui()

        # Render to the Django template
        return render(request, 'ambulance.html')

    else:
        # Render the default page when no POST request
        return render(request, 'ambulance.html')
    

import cv2
import time
import math
import numpy as np
from collections import OrderedDict
from ultralytics import YOLO
from django.shortcuts import render

class CentroidTracker:
    def __init__(self, max_distance=50):
        self.next_object_id = 1
        self.objects = OrderedDict()
        self.max_distance = max_distance

    def register(self, centroid):
        """ Register a new vehicle with a unique ID. """
        self.objects[self.next_object_id] = centroid
        self.next_object_id += 1

    def update(self, new_centroids):
        """ Update tracked vehicles and avoid duplicates. """
        updated_ids = set()
        for new_c in new_centroids:
            matched = False
            for obj_id, existing_c in self.objects.items():
                distance = np.linalg.norm(np.array(existing_c) - np.array(new_c))
                if distance < self.max_distance:
                    self.objects[obj_id] = new_c
                    updated_ids.add(obj_id)
                    matched = True
                    break
            
            if not matched:
                self.register(new_c)

        return len(self.objects)  

def calculate_signal_time(vehicle_count):
    """ Calculate the signal time based on the vehicle count. """
    if 20 < vehicle_count <= 30:
        return 25
    elif vehicle_count > 30:
        return 20
    elif 10 < vehicle_count <= 20:
        return 15
    elif 6 < vehicle_count <= 10:
        return 10
    else:
        return 5




def Deploy_9(request):
    if request.method == 'POST':
        model = YOLO('APP/vehical1.pt')

        video_files = [request.FILES.get(f'video{i}') for i in range(1, 5)]
        video_files = [f for f in video_files if f]

        if not video_files:
            return render(request, '9_Deploy.html', {'error': 'No videos uploaded'})

        caps = [cv2.VideoCapture(video_file.temporary_file_path()) for video_file in video_files]
        signal_times = []
        vehicle_counts = []
        monitoring_time = 20
        active_video_index = 0

        while active_video_index < len(video_files):
            start_time = time.time()
            print(f"Monitoring video {active_video_index + 1} for {monitoring_time} seconds...")

            cap = caps[active_video_index]
            tracker = CentroidTracker(max_distance=50)
            final_vehicle_count = 0

            while time.time() - start_time < monitoring_time:
                ret, frame = cap.read()
                if not ret:
                    print(f"End of video {active_video_index + 1}.")
                    break

                frame = cv2.resize(frame, (640, 480))
                results = model(frame, stream=True)

                centroids = []
                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        confidence = math.ceil(box.conf[0] * 100)
                        class_id = int(box.cls[0])

                        if confidence > 30 and class_id != 0:
                            centroid = ((x1 + x2) // 2, (y1 + y2) // 2)
                            centroids.append(centroid)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(frame, f'{confidence}%', (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                unique_vehicle_count = tracker.update(centroids)
                final_vehicle_count = unique_vehicle_count

                # Countdown timer
                elapsed_time = int(time.time() - start_time)
                remaining_time = max(0, monitoring_time - elapsed_time)
                cv2.putText(frame, f"Monitoring Time: {remaining_time}s", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # Vehicle count
                cv2.putText(frame, f"Vehicles: {unique_vehicle_count}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                # Red = active, green = gray
                cv2.circle(frame, (600, 50), 20, (0, 0, 255), -1)
                cv2.putText(frame, "RED", (580, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                cv2.circle(frame, (600, 120), 20, (128, 128, 128), -1)
                cv2.putText(frame, "GREEN", (570, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 2)

                cv2.imshow(f"Video {active_video_index + 1}", frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # Monitoring complete, now show GREEN for this video
            vehicle_counts.append(final_vehicle_count)
            signal_time = calculate_signal_time(final_vehicle_count)
            signal_times.append(signal_time)

            print(f"Completed monitoring video {active_video_index + 1}. Signal time: {signal_time} seconds")

            done_time = time.time()
            while time.time() - done_time < 2:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.resize(frame, (640, 480))

                # Green ON
                cv2.circle(frame, (600, 120), 20, (0, 255, 0), -1)
                cv2.putText(frame, "GREEN", (570, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Red OFF
                cv2.circle(frame, (600, 50), 20, (128, 128, 128), -1)
                cv2.putText(frame, "RED", (580, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 2)

                cv2.putText(frame, "Monitoring Completed", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                cv2.imshow(f"Video {active_video_index + 1}", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # 🔁 After green phase ends, reset previous video to RED
            if active_video_index > 0:
                prev_index = active_video_index - 1
                cap_prev = caps[prev_index]
                print(f"Resetting video {prev_index + 1} to RED")
                reset_start = time.time()
                while time.time() - reset_start < 2:
                    ret, frame_prev = cap_prev.read()
                    if not ret:
                        break
                    frame_prev = cv2.resize(frame_prev, (640, 480))

                    # Red ON
                    cv2.circle(frame_prev, (600, 50), 20, (0, 0, 255), -1)
                    cv2.putText(frame_prev, "RED", (580, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                    # Green OFF
                    cv2.circle(frame_prev, (600, 120), 20, (128, 128, 128), -1)
                    cv2.putText(frame_prev, "GREEN", (570, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 2)

                    cv2.putText(frame_prev, "Reset to RED", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                    cv2.imshow(f"Video {prev_index + 1}", frame_prev)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

            monitoring_time = signal_time
            active_video_index += 1

        for cap in caps:
            cap.release()
        cv2.destroyAllWindows()

        return render(request, '9_Deploy.html', {
            'signal_times': signal_times,
            'vehicle_counts': vehicle_counts,
            'video_labels': [f"Video {i + 1}" for i in range(len(video_files))]
        })

    else:
        return render(request, '9_Deploy.html')


#final




import cv2
import time
import math
import numpy as np
from collections import OrderedDict
from ultralytics import YOLO

import cv2
import time
import math
import numpy as np
from collections import OrderedDict
from ultralytics import YOLO
from django.shortcuts import render

# Centroid Tracker to Track Unique Vehicles
class CentroidTracker:
    def __init__(self, max_distance=50):
        self.next_object_id = 1
        self.objects = OrderedDict()
        self.max_distance = max_distance

    def register(self, centroid):
        """ Register a new vehicle with a unique ID. """
        self.objects[self.next_object_id] = centroid
        self.next_object_id += 1

    def update(self, new_centroids):
        """ Update tracked vehicles and avoid duplicates. """
        updated_ids = set()
        for new_c in new_centroids:
            matched = False
            for obj_id, existing_c in self.objects.items():
                distance = np.linalg.norm(np.array(existing_c) - np.array(new_c))
                if distance < self.max_distance:
                    self.objects[obj_id] = new_c
                    updated_ids.add(obj_id)
                    matched = True
                    break
            
            if not matched:
                self.register(new_c)

        return len(self.objects)  # Return the number of unique vehicles


def calculate_signal_time(vehicle_count):
    """ Calculate the signal time based on the vehicle count. """
    if 20 < vehicle_count <= 30:
        return 25
    elif vehicle_count > 30:
        return 20
    elif 10 < vehicle_count <= 20:
        return 15
    elif 6 < vehicle_count <= 10:
        return 10
    else:
        return 5


def Deploy_10(request):
    """ Django View to Start Detection Only When Button is Clicked """
    if request.method == 'POST':
        model = YOLO('APP/vehical1.pt')
        tracker = CentroidTracker(max_distance=50)

        cap = cv2.VideoCapture(0)  # Open Camera (0 for default webcam)
        monitoring_time = 30  # Initial monitoring time

        unique_vehicle_count = 0

        start_time = time.time()
        while time.time() - start_time < monitoring_time:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (640, 480))
            results = model(frame, stream=True)

            centroids = []
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    confidence = math.ceil(box.conf[0] * 100)

                    if confidence > 30:
                        centroid = ((x1 + x2) // 2, (y1 + y2) // 2)
                        centroids.append(centroid)

                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f'{confidence}%', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            unique_vehicle_count = tracker.update(centroids)

            # Display detection window (closes when user presses 'q')
            cv2.imshow("Vehicle Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        # Calculate the recommended signal time
        signal_time = calculate_signal_time(unique_vehicle_count)

        # Render the results on the page
        return render(request, '10_Deploy.html', {
            'vehicle_count': unique_vehicle_count,
            'signal_time': signal_time
        })

    # If GET request, just load the page without starting detection
    return render(request, '10_Deploy.html')


def Deploy_11(request):
    if request.method == 'POST':
        from ultralytics import YOLO
        import cvzone
        import cv2
        import math
        import pyttsx3
        from APP.models import DetectedAccident
        import openpyxl

        # Start webcam
        cap = cv2.VideoCapture(0)

        model = YOLO('APP/best.pt')

        # List of class names (make sure it matches your model training)
        classnames = ['Accident']

        # Create a new Excel workbook
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.append(['Frame Number', 'Class', 'Confidence', 'Coordinates'])

        frame_number = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (640, 480))
            result = model(frame, stream=True)

            for info in result:
                boxes = info.boxes
                for box in boxes:
                    confidence = box.conf[0]
                    confidence = math.ceil(confidence * 100)
                    class_index = int(box.cls[0])

                    if confidence > 50:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])

                        if 0 <= class_index < len(classnames):
                            class_name = classnames[class_index]
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 5)
                            cvzone.putTextRect(frame, f'{class_name} {confidence}%',
                                               [x1 + 8, y1 + 100], scale=1.5, thickness=2)

                            if class_name == "Accident":
                                # Text-to-speech alert
                                alert_msg = "Accident " * 10
                                engine = pyttsx3.init()
                                engine.say(alert_msg)
                                engine.runAndWait()

                                frame_number += 1
                                coordinates = f'({x1}, {y1}) to ({x2}, {y2})'

                                # Save to database
                                detected_accident = DetectedAccident(
                                    frame_number=frame_number,
                                    class_name=class_name,
                                    confidence=confidence,
                                    coordinates=coordinates
                                )
                                detected_accident.save()

                                print(f"Detected: {class_name}, Confidence: {confidence}, Coordinates: {coordinates}")
                            else:
                                print("No Accident")
                        else:
                            print(f"Unknown class index: {class_index}")
                    else:
                        print("Low confidence detection")

            cv2.imshow('frame', frame)

            # Exit if ESC is pressed
            if cv2.waitKey(1) == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

        # Fetch saved accident records
        saved_accidents = DetectedAccident.objects.all()
        return render(request, '11_Deploy.html', {"saved_accidents": saved_accidents})

    else:
        return render(request, '11_Deploy.html')

        

def Logout(request):
    logout(request)
    return redirect('Login_3')
