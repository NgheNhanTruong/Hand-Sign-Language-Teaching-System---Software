import speech_recognition
robot_ear = speech_recognition.Recognizer()
with speech_recognition.Microphone() as mic:
    #audio = robot_ear.adjust_for_ambient_noise(mic,duration=4)
    print("Robot: Tôi đang lắng nghe bạn")
    audio = robot_ear.record(mic,duration=4)
try:
    you = robot_ear.recognize_google(audio,language="vi")
except: 
    you ="nothing"
print(you)
