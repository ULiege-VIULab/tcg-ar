import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2
from core.config import *
import platform
from core.shared_memory import Shared_frame_buffer
import signal
import subprocess
import sys
import time
import numpy as np
from imageio_ffmpeg import get_ffmpeg_exe

class Detected_camera():
    def __init__(self, cam_id, number_open_cam):
        self.nb_cam = len(cam_id)
        self.cam_id = cam_id
        self.number_open_cam = number_open_cam

    def get_nb_cam_available(self):
        return self.nb_cam

    def get_nb_open_cam(self):
        return self.number_open_cam
    
    """
    Give the id of the an available camera.
    Input:  -cam_number: The number of the camera we want to retrieve the ID. Starting from 0 to the number of available camera - 1
    Output: The id as integer of the wanted camera. -1 if bad index. 
    """
    def get_cam_id(self, cam_number):
        if cam_number < 0 or cam_number >= self.nb_cam:
            return -1
        else:
            return self.cam_id[cam_number]

    """
    Give the index of the an available camera.
    Input:  -cam_id: The ID of the camera we want to retrieve the number.
    Output: The index as integer of the wanted camera. -1 if bad ID. 
    """
    def get_cam_number(self, cam_id):
        try:
            return self.cam_id.index(cam_id)
        except ValueError:
            return -1

    """
    Give the ids of available cameras.
    Input:  None
    Output: The ids as a list of integer of the available camera.
    """
    def get_cam_ids(self):
        return self.cam_id

    """
    Remove a camera from the list of detected camera.
    Input:  -cam_number: The number of the camera we want to remove. Starting from 0 to the number of available camera - 1
    Output: None or -1 if bad index.
    """
    def remove_cam(self, cam_number):
        if cam_number < 0 or cam_number >= self.nb_cam:
            return -1
        else:
            self.cam_id.pop(cam_number)
            self.nb_cam -= 1

"""
Detect the number of camera available to use.
Input: None.
Output: -camera_available: an Detected_camera object that give access to the number and id of camera available to use.
"""
def detect_camera():
    list_id_camera_available = list()
    id_cam = 0

    while True:
        #Microsoft Media Foundation backend API is used here over DirectShow due to the recommendation of Windows.
        #https://learn.microsoft.com/en-us/windows/win32/directshow/about-video-capture-in-directshow
        #On linux libV4l is used, for other OS the backend is detected automaticaly.
        if platform.system() == "Windows":
            cam = cv2.VideoCapture(id_cam, cv2.CAP_MSMF)
        elif platform.system() == "Linux":
            cam = cv2.VideoCapture(id_cam, cv2.CAP_V4L2)
        else:
            cam = cv2.VideoCapture(id_cam)

        if not cam.isOpened():
            break
        else:
            if cam.grab():
                list_id_camera_available.append(id_cam)
            id_cam += 1
        
        cam.release()

    return Detected_camera(list_id_camera_available, id_cam)

"""
Start to use the given camera. Configure it to obtain 1080p image and store it in a Shared_frame_buffer where the share memory should be initialise before calling this function.
Input:  -camera_number: an integer refering the camera to use.
        -lock: a Lock object to insure process safe buffer.
Output: None.
"""
def webcam_read(camera_number, lock):
    #Terminate program handler
    def termination_handler(signum, frame):
        print("Stop input process %d" % camera_number)
        try:
            webcam_feed.release()
            lock.release()
        except ValueError:
            exit(0)
        exit(0)
    signal.signal(signal.SIGTERM, termination_handler)

    buffer = Shared_frame_buffer(id = camera_number, 
                                 io = 0,
                                 length = BUFFER_LENGTH, 
                                 resolution = (HEIGHT,WIDTH,CHANNELS), 
                                 existing_shm = True)

    #Microsoft Media Foundation backend API is used here over DirectShow due to the recommendation of Windows.
    #https://learn.microsoft.com/en-us/windows/win32/directshow/about-video-capture-in-directshow
    #On linux libV4l is used, for other OS the backend is detected automaticaly.
    if platform.system() == "Windows":
        webcam_feed = cv2.VideoCapture(camera_number, cv2.CAP_MSMF)
    elif platform.system() == "Linux":
        webcam_feed = cv2.VideoCapture(camera_number, cv2.CAP_V4L2)
    else:
        webcam_feed = cv2.VideoCapture(camera_number)

    if not webcam_feed.isOpened():
        print("The camera %d can not be initialise." % camera_number, file = sys.stderr)
        webcam_feed.release()
        return()
    webcam_feed.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    webcam_feed.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    webcam_feed.set(cv2.CAP_PROP_FPS, FRAMERATE)

    ret, frame = webcam_feed.read()

    while ret:
        #change resolution of image to 1080p if not native from camera
        if frame.shape != (HEIGHT, WIDTH, CHANNELS):
            frame = cv2.resize(frame, (WIDTH, HEIGHT), interpolation = cv2.INTER_CUBIC)

        lock.acquire(True)
        try:
            buffer.put(frame)
        except BufferError:
            pass
            #print("input buffer %d full" % camera_number, file = sys.stderr)
        lock.release()

        ret, frame = webcam_feed.read()

    print("The camera %d stop working." % camera_number, file = sys.stderr)
    webcam_feed.release()

"""
Start a given camera. Configure it to obtain 1080p image and store the define number of frames in a Shared_frame_buffer where the share memory should be initialise before calling this function.
Input:  -camera_number: an integer refering the camera to use.
        -nb_frames: an integer refering the number of frame to read from the camera and store in the buffer.
        -lock: a Lock object to insure process safe buffer.
Output: None.
"""
def webcam_read_nb_frames(camera_number, nb_frames, lock):
    #Terminate program handler
    def termination_handler(signum, frame):
        print("Stop input process %d" % camera_number)
        try:
            webcam_feed.release()
            lock.release()
        except:
            exit(0)
        exit(0)  
    signal.signal(signal.SIGTERM, termination_handler)

    buffer = Shared_frame_buffer(id = camera_number, 
                                 io = 0,
                                 length = BUFFER_LENGTH, 
                                 resolution = (HEIGHT,WIDTH,CHANNELS), 
                                 existing_shm = True)

    #Microsoft Media Foundation backend API is used here over DirectShow due to the recommendation of Windows.
    #https://learn.microsoft.com/en-us/windows/win32/directshow/about-video-capture-in-directshow
    #On linux libV4l is used, for other OS the backend is detected automaticaly.
    if platform.system() == "Windows":
        webcam_feed = cv2.VideoCapture(camera_number, cv2.CAP_MSMF)
    elif platform.system() == "Linux":
        webcam_feed = cv2.VideoCapture(camera_number, cv2.CAP_V4L2)
    else:
        webcam_feed = cv2.VideoCapture(camera_number)

    if not webcam_feed.isOpened():
        print("The camera %d can not be initialise." % camera_number, file = sys.stderr)
        webcam_feed.release()
        return()
    webcam_feed.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    webcam_feed.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    webcam_feed.set(cv2.CAP_PROP_FPS, FRAMERATE)

    for _ in range(nb_frames):
        ret, frame = webcam_feed.read()
        if not(ret):
            print("The camera %d stop working." % camera_number, file = sys.stderr)
            break

        #change resolution of image to 1080p if not native from camera
        if frame.shape != (HEIGHT, WIDTH, CHANNELS):
            frame = cv2.resize(frame, (WIDTH, HEIGHT), interpolation = cv2.INTER_CUBIC)

        lock.acquire(True)
        try:
            buffer.put(frame)
        except BufferError:
            print("buffer %d full" % camera_number, file = sys.stderr)
        lock.release()

    webcam_feed.release()

"""
Start to read frame from a Shared_frame_buffer define by the camera number where the share memory should be initialise before.
Then show the frames.
Input:  -camera_number: an integer refering the camera to use.
        -lock: a lock object to insure process safe buffer.
Output: None.
"""
def read_buffer_and_show(camera_number, lock):
    #Terminate program handler
    def termination_handler(signum, frame):
        print("Stop output process %d" % camera_number)
        try:
            cv2.destroyAllWindows()
            lock.release()
        except:
            exit(0)
        exit(0)
    signal.signal(signal.SIGTERM, termination_handler)

    buffer = Shared_frame_buffer(id = camera_number, 
                                 io = 0,
                                 length = BUFFER_LENGTH, 
                                 resolution = (HEIGHT,WIDTH,CHANNELS), 
                                 existing_shm = True)

    while True:
        lock.acquire(True)
        try:
            frame = buffer.pop()
            cv2.imshow('webcam_' + str(camera_number), frame)
            cv2.waitKey(1)
        except BufferError:
            print("buffer %d empty" % camera_number, file = sys.stderr)
        lock.release()

"""
Start to read a determine number of frame frame from a Shared_frame_buffer define by the camera number where the share memory should be initialise before.
Then show the frames.
Input:  -camera_number: an integer refering the camera to use.
        -nb_frames: an integer refering the number of frame to read from the buffer.
        -lock: a lock object to insure process safe buffer.
Output: None.
"""
def read_nb_frames_from_buffer_and_show(camera_number, nb_frames, lock):
    #Terminate program handler
    def termination_handler(signum, frame):
        print("Stop output process %d" % camera_number)
        try:
            cv2.destroyAllWindows()
            lock.release()
        except ValueError:
            exit(0)
        exit(0)
    signal.signal(signal.SIGTERM, termination_handler)

    buffer = Shared_frame_buffer(id = camera_number, 
                                 io = 0,
                                 length = BUFFER_LENGTH, 
                                 resolution = (HEIGHT,WIDTH,CHANNELS), 
                                 existing_shm = True)

    for _ in range(nb_frames):
        lock.acquire(True)
        try:
            frame = buffer.pop()
            cv2.imshow('webcam_' + str(camera_number), frame)
            cv2.waitKey(1)
        except BufferError:
            print("buffer %d empty" % camera_number, file = sys.stderr)
        lock.release()

    cv2.destroyAllWindows()

"""
Read and return a number of frame from a buffer.
Input:  -buffer: a Shared_frame_buffer refering a share memory space
        -lock: a lock object to insure process safe buffer.
Output: -frames: a list of numpy array defining the frames.
"""
def read_nb_frames_from_buffer(nb_frame, buffer, lock):
    frames = list()

    lock.acquire(True)
    for _ in range(nb_frame):
        try:
            frame = buffer.pop()
            frames.append(frame)
        except BufferError:
            print("buffer empty", file = sys.stderr)
            break
    lock.release()

    return frames

"""
Read and return a frame from a buffer.
Input:  -buffer: a Shared_frame_buffer refering a share memory space.
        -lock: a lock object to insure process safe buffer.
Output: -frame: a numpy array defining the frame.
                In the case where the buffer is empty return -1.
"""
def read_a_frame_from_buffer(buffer, lock):
    frame = -1

    lock.acquire(True)
    try:
        frame = buffer.pop()
    except BufferError:
        pass
        #print("input buffer empty", file = sys.stderr)
    lock.release()

    return frame

"""
Start to send frame to an RSTP server.
Input:  -camera_number: an integer refering the id of the camera frame to send.
        -lock: a Lock object to insure process safe buffer.
        -wait_event: an Event object to stop the execution of the process.
Output: None.
"""
def send_frame_RSTP(camera_number, lock, wait_event):
    #Terminate program handler
    def termination_handler(signum, frame):
        print("Stop input process %d" % camera_number)
        try:
            sender.stdin.close()
            sender.wait()
            lock.release()
        except (ValueError, OSError):
            exit(0)
        exit(0)
    signal.signal(signal.SIGTERM, termination_handler)

    buffer = Shared_frame_buffer(id = camera_number, 
                                 io = 1,
                                 length = BUFFER_LENGTH, 
                                 resolution = (HEIGHT,WIDTH,CHANNELS), 
                                 existing_shm = True)

    ffmpeg_exe = get_ffmpeg_exe()
    ffmpeg_cmd = [
        ffmpeg_exe, '-y',
        '-f', 'rawvideo', '-pixel_format', 'bgr24',
        '-video_size', f'{WIDTH}x{HEIGHT}',
        '-framerate', str(STREAM_FRAMERATE),
        '-i', 'pipe:0',
        '-c:v', 'libx264', '-preset', 'veryfast', '-tune', 'zerolatency',
        '-g', str(STREAM_FRAMERATE), '-bufsize', '500k',
        '-b:v', '4000k', '-pix_fmt', 'yuv420p',
        '-f', 'rtsp', '-rtsp_transport', 'tcp',
        RTSP_BASE_URL + str(camera_number),
    ]
    popen_kwargs = {'creationflags': subprocess.CREATE_NO_WINDOW} if platform.system() == 'Windows' else {}

    def _start_sender():
        return subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                **popen_kwargs)

    sender = _start_sender()

    frame = np.zeros((WIDTH, HEIGHT, CHANNELS), dtype=np.uint8)
    _frame_interval = 1.0 / STREAM_FRAMERATE

    while True:
        _t0 = time.monotonic()
        wait_event.wait()
        if lock.acquire(False):
            try:
                frame = buffer.pop()
            except BufferError:
                pass
                #print("output buffer %d empty" % camera_number, file = sys.stderr)
            lock.release()
        try:
            sender.stdin.write(frame.tobytes())
        except (BrokenPipeError, OSError):
            # ffmpeg exited (e.g. RTSP server restarted); relaunch and retry
            try:
                sender.wait(timeout=2)
            except subprocess.TimeoutExpired:
                sender.kill()
            sender = _start_sender()
            try:
                sender.stdin.write(frame.tobytes())
            except (BrokenPipeError, OSError):
                pass
        _remaining = _frame_interval - (time.monotonic() - _t0)
        if _remaining > 0:
            time.sleep(_remaining)

#some test
if __name__ == '__main__':
    detected_cam = detect_camera()
    print(detected_cam.get_nb_cam_available())
    print(detected_cam.get_cam_id(0))
    print(detected_cam.get_cam_id(1))
    print(detected_cam.get_cam_id(-5))