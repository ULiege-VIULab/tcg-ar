from inference.io_module import detect_camera, webcam_read, read_buffer_and_show
from inference.user_interface import Main_window
from inference.style import STYLESHEET
from core.shared_memory import Shared_frame_buffer
from multiprocessing import Process, Lock
from PySide6 import QtWidgets, QtGui
import signal
from core.config import *

def main():
    #Terminate program handler
    def termination_handler(signum, frame):
        print("Closing application")
        mainwindow.centralWidget().closeEvent(QtGui.QCloseEvent())
        print("Application closed")
        exit(0)
    signal.signal(signal.SIGTERM, termination_handler)

    app = QtWidgets.QApplication([])
    app.setStyleSheet(STYLESHEET)
    screen_size = app.primaryScreen().size()

    mainwindow = Main_window(screen_size)
    mainwindow.show()
    app.exec()

    signal.raise_signal(signal.SIGTERM)

def main_without_gui():
    # Terminate program handler
    def interrupt_handler(signum, frame):
        print("")
        print("Ending processes")
        for id_proc in range(detected_camera.get_nb_cam_available()):
            input_processses[id_proc].terminate()
            output_processes[id_proc].terminate()
        print("Ended processes")
        exit(0)
    signal.signal(signal.SIGINT, interrupt_handler)

    detected_camera = detect_camera()

    locks = list()
    shared_frame_buffers = list()
    input_processses = list()
    output_processes = list()

    for id_cam in detected_camera.get_cam_ids():
        locks.append(Lock())
        shared_frame_buffer = Shared_frame_buffer(id = id_cam, 
                                                  io = 0,
                                                  length = BUFFER_LENGTH, 
                                                  resolution = (HEIGHT, WIDTH, CHANNELS), 
                                                  existing_shm = False)
        shared_frame_buffers.append(shared_frame_buffer)

    for id_proc, id_cam in enumerate(detected_camera.get_cam_ids()):
        input_processses.append(Process(target = webcam_read, args = (id_cam, locks[id_proc])))
        output_processes.append(Process(target = read_buffer_and_show, args = (id_cam, locks[id_proc])))

    for id_proc in range(detected_camera.get_nb_cam_available()):
        input_processses[id_proc].start()
        output_processes[id_proc].start()

    for id_proc in range(detected_camera.get_nb_cam_available()):
        input_processses[id_proc].join()
        output_processes[id_proc].join()

if __name__ == '__main__':
    main()