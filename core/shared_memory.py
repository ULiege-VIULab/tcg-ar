"""
Circular frame buffer over ``multiprocessing.shared_memory``.

Named deterministically by ``(camera id, io)`` where ``io=0`` is the raw camera
input and ``io=1`` is the augmented / identification output feed.  ``BUFFER_LENGTH``
is 1 everywhere: the design intentionally drops stale frames to stay real-time.

The buffer methods are NOT multiprocessing-safe on their own -- every access must
hold the per-buffer ``Lock``.
"""

import copy

import numpy as np
from multiprocessing import shared_memory


class Shared_frame_buffer:
    ID_INDEX = 0
    LENGTH_INDEX = 1
    HEIGHT_INDEX = 2
    WIDTH_INDEX = 3
    CHANNELS_INDEX = 4
    HEAD_INDEX = 5
    TAIL_INDEX = 6
    NB_ELEMENTS_INDEX = 7

    def __init__(self, id, io, length, resolution, existing_shm):
        height, width, channels = resolution[0], resolution[1], resolution[2]

        if existing_shm:
            self.shared_list_variable = shared_memory.ShareableList(
                name=self.__create_variable_list_name_from_ID(id, io))
            self.shared_buffer = shared_memory.SharedMemory(
                name=self.__create_buffer_name_from_ID(id, io))
        else:
            self.shared_list_variable = shared_memory.ShareableList(
                [id, length, height, width, channels, 0, 0, 0],
                name=self.__create_variable_list_name_from_ID(id, io))
            self.shared_buffer = shared_memory.SharedMemory(
                name=self.__create_buffer_name_from_ID(id, io),
                create=True, size=length * height * width * channels)

        self.shared_frame_buffer = np.ndarray(
            shape=(length, height, width, channels), dtype=np.uint8, buffer=self.shared_buffer.buf)
        self.copy = existing_shm

    def __del__(self):
        self.delete()

    def size(self):
        return self.shared_list_variable[self.LENGTH_INDEX]

    def resolution(self):
        return (self.shared_list_variable[self.HEIGHT_INDEX],
                self.shared_list_variable[self.WIDTH_INDEX],
                self.shared_list_variable[self.CHANNELS_INDEX])

    def empty(self):
        return self.shared_list_variable[self.NB_ELEMENTS_INDEX] == 0

    def full(self):
        return self.shared_list_variable[self.NB_ELEMENTS_INDEX] == self.shared_list_variable[self.LENGTH_INDEX]

    def put(self, frame):
        if self.full():
            raise BufferError
        tail = self.shared_list_variable[self.TAIL_INDEX]
        # Assigning into the shared-memory ndarray already performs a full copy of the
        # frame into shared memory, so an extra copy.deepcopy(frame) here just wasted a
        # ~6 MB allocation + copy on every put (per stream, per frame). Assign directly.
        self.shared_frame_buffer[tail] = frame
        self.shared_list_variable[self.NB_ELEMENTS_INDEX] += 1
        self.shared_list_variable[self.TAIL_INDEX] = (tail + 1) % self.shared_list_variable[self.LENGTH_INDEX]

    def get_index(self, index):
        if self.empty():
            raise BufferError
        if index >= self.shared_list_variable[self.NB_ELEMENTS_INDEX]:
            raise IndexError
        circular_index = (self.shared_list_variable[self.HEAD_INDEX] + index) % self.shared_list_variable[self.LENGTH_INDEX]
        return self.shared_frame_buffer[circular_index]

    def get(self):
        if self.empty():
            raise BufferError
        head = self.shared_list_variable[self.HEAD_INDEX]
        return self.shared_frame_buffer[head]

    def pop(self):
        if self.empty():
            raise BufferError
        previous_head = self.shared_list_variable[self.HEAD_INDEX]
        self.shared_list_variable[self.HEAD_INDEX] = (previous_head + 1) % self.shared_list_variable[self.LENGTH_INDEX]
        self.shared_list_variable[self.NB_ELEMENTS_INDEX] -= 1
        return copy.deepcopy(self.shared_frame_buffer[previous_head])

    def delete(self):
        if self.copy:
            self.shared_buffer.close()
            self.shared_list_variable.shm.close()
        else:
            self.shared_buffer.close()
            self.shared_buffer.unlink()
            self.shared_list_variable.shm.close()
            self.shared_list_variable.shm.unlink()

    def __create_buffer_name_from_ID(self, id, io):
        return "shared_buffer_" + str(id) + str(io)

    def __create_variable_list_name_from_ID(self, id, io):
        return "shared_variable_" + str(id) + str(io)
