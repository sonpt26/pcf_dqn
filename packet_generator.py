import threading
import queue
import time
import random
import numpy as np

class PacketGenerator:
    def __init__(self, packet_queue, packet_size_bytes, packet_rate_distribution, packet_rate_params, drop_threshold):
        self.packet_queue = packet_queue
        self.packet_size_bytes = packet_size_bytes
        self.packet_rate_distribution = packet_rate_distribution
        self.packet_rate_params = packet_rate_params
        self.drop_threshold = drop_threshold
        self.stop_event = threading.Event()        

    def generate_packets(self):
        while not self.stop_event.is_set():
            # Generate inter-arrival time based on the chosen distribution
            inter_arrival_time = getattr(np.random, self.packet_rate_distribution)(*self.packet_rate_params)
            # print(inter_arrival_time)
            # # Wait for inter-arrival time
            time.sleep(inter_arrival_time)
            if self.packet_queue.qsize() < self.drop_threshold:
                # Generate packet
                packet = bytearray(self.packet_size_bytes)
                # Record the time when the packet enters the queue
                queuing_start_time = time.time()
                # Put packet into the queue along with its queuing start time
                try:
                    self.packet_queue.put((packet, queuing_start_time))
                except queue.Full:
                    print("Packet queue is full, dropping packet.")
            else:
                print("Packet queue is full, dropping packet.")

    def stop(self):
        self.stop_event.set()