import threading
import queue
import time
import random
import numpy as np

class PacketDeliverySystem:
    def __init__(self, packet_queue, process_rate_mbps, name):
        self.name = name
        self.packet_queue = packet_queue
        self.process_rate_mbps = process_rate_mbps        
        self.stop_event = threading.Event()
        self.latency_list = []
        self.print_interval = 5
        self.total_bytes_processed = 0
        self.total_latency = 0
        self.packet_count = 0        
        self.start_time = time.time()


    def deliver_packets(self):
        while not self.stop_event.is_set():
            try:
                packet, queuing_start_time = self.packet_queue.get(timeout=1)  # Get packet from the queue with a timeout
                # Calculate queuing time
                queuing_time = (time.time() - queuing_start_time) * 1000                
                self.total_bytes_processed_wifi += len(packet)                                
                processing_time = len(packet) * 8 / (self.process_rate_mbps * 10**6) * 1000 # Calculate time to process packet
                # Calculate total delay time
                total_delay_time = queuing_time + processing_time
                self.latency_list.append(total_delay_time)
                self.total_bytes_processed += len(packet)
                self.total_latency += total_delay_time
                self.packet_count += 1
                # Sleep for processing time
                time.sleep(processing_time/1000)
                # print(f"Packet delivered via {channel} with a total delay of {total_delay_time} ms.")
            except queue.Empty:
                pass  # Queue is empty, continue looping

            if time.time() - self.start_time >= self.print_interval:
                self.print_stats()
                self.start_time = time.time()
                self.total_bytes_processed = 0
                self.total_latency = 0
                self.packet_count = 0    
                self.latency_list = []            

    def stop(self):
        self.stop_event.set()    

    def get_latency_list(self):
        return self.latency_list
    
    def print_stats(self):
        elapsed_time = time.time() - self.start_time
        bitrate = (self.total_bytes_processed * 8) / (elapsed_time * 10**6)  # Calculate bitrate in Mbps
        average_latency = self.total_latency / self.packet_count if self.packet_count > 0 else 0
        print("Channel :", self.name, ", bitrate: ", bitrate, ", latency: ",average_latency, " ms")