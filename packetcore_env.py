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
        self.latency_list = []

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

    def get_latency_list(self):
        return self.latency_list

class PacketDeliverySystem:
    def __init__(self, packet_queue, wifi_packet_rate_mbps, _5g_packet_rate_mbps):
        self.packet_queue = packet_queue
        self.wifi_packet_rate_mbps = wifi_packet_rate_mbps
        self._5g_packet_rate_mbps = _5g_packet_rate_mbps
        self.stop_event = threading.Event()
        self.latency_list = []
        self.print_interval = 5
        self.total_bytes_processed = 0
        self.total_latency = 0
        self.packet_count = 0
        self.total_bytes_processed_wifi = 0
        self.total_bytes_processed_5g = 0
        self.start_time = time.time()


    def deliver_packets(self):
        while not self.stop_event.is_set():
            try:
                packet, queuing_start_time = self.packet_queue.get(timeout=1)  # Get packet from the queue with a timeout
                # Calculate queuing time
                queuing_time = (time.time() - queuing_start_time) * 1000
                # Select channel (Wi-Fi or 5G) randomly
                channel = random.choice(["Wi-Fi", "5G"])
                if channel == "Wi-Fi":
                    packet_rate_mbps = self.wifi_packet_rate_mbps
                    self.total_bytes_processed_wifi += len(packet)
                else:
                    packet_rate_mbps = self._5g_packet_rate_mbps
                    self.total_bytes_processed_5g += len(packet)
                # Process packet (e.g., send it over the selected channel)
                # Here we'll just simulate processing time
                processing_time = len(packet) * 8 / (packet_rate_mbps * 10**6) * 1000 # Calculate time to process packet
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
                self.total_bytes_processed_wifi = 0
                self.total_bytes_processed_5g = 0

    def stop(self):
        self.stop_event.set()    

    def get_latency_list(self):
        return self.latency_list
    
    def print_stats(self):
        elapsed_time = time.time() - self.start_time
        bitrate = (self.total_bytes_processed * 8) / (elapsed_time * 10**6)  # Calculate bitrate in Mbps
        average_latency = self.total_latency / self.packet_count if self.packet_count > 0 else 0        
        bitrate_wifi = (self.total_bytes_processed_wifi * 8) / (elapsed_time * 10**6)  # Calculate bitrate in Mbps
        bitrate_5g = (self.total_bytes_processed_5g * 8) / (elapsed_time * 10**6)  # Calculate bitrate in Mbps
        print("Bitrate:", bitrate, "Mbps")
        print("Bitrate (Wi-Fi):", bitrate_wifi, "Mbps")
        print("Bitrate (5G):", bitrate_5g, "Mbps")
        print("Average Latency:", average_latency, "ms")
        print("========================================")

# Example usage
packet_queue = queue.Queue(maxsize=10000)  # Adjust queue size as needed
packet_size_bytes = 1024  # Adjust packet size as needed
wifi_rate_mbps = 100  # Rate of 10 Mbps
nr_rate_mbps = 400  # Rate of 10 Mbps
drop_threshold = 900  # Adjust drop threshold as needed
packet_rate_distribution = "exponential"  # Exponential distribution for inter-arrival times
packet_rate_params = (0.0001,)  # Parameter for exponential distribution (mean inter-arrival time)

generator = PacketGenerator(packet_queue, packet_size_bytes, packet_rate_distribution, packet_rate_params, drop_threshold)
delivery_system = PacketDeliverySystem(packet_queue, nr_rate_mbps, wifi_rate_mbps)

generator_thread = threading.Thread(target=generator.generate_packets)
delivery_thread = threading.Thread(target=delivery_system.deliver_packets)

generator_thread.start()
delivery_thread.start()

# Let the generator and delivery system run for some time
time.sleep(60)

# Stop the generator and delivery system
generator.stop()
delivery_system.stop()

generator_thread.join()
delivery_thread.join()

# Calculate average latency of all packets
latency_list = generator.get_latency_list() + delivery_system.get_latency_list()
average_latency = sum(latency_list) / len(latency_list)
print("Average latency of all packets:", average_latency, "seconds")
