import time
import random
import threading
from atomiclong import AtomicLong
# Function to generate a packet of random size (in bytes)
class Generator:
    def __init__(self, total_bits_sent, mbps):
        self.total_bits_sent = total_bits_sent
        self.mbps = mbps
        self.count = 0

    def generate_packet(self):
        # Assuming packet size between 64 bytes (minimum Ethernet frame size) and 1500 bytes (maximum Ethernet frame size)
        return bytearray(random.randint(128, 256))

    # Function to generate packets at a specified rate (in Mbps)
    def generate_packets(self):        
        while True:
            # Generate a packet
            packet = self.generate_packet()
            time_to_sleep=(8 * len(packet)) / (self.mbps * 10**6)
            # if self.count == 1000:
            #     print("Sleep time ", time_to_sleep*1000, " ms")
            if time_to_sleep > 0:
                time.sleep(time_to_sleep)
                # Update total bits sent
                self.total_bits_sent += 8 * len(packet)
                self.count += 1
            else:
                print("time < 0")
                pass

total_bits_sent = AtomicLong(0)
for i in range(10):
    gen = Generator(total_bits_sent, 100)
    generator_thread = threading.Thread(target=gen.generate_packets)    
    generator_thread.start()

while True:
    time.sleep(5)
    actual_rate_mbps = total_bits_sent.value / (10**6 * 5)
    print("Actual transmission rate: {:.2f} Mbps".format(actual_rate_mbps))
    total_bits_sent.value = 0