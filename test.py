import gym
import random
import time
import threading

class NetworkEnvironment(gym.Env):
    def __init__(self):
        self.queue_size = 100
        self.packet_generation_rates = {
            'TF1': 20,  # mbps
            'TF2': 50,  # mbps
            'TF3': 200  # mbps
        }
        self.packet_sizes = {
            'TF1': 200,  # bytes
            'TF2': 500,  # bytes
            'TF3': 1024  # bytes
        }
        self.processing_rates = {
            '5G': 400,  # mbps
            'WiFi': 100  # mbps
        }
        self.revenue_rates = {
            'TF1': 1,  # $/mb
            'TF2': 2,  # $/mb
            'TF3': 3  # $/mb
        }
        self.queue = []
        self.packet_counts = {'TF1': 0, 'TF2': 0, 'TF3': 0}
        self.drop_rates = {'TF1': 0, 'TF2': 0, 'TF3': 0}
        self.revenue = 0
        self.state = {
            'packet_generation_rates': self.packet_generation_rates,
            'queue_length': 0,
            'drop_rates': self.drop_rates,
            'revenue': self.revenue
        }
    def generate_packets(self):
        # Generate packets according to the packet generation rates
        for packet_type in ['TF1', 'TF2', 'TF3']:
            rate = self.packet_generation_rates[packet_type]
            num_packets = random.expovariate(rate)
            for _ in range(num_packets):
                packet = {'type': packet_type, 'size': self.packet_sizes[packet_type]}
                self.queue.append(packet)
                self.packet_counts[packet_type] += 1

    def process_packets(self, action):
        # Process packets based on the action
        throughput = 0
        queue_length = len(self.queue)
        drops = 0
        for packet in self.queue:
            if random.random() < action:
                # Process packet through 5G network
                processing_time = packet['size'] / self.processing_rates['5G']
                self.revenue += self.revenue_rates[packet['type']] * packet['size']
            else:
                # Process packet through WiFi network
                processing_time = packet['size'] / self.processing_rates['WiFi']
            throughput += packet['size'] / processing_time
            if queue_length > self.queue_size:
                drops += 1
                self.drop_rates[packet['type']] += 1
        self.drop_rates = {k: v / queue_length for k, v in self.drop_rates.items()}
        return throughput, queue_length, drops

    def step(self, action):
        # Create a thread to generate packets
        packet_thread = threading.Thread(target=self.generate_packets)
        packet_thread.start()

        # Create a thread to process packets
        process_thread = threading.Thread(target=self.process_packets, args=(action,))
        process_thread.start()

        # Wait for the threads to finish
        packet_thread.join()
        process_thread.join()

        # Calculate the reward
        reward = self.revenue - self.drop_rates['TF1'] * 10 - self.drop_rates['TF2'] * 20 - self.drop_rates['TF3'] * 30

        # Update the state
        self.state['packet_generation_rates'] = self.packet_generation_rates
        self.state['queue_length'] = len(self.queue)
        self.state['drop_rates'] = self.drop_rates
        self.state['revenue'] = self.revenue

        # Log statistics every 5s
        if time.time() % 5 == 0:
            print(f'Packet generation rates: {self.packet_generation_rates}')
            print(f'Queue length: {len(self.queue)}')
            print(f'Drop rates: {self.drop_rates}')
            print(f'Revenue: {self.revenue}')
        return self.state, reward, False, {}

    def reset(self):
        self.queue = []
        self.packet_counts = {'TF1': 0, 'TF2': 0, 'TF3': 0}
        self.drop_rates = {'TF1': 0, 'TF2': 0, 'TF3': 0}
        self.revenue = 0
        self.state = {
            'packet_generation_rates': self.packet_generation_rates,
            'queue_length': 0,
            'drop_rates': self.drop_rates,
            'revenue': self.revenue
        }
        return self.state

if __name__ == '__main__':
    env = NetworkEnvironment()
    state = env.reset()
    for _ in range(10):
        action = random.random()  # Random action
        state, reward, done, _ = env.step(action)
        print(f'State: {state}, Reward: {reward}, Done: {done}')