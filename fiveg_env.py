import numpy as np

class CoreNetworkEnvironment:
    def __init__(self, num_channels=2, num_flows=3, max_queue_size=100, transmission_rates=[10, 8]):
        self.num_channels = num_channels
        self.num_flows = num_flows
        self.max_queue_size = max_queue_size
        self.transmission_rates = transmission_rates
        self.channel_queues = np.zeros(num_channels)
        self.flow_emission_rates = np.zeros(num_flows)
        self.latencies = []  # List to store latencies of transmitted packages
        self.drop_percentage = 0
        self.total_packets = 0
        self.flow_prices = np.random.randint(1, 5, num_flows)  # Random prices for each flow

        
    def reset(self):
        self.channel_queues = np.zeros(self.num_channels)
        self.flow_emission_rates = np.random.randint(1, 10, self.num_flows)
        self.latencies = []
        self.drop_percentage = 0
        self.total_packets = 0
        return self._get_state()
    
    def step(self, action):
        # Simulate data transmission based on action
        for i in range(self.num_flows):
            channel = np.random.choice([0, 1], p=[action, 1-action])
            if self.channel_queues[channel] < self.max_queue_size:
                transmission_time = self.flow_emission_rates[i] / self.transmission_rates[channel]
                queue_time = self.channel_queues[channel] / self.transmission_rates[channel]
                latency = transmission_time + queue_time
                self.latencies.append(latency)
                
                self.channel_queues[channel] += self.flow_emission_rates[i]
                self.total_packets += self.flow_emission_rates[i]
                
                # Calculate revenue if transmitted via 5G
                if channel == 0:
                    self.total_revenue += self.flow_prices[i] * self.flow_emission_rates[i]
            else:
                self.drop_percentage += 1
                
        # Calculate latency
        avg_latency = np.mean(self.latencies)
        
        # Calculate reward
        reward = -avg_latency - self.drop_percentage - self.total_revenue
        
        return self._get_state(), reward
    
    def _get_state(self):
        state = np.concatenate((self.channel_queues, self.flow_emission_rates))
        state = np.append(state, [self.drop_percentage])
        return state
