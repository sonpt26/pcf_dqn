import queue
import gym
from gym import spaces
import numpy as np
import threading
import time
from atomiclong import AtomicLong
import timeit


class Packet:
    def __init__(self, time, traffic_class) -> None:
        self.start = time
        self.traffic_class = traffic_class
        pass

    def get_start(self):
        return self.start

    def get_traffic_class(self):
        return self.traffic_class


class NetworkEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self):
        super(NetworkEnv, self).__init__()
        # Parameters
        self.scale_factor = 1
        self.generator_setting = {
            "TF1": {"num_thread": 2, "packet_size": 1024, "rate": 10, "price": 1},
            "TF2": {"num_thread": 2, "packet_size": 1024, "rate": 20, "price": 2},
            "TF3": {"num_thread": 1, "packet_size": 10240, "rate": 10, "price": 3},
        }
        self.processor_setting = {
            "NR": {"num_thread": 3, "limit": 200, "rate": 400, "revenue_factor": 1},
            "WF": {"num_thread": 1, "limit": 200, "rate": 100, "revenue_factor": 0},
        }
        self.packet_generation_interval = 5  # seconds
        self.total_simulation_time = 3600  # seconds
        self.traffic_classes = list(self.generator_setting.keys())
        self.choices = list(self.processor_setting.keys())
        self.action_space = spaces.Box(
            low=0, high=1, shape=(len(self.traffic_classes),), dtype=np.float32
        )
        self.queue = {}
        self.queue_limit = {}
        for key, value in self.processor_setting.items():
            self.queue[key] = queue.Queue(value["limit"] * value["num_thread"])

        self.observation_space = spaces.Box(low=0, high=100, dtype=np.float32)
        self.total_revenue = AtomicLong(0)
        self.current_time = 0

    def packet_generator(self, traffic_class, action):
        setting = self.generator_setting[traffic_class]
        packet_size_bytes = setting["packet_size"]
        target_throughput_mbps = setting["rate"]
        accum_counter = self.accumulators[traffic_class]
        time_to_wait = packet_size_bytes * 8 / (target_throughput_mbps * 1e6)
        start_time = time.time()
        proprotion_5G = action[self.traffic_classes.index(traffic_class)]
        weights = [proprotion_5G, 1 - proprotion_5G]
        print("Traffic class", traffic_class, weights)
        while True:
            accum_counter["total"] += 1
            choice = np.random.choice(a=self.choices, p=weights)
            queue = self.queue[choice]
            if queue.full():
                accum_counter["drop"] += 1
            else:
                packet = Packet(time.time_ns(), traffic_class)
                queue.put(packet)
                accum_counter[choice] += 1
            timeit.time.sleep(time_to_wait)
            if time.time() - start_time > self.total_simulation_time:
                break

    def packet_processor(self, tech, rate, queue):
        start = time.time()
        print("Processor", tech, rate, "mbps")
        while True:
            item = queue.get()
            if item is None:
                time.sleep(0.0001)
                continue
            traffic_class = item.get_traffic_class()
            packet_size = self.generator_setting[traffic_class]["packet_size"]
            process_time = packet_size * 1.0 * 8 / (rate * 1e6)
            timeit.time.sleep(process_time)
            latency = time.time_ns() - item.get_start()
            if latency <= 0:
                print("Negative time", latency)
            else:
                self.accumulators[traffic_class]["latency"].append(latency)
                self.stat[tech][traffic_class]["revenue"] += 1
                self.stat[tech][traffic_class]["packet_count"] += 1
            if time.time() - start > self.total_simulation_time:
                break

    def print_stat(self):
        start = time.time()
        start_interval = start
        while True:
            if time.time() - start_interval < 5:
                time.sleep(1)
                continue

            longest = 0
            for key, value in self.accumulators.items():
                log_str = key
                for k, v in value.items():
                    if k == "latency":
                        log_str += (
                            ". latency: " + str(round(np.mean(v) / 1e6, 2)) + " ms"
                        )
                    else:
                        throughput = self.scale_factor * (
                            v.value
                            * self.generator_setting[key]["packet_size"]
                            * 8
                            / (5 * 1e6)
                        )
                        log_str += ". " + k + ": " + str(round(throughput, 2)) + " mbps"
                        v.value = 0
                print(log_str)
                longest = max(longest, len(log_str))

            for key, value in self.stat.items():
                log_str = key
                total_revenue = 0
                total_data = 0
                rev_factor = self.processor_setting[key]["revenue_factor"]
                for k, v in value.items():
                    tf_rev = (
                        self.generator_setting[k]["price"]
                        * self.generator_setting[k]["packet_size"]
                        * v["revenue"].value
                        / 8
                        / 1e6
                    )
                    total_revenue += tf_rev
                    total_data += self.scale_factor * (
                        v["packet_count"].value
                        * self.generator_setting[k]["packet_size"]
                        * 8
                    )
                    throughput = self.scale_factor * (
                        v['packet_count'].value
                        * self.generator_setting[k]["packet_size"]
                        * 8
                        / (5 * 1e6)
                    )
                    log_str += (
                        ". "
                        + k
                        + ": "
                        + str(round(tf_rev, 2))
                        + "$/"
                        + str(round(throughput, 2))
                        + " mbps"
                    )
                log_str += (
                    ". Total: " + str(round(total_revenue * rev_factor, 2)) + "$/"+ str(round(total_data / (5 * 1e6), 2)) + " mbps"
                )                
                print(log_str)

            separator = "=" * longest
            print(separator)
            start_interval = time.time()
            if time.time() - start > self.total_simulation_time:
                break

    def step(self, action):
        self.generators = {}
        self.accumulators = {}
        self.stat = {}
        for key, value in self.processor_setting.items():
            my_queue = self.queue[key]
            self.stat[key] = {}
            for i in range(value["num_thread"]):
                processor = threading.Thread(
                    target=self.packet_processor,
                    args=(
                        key,
                        value["rate"],
                        my_queue,
                    ),
                )
                processor.start()
            for tf in self.generator_setting.keys():
                self.stat[key][tf] = {
                    "revenue": AtomicLong(0),
                    "packet_count": AtomicLong(0),
                }

        for key, value in self.generator_setting.items():
            self.accumulators[key] = {}
            self.accumulators[key]["total"] = AtomicLong(0)
            self.accumulators[key]["drop"] = AtomicLong(0)
            self.accumulators[key]["latency"] = []

            for val in self.choices:
                self.accumulators[key][val] = AtomicLong(0)

            for i in range(value["num_thread"]):
                packet_generator_thread = threading.Thread(
                    target=self.packet_generator,
                    args=(
                        key,
                        action,
                    ),
                )
                packet_generator_thread.start()

        log_thread = threading.Thread(target=self.print_stat)
        log_thread.start()
        return [], 0, False, {}

    def reset(self):
        pass

    def render(self, mode="human"):
        pass

    def close(self):
        pass


env = NetworkEnv()
observation = env.reset()
action = env.action_space.sample()
# action = [1,1,1,1]
# action = [1, 1, 1]
observation, reward, done, _ = env.step(action)
