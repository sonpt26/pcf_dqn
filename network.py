import queue
import gym
from gym import spaces
import numpy as np
import threading
import time
from atomiclong import AtomicLong
import timeit
from queue import Empty


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
        self.queue_max_utilization = 0.1
        self.scale_factor = 1
        self.reward_factor = {"qos": 0.5, "revenue": 0.5}
        self.generator_setting = {
            "TF1": {
                "num_thread": 2,
                "packet_size": 512,
                "rate": 5,
                "price": 10,
                "qos_latency_ms": 4,
            },
            "TF2": {
                "num_thread": 2,
                "packet_size": 1024,
                "rate": 6,
                "price": 10,
                "qos_latency_ms": 10,
            },
            "TF3": {
                "num_thread": 5,
                "packet_size": 1500,
                "rate": 15,
                "price": 30,
                "qos_latency_ms": 20,
            },
        }
        self.processor_setting = {
            "NR": {"num_thread": 2, "limit": 500, "rate": 200, "revenue_factor": 0.9},
            "WF": {"num_thread": 2, "limit": 500, "rate": 100, "revenue_factor": 0.1},
        }
        self.timeout_processor = 0.5
        self.total_simulation_time = 10  # seconds
        self.stat_interval = 2
        self.traffic_classes = list(self.generator_setting.keys())
        self.choices = list(self.processor_setting.keys())
        self.action_space = spaces.Box(
            low=0, high=1, shape=(len(self.traffic_classes),), dtype=np.float32
        )
        self.is_drop = False
        self.init_queue()
        state_row = len(self.generator_setting)
        # [latency, tech[i]_throughput, tech[i]_queue]
        state_col = len(self.processor_setting) * 2 + 1
        self.state_shape = (state_row, state_col)
        self.observation_space = spaces.Box(
            low=0, high=1, dtype=np.float32, shape=self.state_shape
        )
        self.sigmoid_state = True

    def init_queue(self):
        self.queue = {}
        for key, value in self.processor_setting.items():
            if self.is_drop:
                self.queue[key] = queue.Queue(value["limit"] * value["num_thread"])
            else:
                self.queue[key] = queue.Queue()

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
                self.stat[choice][traffic_class]["loss"] += 1
            else:
                packet = Packet(time.time_ns(), traffic_class)
                queue.put(packet)
                accum_counter[choice] += 1
            timeit.time.sleep(time_to_wait)
            if time.time() - start_time > self.total_simulation_time:
                print("Generator finish", traffic_class)
                self.generators_finish += 1
                break

    def spinwait_nano(delay):
        target = perf_counter_ns() + delay * 1000
        while perf_counter_ns() < target:
            pass

    def packet_processor(self, tech, rate, queue):
        start = time.time()
        print("Processor", tech, rate, "mbps")
        while True:
            try:
                item = queue.get(timeout=self.timeout_processor)
                if item is None:
                    time.sleep(0.0001)
                    continue
                else:
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
                    queue.task_done()
            except Exception as error:
                if type(error) is Empty:
                    if time.time() - start > self.total_simulation_time + 2:
                        print("Processor finish", tech)
                        self.processors_finish += 1
                        break
                    continue
                print("Exception", error)

    def print_stat(self):
        start = time.time()
        start_interval = start
        while True:
            if (
                self.processors_finish.value > 0
                and self.generators_finish.value > 0
                and self.processors_finish.value == len(self.list_processor_threads)
                and self.generators_finish.value == len(self.list_generator_threads)
            ):
                print("Finish monitor")
                return

            if time.time() - start_interval < self.stat_interval:
                time.sleep(1)
                continue

            longest = 0
            for tf, value in self.accumulators.items():
                log_str = tf
                for k, v in value.items():
                    if k == "latency":
                        latency = np.mean(v) / 1e6
                        log_str += ". latency: " + str(round(latency, 2)) + " ms"
                        self.state_snapshot[tf]["latency"].append(latency)
                    else:
                        total_processed = self.scale_factor * (
                            v.value
                            * self.generator_setting[tf]["packet_size"]
                            * 8
                            / 1e6
                        )
                        throughput = total_processed / self.stat_interval
                        log_str += ". " + k + ": " + str(round(throughput, 2)) + " mbps"
                        v.value = 0
                print(log_str)
                longest = max(longest, len(log_str))

            for tech, value in self.stat.items():
                log_str = tech
                total_revenue = 0
                total_loss = 0
                total_data = 0
                rev_factor = self.processor_setting[tech]["revenue_factor"]
                for tf, v in value.items():
                    tf_rev = (
                        self.generator_setting[tf]["price"]
                        * self.generator_setting[tf]["packet_size"]
                        * v["revenue"].value
                        / 8
                        / 1e6
                    )
                    tf_loss = (
                        self.generator_setting[tf]["price"]
                        * self.generator_setting[tf]["packet_size"]
                        * v["loss"].value
                        / 8
                        / 1e6
                    )
                    total_revenue += self.scale_factor * tf_rev
                    total_loss += self.scale_factor * tf_loss
                    total_data += self.scale_factor * (
                        v["packet_count"].value
                        * self.generator_setting[tf]["packet_size"]
                        * 8
                    )
                    throughput = self.scale_factor * (
                        v["packet_count"].value
                        * self.generator_setting[tf]["packet_size"]
                        * 8
                        / (self.stat_interval * 1e6)
                    )
                    self.state_snapshot[tf]["throughput"][tech].append(throughput)
                    log_str += (
                        ". "
                        + tf
                        + ". R: "
                        + str(round(tf_rev, 2))
                        + "$. L: "
                        + str(round(tf_loss, 2))
                        + "$. T: "
                        + str(round(throughput, 2))
                        + " mbps"
                    )
                    v["packet_count"].value = 0
                log_str += (
                    "|All. R: "
                    + str(round(total_revenue * rev_factor, 2))
                    + "$. L: "
                    + str(round(total_loss * rev_factor, 2))
                    + "$. T: "
                    + str(round(total_data / (self.stat_interval * 1e6), 2))
                    + " mbps"
                )
                print(log_str)
                longest = max(longest, len(log_str))

            separator = "=" * longest
            print("Queue.", self.get_queue_status())
            print(separator)
            start_interval = time.time()

    def get_queue_status(self):
        result = ""
        for tech, value in self.queue.items():
            max_queue_size = (
                self.processor_setting[tech]["limit"]
                * self.processor_setting[tech]["num_thread"]
            )
            percent = value.qsize() / max_queue_size
            result += tech + ": " + str(round(percent, 2)) + ". "
            for tf, val in self.generator_setting.items():
                self.state_snapshot[tf]["queue"][tech].append(percent)
        return result

    def step(self, action):
        # (TF[i]_throughtput_tech[k], TF[i]_latency, queue_load_tech[k])
        self.state_snapshot = {}
        for tf, setting in self.generator_setting.items():
            self.state_snapshot[tf] = {"latency": [], "throughput": {}, "queue": {}}
            for tech, v in self.processor_setting.items():
                self.state_snapshot[tf]["throughput"][tech] = []
                self.state_snapshot[tf]["queue"][tech] = []
        self.generators = {}
        self.accumulators = {}
        self.stat = {}
        self.processors_finish = AtomicLong(0)
        self.generators_finish = AtomicLong(0)
        self.list_generator_threads = []
        self.list_processor_threads = []
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
                self.list_processor_threads.append(processor)

            for tf in self.generator_setting.keys():
                self.stat[key][tf] = {
                    "revenue": AtomicLong(0),
                    "packet_count": AtomicLong(0),
                    "loss": AtomicLong(0),
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
                self.list_generator_threads.append(packet_generator_thread)

        start_step = time.time()

        for t in self.list_processor_threads:
            t.start()

        for t in self.list_generator_threads:
            t.start()

        log_thread = threading.Thread(target=self.print_stat)
        log_thread.start()

        for t in self.list_processor_threads:
            t.join()

        for t in self.list_generator_threads:
            t.join()

        log_thread.join()

        print(
            "Finish step. Queue {" + self.get_queue_status() + "}.",
            "Total time:",
            str(round(time.time() - start_step, 2)),
            "s",
        )
        # print(self.state_snapshot)
        state, reward, terminated = self.get_current_state_and_reward()
        # print(state, reward)
        return state, reward, terminated, {}

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def get_current_state_and_reward(self):
        state_arr = []
        reward_qos = []
        reward_revenue = 0
        qos_violated = 0
        queue_violated = 0
        for tf, value in self.state_snapshot.items():
            # [latency, nr_throughput, wf_throughput, nr_queue, wf_queue]
            tf_qos_latency = self.generator_setting[tf]["qos_latency_ms"]
            mean_latency = np.mean(self.state_snapshot[tf]["latency"]).item()
            qos_ratio = mean_latency / tf_qos_latency
            tf_val = [qos_ratio]
            if qos_ratio > 1:
                qos_violated += 1
            reward_qos.append(1 / qos_ratio)
            # normalize
            for tech, val in value["throughput"].items():
                arr = np.array(val)
                non_zero_elements = arr[arr != 0]
                mean_non_zero = 0
                max_mbps = (
                    self.processor_setting[tech]["num_thread"]
                    * self.processor_setting[tech]["rate"]
                )
                if non_zero_elements.size > 0:
                    mean_non_zero = np.mean(non_zero_elements)
                # normalize throughput
                tf_val.append(mean_non_zero / max_mbps)

            for tech, val in value["queue"].items():
                arr = np.array(val)
                mean_non_zero = 0
                if arr.size > 0:
                    mean_non_zero = np.mean(arr)
                if mean_non_zero > self.queue_max_utilization:
                    queue_violated += 1
                tf_val.append(mean_non_zero)
            state_arr.append(np.array(tf_val))

        final_state = np.array(state_arr)
        # print("origin", state)
        # print("sigmoid", self.sigmoid(state))
        if self.sigmoid_state:
            final_state = self.sigmoid(final_state)
        # maxmimum revenue
        max_rev_tech = max(
            self.processor_setting,
            key=lambda item: self.processor_setting[item]["revenue_factor"],
        )
        processed_tf = {}
        for tf, value in self.generator_setting.items():
            processed_tf[tf] = 0
        total_revenue = 0
        for tech, value in self.stat.items():
            for tf, val in value.items():
                processed = (
                    val["revenue"].value
                    * self.generator_setting[tf]["packet_size"]
                    * self.generator_setting[tf]["price"]
                    / 8
                    / 1e6
                )
                processed_tf[tf] += processed
                total_revenue += (
                    processed * self.processor_setting[tech]["revenue_factor"]
                )

        max_revenue = (
            sum(processed_tf.values())
            * self.processor_setting[max_rev_tech]["revenue_factor"]
        )

        reward_revenue = 0
        if max_revenue > 0:
            reward_revenue = total_revenue / max_revenue
        final_reward = (
            self.reward_factor["qos"] * self.sigmoid(np.mean(reward_qos).item())
            + self.reward_factor["revenue"] * reward_revenue
        )
        terminal = qos_violated == 0 and queue_violated == 0
        print(
            "Max rev:",
            max_revenue,
            "real rev:",
            total_revenue,
            "qos_violated:",
            qos_violated,
            "queue_violated:",
            queue_violated,
            "terminated:",
            terminal,
            "reward:",
            final_reward,
        )
        if max_revenue == 0:
            return final_state, 0, terminal
        return final_state, final_reward, terminal

    def reset(self):
        print("Reset env")
        self.init_queue()
        return np.zeros(self.state_shape), {}
        # for k, q in self.queue.items():
        #     q.clear()

    def render(self, mode="human"):
        print("Render not implemented")
        pass

    def get_action_shape(self):
        return self.action_space.sample().shape

    def get_state_shape(self):
        return self.observation_space.sample().shape

    def close(self):
        pass


# env = NetworkEnv()
# observation = env.reset()
# print(env.get_action_shape())
# print(env.get_state_shape())
# while True:
#     action = env.action_space.sample()
#     # action = [1, 0.25, 0.5]
#     observation, reward, done, _ = env.step(action)
#     print("observation", observation, observation.shape)
#     env.reset()
#     time.sleep(1)
#     print("$$$$$$$$$$$$$$$$$$$$$")
# action = [1, 1, 1]
# NR. TF1. R: 63.28$. L: 11.38$. T: 143.88 mbps. TF2. R: 221.95$. L: 33.4$. T: 251.45 mbps. TF3. R: 10.18$. L: 3.46$. T: 77.99 mbps|All. R: 2954.07$. L: 482.42$. T: 473.32 mbps
# action = [1, 0.5, 1]
# NR. TF1. R: 70.45$. L: 0.0$. T: 145.44 mbps. TF2. R: 117.94$. L: 0.0$. T: 121.31 mbps. TF3. R: 13.43$. L: 0.0$. T: 94.54 mbps|All. R: 2018.15$. L: 0.0$. T: 361.28 mbps
