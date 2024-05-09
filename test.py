import queue

# Define a custom object class
class CustomObject:
    def __init__(self, name, value):
        self.name = name
        self.value = value

    def __repr__(self):
        return f"CustomObject(name='{self.name}', value={self.value})"

# Create a queue
custom_queue = queue.Queue()

# Put custom objects into the queue
custom_queue.put(CustomObject("Object 1", 10))
custom_queue.put(CustomObject("Object 2", 20))
custom_queue.put(CustomObject("Object 3", 30))

print(custom_queue.full())
# Get custom objects from the queue
obj1 = custom_queue.get()
obj2 = custom_queue.get()
obj3 = custom_queue.get()

# Display the retrieved objects
print("Retrieved Objects:")
print(obj1)
print(obj2)
print(obj3)