import os

# Detect the number of available CPU cores  #72 processors
available_processors = os.cpu_count()
print(f"Available processors: {available_processors}")