import time
start = time.monotonic()
print("Beginning")
for i in range(6):
    print(i)
    time.sleep(5)
end = time.monotonic()
print(f"Ended after {end-start} seconds")
