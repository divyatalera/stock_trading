import schedule
import time
import os

def run_script():
    os.system("/Users/divyatalera/Desktop/trading/upstox_env/bin/python /Users/divyatalera/Desktop/trading/test1.py")

# Schedule the task every 30 minutes
schedule.every(1).minutes.do(run_script)

while True:
    schedule.run_pending()
    time.sleep(1)


