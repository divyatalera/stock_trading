# import schedule
# import streamlit as st
# import time
# from datetime import datetime

# def print_hello_world():
#     st.write(f"Hello world at {datetime.now()}")

# # Schedule the task
# schedule.every(1).minutes.do(print_hello_world)

# # Run the scheduler
# while True:
#     schedule.run_pending()
#     time.sleep(1)  # Wait a second before checking again

import os
os.system("streamlit run /workspaces/stock_trading/strategy.py --server.port=8501")
