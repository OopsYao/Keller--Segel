from datetime import datetime
import os

time_str = datetime.now().strftime('%m%d-%H%M')
artifacts_dir = f'artifacts/{time_str}'
os.mkdir(artifacts_dir)
