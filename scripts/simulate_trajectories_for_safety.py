
import os
import h5py
import numpy as np
import time
from datetime import datetime
import matplotlib.pyplot as plt
import scipy.spatial.transform as tf
import traceback
import threading
import subprocess

from RainbowInterface import RainbowInterface
from ExperimentInterface import ExperimentInterface

from pathlib import Path
# Find the "data" folder by searching up and down from the current folder.
data_root_dir = None
current_dir = Path(__file__).resolve().parent
for p in current_dir.rglob("*"):
  if p.is_dir() and p.name == "data":
    data_root_dir = p.resolve()
if data_root_dir is None:
  parent = current_dir
  for _ in range(5):
    data_root_dir = parent/"data"
    if data_root_dir.exists() and data_root_dir.is_dir():
      data_root_dir = data_root_dir.resolve()
      break
    else:
      data_root_dir = None
    parent = parent.parent
experiments_root_dir = os.path.join(data_root_dir, 'best_experiments')

output_filepath = 'simulation_results.csv'
if not os.path.exists(output_filepath):
  # Create the file and write the headers.
  fout = open(output_filepath, 'a')
  fout.write('Simulation Start Time,Model,Task,Trial Index')
  fout.write(',Simulation Test 1,Simulation Test 2,Simulation Test 3')
  fout.close()
else:
  # Read existing contents.
  fout = open(output_filepath, 'r')
  lines = fout.readlines()
  fout.close()
  fout = open(output_filepath, 'w')
  # write the headers
  fout.write(lines[0].strip())
  lines = lines[1:]
  # write back complete lines
  for line in lines:
    line_entries = [x.strip() for x in line.split(',')]
    print('see line entries:', line_entries)
    if len(line_entries) < 5:
      continue
    if len(line_entries) < 7 and '1' not in line_entries[4:]:
      continue
    fout.write('\n' + line.strip())
  fout.close()
  
# # Kill any lingering processes from previous runs.
# print('killing past processes')
def wait_for_docker_shutdown(timeout_s=30, poll_interval_s=1):
  start = time.time()
  while time.time() - start < timeout_s:
    result = subprocess.run(["pgrep", "-f", "docker"],
                            stdout=subprocess.PIPE,
                            text=True)
    if not result.stdout.strip():
      # no docker processes found
      return True
    time.sleep(poll_interval_s)
  return False
def wait_for_docker_startup(timeout_s=30, poll_interval_s=1):
  start = time.time()
  while time.time() - start < timeout_s:
    result = subprocess.run(["pgrep", "-f", "docker"],
                            stdout=subprocess.PIPE,
                            text=True)
    if result.stdout.strip():
      # docker processes found
      return True
    time.sleep(poll_interval_s)
  return False
# def cleanup_processes():
#   # 1. Kill all docker processes
#   try:
#     subprocess.run(["sudo", "pkill", "-f", "docker"], check=False)
#   except Exception as e:
#     pass
#   # 2. Kill all other instances of this script
#   current_pid = os.getpid()
#   try:
#     # List processes that match the script name
#     result = subprocess.run(
#       ["pgrep", "-f", "simulate_trajectories_for_safety"],
#       stdout=subprocess.PIPE,
#       text=True,
#       check=False
#     )
#     if result.stdout:
#       for pid_str in result.stdout.splitlines():
#         pid = int(pid_str)
#         if pid != current_pid:
#           subprocess.run(["kill", "-9", str(pid)], check=False)
#   except Exception as e:
#     pass
# cleanup_processes()

# Start a thread to continuously make sure the docker is running.
DOCKER_CMD = [
    "sudo", "docker", "run", "--rm",
    "-e", "DISPLAY=128.30.9.6:0",
    "-v", "/tmp/.X11-unix:/tmp/.X11-unix",
    "-p", "50051:50051",
    "rainbowroboticsofficial/rby1-sim"
]
stop_docker_thread = False  # global flag
restart_docker = False
docker_is_running = False  # global flag
def run_docker():
  print('docker thread')
  global stop_docker_thread, docker_is_running, restart_docker
  while not stop_docker_thread:
    print('docker thread loop')
    try:
      # Launch the docker process (non-blocking)
      print('docker launch command')
      proc = subprocess.Popen(DOCKER_CMD)
      print('waiting')
      wait_for_docker_startup(timeout_s=30)
      time.sleep(5)
      print('docker is launched!')
      restart_docker = False
      docker_is_running = True
      # Poll until it finishes or a stop/restart flag is set
      while proc.poll() is None and not stop_docker_thread and not restart_docker:
        time.sleep(0.1)
      docker_is_running = False # is already stopped or about to be stopped, so others shouldn't use it
      should_restart_docker = restart_docker # get a local copy of the global flag
      restart_docker = False # reset the global flag
      # Stop the docker if desired.
      if stop_docker_thread or should_restart_docker:
        proc.terminate()  # try graceful stop
        try:
          proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
          proc.kill()  # force kill if needed
        for i in range(3):
          subprocess.run(["sudo", "pkill", "-f", "docker"], check=False)
          time.sleep(1)
        wait_for_docker_shutdown(timeout_s=60)
        time.sleep(5)
        if stop_docker_thread:
          break  # exit the loop
      else:
        pass
    except Exception as e:
      docker_is_running = False
    time.sleep(0.5)
# Start background thread to launch the docker.
print('starting docker thread')
docker_thread = threading.Thread(target=run_docker, daemon=True)
docker_thread.start()

# Loop through each task, model, and trial to simulate it.
for task in sorted(os.listdir(data_root_dir)):
  task_dir = os.path.join(data_root_dir, task)
  if task == 'scooping_powder':
    task_name_for_trajectory_loading = 'scoop'
  if task == 'pouring':
    task_name_for_trajectory_loading = 'pour'
  if task == 'stirring':
    task_name_for_trajectory_loading = 'stir'
  # model_filenames = ['human']
  model_filenames = ['human'] + sorted(os.listdir(task_dir))
  for model_filename in model_filenames:
    use_human_trajectories = model_filename == 'human'
    if not use_human_trajectories:
      model_filepath = os.path.join(task_dir, model_filename)
      model = model_filename.replace('_inference.hdf5', '')
    else:
      model_filename = sorted(os.listdir(task_dir))[0] # can use any model since they all have the same human trajectories
      model_filepath = os.path.join(task_dir, model_filename)
      model = os.path.basename(model_filename).replace('_inference.hdf5', '')
    experiment_interface = None
    # Loop through trial indexes until one doesn't exist.
    reached_end_of_test_set = False
    for trial_index in range(1000):
      # Check if already finished the test set.
      if reached_end_of_test_set:
        break
      # Check if the output file already contains results for this trial.
      trial_results_exist = False
      fin = open(output_filepath, 'r')
      for line in fin.readlines():
        try:
          line_split = line.split()
          (timestamp, fin_model, fin_task, fin_trial_index, *_) = line.split(',')
          if (use_human_trajectories and fin_model == 'human') or (not use_human_trajectories and fin_model == model):
            if fin_task == task and int(fin_trial_index) == trial_index:
              trial_results_exist = True
              break
        except:
          pass
      fin.close()
      if trial_results_exist:
        continue
      # Initialize state and stopping criteria state.
      started_output_line = False
      attempt_counter = 0
      simulation_success = False
      # Try to simulate a few times or until success.
      while not simulation_success and attempt_counter < 3:
        attempt_counter += 1
        # Create an experiment interface if needed.
        while experiment_interface is None:
          while not docker_is_running:
            time.sleep(1)
          try:
            experiment_interface = ExperimentInterface(
                model_name=model,
                use_human_trajectories=use_human_trajectories,
                simulation=True,
                is_device_upc=False,
                data_folder=None, #os.path.realpath('../data'),
            )
            experiment_interface.process_commands("speed 1")
          except:
            experiment_interface = None
            time.sleep(1)
        # Load the trajectory.
        try:
          experiment_interface.process_commands("model %s" % model)
          if use_human_trajectories:
            experiment_interface.process_commands("model human")
          experiment_interface.process_commands("load %s %d" % (task_name_for_trajectory_loading, trial_index))
          # Start a line in the results file for this trial.
          if not started_output_line:
            fout = open(output_filepath, 'a')
            fout.write('\n%s,%s,%s,%2d' % (datetime.now().strftime("%Y-%m-%d %H-%M-%S"),
                                           model if not use_human_trajectories else 'human',
                                           task, trial_index))
            fout.close()
            started_output_line = True
        except AssertionError:
          experiment_interface = None
          reached_end_of_test_set = True
          break
        # Go to the home position.
        while not docker_is_running:
          time.sleep(1)
        try:
          experiment_interface.process_commands("home")
          time.sleep(0.5)
          experiment_interface.process_commands("home")
          time.sleep(0.5)
        except AssertionError:
          # Record the failure.
          fout = open(output_filepath, 'a')
          fout.write(',Home fail')
          fout.close()
          # Restart the docker and the experiment interface.
          experiment_interface = None
          restart_docker = True
          while docker_is_running:
            time.sleep(0.5)
          # Try again.
          continue
        # Run the trajectory.
        while not docker_is_running:
          time.sleep(1)
        try:
          experiment_interface.process_commands("run noprompt")
          # Record a success!
          fout = open(output_filepath, 'a')
          fout.write(',1')
          fout.close()
          simulation_success = True
        except:
          # Record the failure.
          fout = open(output_filepath, 'a')
          fout.write(',0')
          fout.close()
          # Restart the docker and the experiment interface.
          experiment_interface = None
          restart_docker = True
          while docker_is_running:
            time.sleep(0.5)
          continue
    
# Stop the docker thread.
stop_docker_thread = True
docker_thread.join()

















