#!/usr/bin/env python

############
#
# Copyright (c) 2025 Joseph DelPreto / MIT CSAIL
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
# WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR
# IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# Created 2025 by Joseph DelPreto [https://josephdelpreto.com].
# [add additional updates and authors as desired]
#
############

#######################################
# Sample cron entry to run this script:
# @reboot sleep 30 && (/home/nvidia/miniconda3/bin/python /home/nvidia/drl/post_ip_address.py > /home/nvidia/drl/post_ip_address_output.txt)
#######################################

import time
from datetime import datetime

import netifaces as ni
import requests
import subprocess
import os
import traceback

###################################################
# CONFIGURATION
###################################################

# Specify the Google Form to use.
form_url = 'https://docs.google.com/forms/d/e/1FAIpQLSeJJAZJytekw7Ss-hdvwJ42Rzkn4uRxzCCAlilmNpFUQFN93g/formResponse'
form_entry_ids = {'ip':'entry.724467893', 'interface':'entry.1020220735', 'ssid':'entry.761336835'}

# Specify polling periods.
poll_ip_period_s = 60
kill_filepath = '/home/pi/Desktop/kill_postip_process.txt'
poll_kill_file_period_s = 60

# Specify network interfaces to check
interfaces = ['eth2', 'eth1', 'eth0']

###################################################
# HELPERS
###################################################

# Get a string of the current date and time.
def now_str():
  return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")

# Get the IP address on the specified interface.
def get_ip(interface):
  try:
    ip = str(ni.ifaddresses(interface)[ni.AF_INET][0]['addr'])
  except:
    ip = None
  return ip

# Get the SSID of the connected Wi-Fi network.
def get_ssid(second_attempt=False):
  subprocess_result = subprocess.Popen('iwgetid',shell=True,stdout=subprocess.PIPE)
  subprocess_output = subprocess_result.communicate()[0],subprocess_result.returncode
  network_name = subprocess_output[0].decode('utf-8')
  network_name = network_name.strip()
  if len(network_name) == 0:
    if second_attempt:
      network_name = None
    else:
      time.sleep(20)
      network_name = get_ssid(second_attempt=True)
  return str(network_name)

###################################################
# MAIN LOGIC
###################################################

# Initialize state.
next_post_time_s = time.time()
prev_ips = dict([(interface, None) for interface in interfaces])
prev_ssid = None

# Periodically check for IP address changes.
while not os.path.exists(kill_filepath):
  if time.time() >= next_post_time_s:
    print('%s: Time to check IP addresses' % now_str())
    for interface in interfaces:
      ip = get_ip(interface)
      ssid = get_ssid()
      print('%s:  Interface "%s": See IP and prev IP:' % (interface, now_str()), ip, prev_ips[interface], type(ip), type(prev_ips[interface]))
      print('%s:    See SSID and prev SSID:' % (now_str()), ssid, prev_ssid, type(ssid), type(prev_ssid))
      if (ip != prev_ips[interface]) or (ssid != prev_ssid and 'wlan' in interface):
        try:
          print('%s:    Sending IP/SSID!' % now_str())
          submission = {form_entry_ids['ip']: str(ip), form_entry_ids['interface']: interface, form_entry_ids['ssid']: str(ssid)}
          requests.post(form_url, submission)
          prev_ips[interface] = ip
          prev_ssid = ssid
          print('')
        except:
          print('%s: ERROR SENDING IP ADDRESS' % now_str())
          traceback.print_exc()
          print('')
      time.sleep(2)
    next_post_time_s = time.time() + poll_ip_period_s
  time.sleep(max(0, min(poll_kill_file_period_s, next_post_time_s - time.time())))

# Clean up.
print('%s: Done!' % now_str())
print('\n')
if os.path.exists(kill_filepath):
  os.remove(kill_filepath)



