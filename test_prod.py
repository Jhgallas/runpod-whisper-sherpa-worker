import json
import os
import requests

# Read test_input.json
with open('test_input.json', 'r') as json_file:
    payload = json.load(json_file)

# Get API key from environment variable
api_key = os.getenv('RUNPOD_API_KEY')
if not api_key:
    raise ValueError("RUNPOD_API_KEY environment variable is not set")

# RunPod API endpoint
url = "YOUR RUNDPOD ENDPOINT HERE"

# Headers
headers = {
    'Content-Type': 'application/json',
    'Authorization': f'Bearer {api_key}'
}

# Make POST request
response = requests.post(url, json=payload, headers=headers)

# Print response
print(f"Status Code: {response.status_code}")
print(f"Response: {response.text}")
