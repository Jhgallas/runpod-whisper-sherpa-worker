import base64
import json
import os

# Read the audio file and encode it to base64
audio_filename = 'test.mp3'
with open(audio_filename, 'rb') as audio_file:
    audio_data = audio_file.read()
    base64_encoded = base64.b64encode(audio_data).decode('utf-8')

# Detect extension
_, ext = os.path.splitext(audio_filename)
ext = ext.lstrip('.')  # Remove the leading dot

# Create the JSON structure
data = {
    "input": {
        "file": base64_encoded,
        "ext": ext
    }
}

# Save to test_input.json
with open('test_input.json', 'w') as json_file:
    json.dump(data, json_file, indent=4)

print("Successfully encoded test.mp3 to base64 and saved to test_input.json")
