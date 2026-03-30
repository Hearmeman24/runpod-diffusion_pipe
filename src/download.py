#!/usr/bin/env python3
import requests
import argparse
import os
import sys

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, required=True, help="CivitAI model ID to download")
parser.add_argument("-t", "--token", type=str, help="CivitAI API token (if not set in environment)")
args = parser.parse_args()


# Determine the token
token = os.getenv("civitai_token", args.token)
if not token:
    print("Error: no token provided. Set the 'civitai_token' environment variable or use --token.")
    sys.exit(1)

# URL of the file to download
url = f"https://civitai.com/api/v1/model-versions/{args.model}"

# Perform the request
try:
    response = requests.get(url, timeout=30)
except requests.RequestException as e:
    print(f"Error: Failed to connect to CivitAI API: {e}")
    sys.exit(1)

if response.status_code == 200:
    try:
        data = response.json()
    except ValueError:
        print("Error: CivitAI API returned invalid JSON.")
        sys.exit(1)

    if not data.get('files'):
        print("Error: No files found for this model version. Check the model ID.")
        sys.exit(1)

    filename = data['files'][0]['name']
    download_url = data['files'][0]['downloadUrl']

    # Use wget with the resolved token
    os.system(
        f'wget "https://civitai.com/api/download/models/{args.model}?type=Model&format=SafeTensor&token={token}" --content-disposition')
else:
    print(f"Error: Failed to retrieve model metadata (HTTP {response.status_code}).")
    if response.status_code == 404:
        print("  The model version ID may be incorrect.")
    elif response.status_code == 401:
        print("  Authentication failed. Check your CivitAI token.")
    sys.exit(1)
