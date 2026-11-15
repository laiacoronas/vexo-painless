"""Deepgram connection and file transcription using the project's .env key."""

import argparse
import json
import mimetypes
import os
from pathlib import Path
import shlex
import sys
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


ENV_PATH = Path(__file__).resolve().parents[1] / '.env'
API_URL = 'https://api.deepgram.com/v1'


def load_api_key(env_path=ENV_PATH):
    key = os.environ.get('DEEPGRAM_API_KEY', '').strip()
    if not key and env_path.exists():
        for line in env_path.read_text().splitlines():
            name, separator, value = line.strip().removeprefix('export ').partition('=')
            if separator and name.strip() == 'DEEPGRAM_API_KEY':
                parts = shlex.split(value, comments=True)
                if len(parts) > 1:
                    raise ValueError('DEEPGRAM_API_KEY must contain a single value.')
                key = parts[0] if parts else ''
    if not key:
        raise ValueError('Set DEEPGRAM_API_KEY in the project .env file or environment.')
    return key

