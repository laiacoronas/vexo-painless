"""Shared conversation identity and Gemini configuration for the voice clients."""

import os
from pathlib import Path
import re
import shlex

ENV_PATH = Path(__file__).resolve().parents[1] / '.env'


def think_settings(prompt, env_path=ENV_PATH):
    """Load server-side Gemini credentials and configure Deepgram's Google endpoint."""
    values = {}
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            name, separator, value = line.strip().removeprefix('export ').partition('=')
            if separator and name.strip() in ('GEMINI_API_KEY', 'GEMINI_MODEL'):
                parts = shlex.split(value, comments=True)
                if len(parts) > 1:
                    raise ValueError(f'{name.strip()} must contain a single value.')
                values[name.strip()] = parts[0] if parts else ''
    key = os.environ.get('GEMINI_API_KEY', values.get('GEMINI_API_KEY', '')).strip()
    model = os.environ.get('GEMINI_MODEL', values.get('GEMINI_MODEL', 'gemini-4-carbon')).strip()
    if not key:
        raise ValueError('Set GEMINI_API_KEY in the project .env file or environment.')
    if not re.fullmatch(r'[a-zA-Z0-9._-]+', model):
        raise ValueError('GEMINI_MODEL must be a valid model identifier.')
    return {
        'provider': {'type': 'google', 'version': 'ai-studio-v1beta'},
        'endpoint': {
            'url': f'https://generativelanguage.googleapis.com/v1beta/models/{model}:streamGenerateContent?alt=sse',
            'headers': {'x-goog-api-key': key},
        },
        'prompt': prompt,
    }


PROMPT = """You are Vexo, a conversational companion that helps a person notice and understand their pain.
Talk with the person naturally about what they say. Use short, warm, direct replies in English, usually one or two sentences. Ask one question at a time when it helps the conversation. Follow their topic and remember what they have already told you. Introduce yourself simply as Vexo when asked who you are.

When a current voice-based pain estimate is supplied, use it as an uncertain signal for a brief check-in. At moderate or high levels, ask how the person is feeling, or follow up on the discomfort they have already described. Give their own report priority. Once they have answered a check-in, continue the conversation; check again when there is a meaningful change. Keep technical scores and model details for explicit questions about them.

Your available context consists of the conversation and any supplied voice estimate. Be honest about uncertainty and your capabilities. Offer ordinary conversational support. Clinical diagnosis and treatment decisions belong with a qualified clinician.

Each voice estimate belongs to its supplied utterance. Background analysis can arrive after a reply; use it as context for a later check-in, and keep its original timing clear. If prediction is disabled or unavailable, respond to the conversation itself. The app records supplied moderate/high voice events automatically; describe a saved event only when the context confirms it was saved."""
