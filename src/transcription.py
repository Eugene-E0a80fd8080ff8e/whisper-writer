import io
import time
import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel
from openai import OpenAI, BadRequestError

from utils import ConfigManager

# Rough per-minute price estimates (USD) used for the verbose cost line.
# Informational only; override per binding with `price_per_minute`.
_PRICE_PER_MINUTE = {
    'whisper-1': 0.006,
    'gpt-4o-transcribe': 0.006,
    'gpt-4o-mini-transcribe': 0.003,
    'gpt-transcribe': 0.006,
    'grok-voice-transcribe-1.0': 0.10 / 60.0,
    'grok-voice-transcribe-2.0': 0.10 / 60.0,
    'whisper-large-v3': 0.111,
    'whisper-large-v3-turbo': 0.04,
}

def create_local_model():
    """
    Create a local model using the faster-whisper library.
    """
    ConfigManager.console_print('Creating local model...')
    local_model_options = ConfigManager.get_config_section('model_options')['local']
    compute_type = local_model_options['compute_type']
    model_path = local_model_options.get('model_path')

    if compute_type == 'int8':
        device = 'cpu'
        ConfigManager.console_print('Using int8 quantization, forcing CPU usage.')
    else:
        device = local_model_options['device']

    try:
        if model_path:
            ConfigManager.console_print(f'Loading model from: {model_path}')
            model = WhisperModel(model_path,
                                 device=device,
                                 compute_type=compute_type,
                                 download_root=None)  # Prevent automatic download
        else:
            model = WhisperModel(local_model_options['model'],
                                 device=device,
                                 compute_type=compute_type)
    except Exception as e:
        ConfigManager.console_print(f'Error initializing WhisperModel: {e}')
        ConfigManager.console_print('Falling back to CPU.')
        model = WhisperModel(model_path or local_model_options['model'],
                             device='cpu',
                             compute_type=compute_type,
                             download_root=None if model_path else None)

    ConfigManager.console_print('Local model created.')
    return model

def _binding_name(binding):
    return binding.get('name') if isinstance(binding, dict) else None

def _verbose_request_header(binding, api_config):
    """Print which model is used and which prompt/options are sent."""
    ConfigManager.verbose_print(f"[verbose] binding '{_binding_name(binding)}' -> "
                                f"provider={api_config['provider']} model={api_config['model']!r} "
                                f"base_url={api_config['base_url']}")
    ConfigManager.verbose_print(f"[verbose] prompt sent: {api_config['initial_prompt']!r}")
    ConfigManager.verbose_print(f"[verbose] language={api_config['language']!r} "
                                f"temperature={api_config['temperature']}")

def _price_estimate(api_config, audio_seconds, response_data):
    """Return a human-readable estimated cost line, or None when the price is unknown."""
    model = api_config['model'] or ''
    rate = api_config.get('price_per_minute')
    if rate is None:
        rate = _PRICE_PER_MINUTE.get(model)
    if rate is None:
        # e.g. OpenRouter-style "openai/whisper-1"
        rate = _PRICE_PER_MINUTE.get(model.split('/')[-1])
    if rate is None:
        return None

    duration = audio_seconds
    if isinstance(response_data, dict):
        response_duration = response_data.get('duration')
        if isinstance(response_duration, (int, float)):
            duration = float(response_duration)

    cost = duration / 60.0 * rate
    return f'${cost:.5f} (estimated, {model} @ ${rate:.6f}/min, {duration:.2f}s audio)'

def _verbose_request_summary(api_config, audio_seconds, elapsed, response_data):
    """Print request timing, the full response metadata, usage and a cost estimate."""
    ConfigManager.verbose_print(f"[verbose] request took {elapsed:.2f}s (audio {audio_seconds:.2f}s)")
    ConfigManager.verbose_print(f"[verbose] response: {response_data}")

    usage = None
    if isinstance(response_data, dict):
        usage = response_data.get('usage') or (response_data.get('x_groq') or {}).get('usage')
    if usage:
        ConfigManager.verbose_print(f"[verbose] usage: {usage}")
    else:
        ConfigManager.verbose_print("[verbose] usage: provider returned no token usage")

    price = _price_estimate(api_config, audio_seconds, response_data)
    if price:
        ConfigManager.verbose_print(f"[verbose] cost: {price}")
    else:
        ConfigManager.verbose_print(f"[verbose] cost: unknown (no price data for model '{api_config['model']}'; "
                                    "set 'price_per_minute' in the binding to get an estimate)")

def transcribe_local(audio_data, local_model=None, binding=None):
    """
    Transcribe an audio file using a local model. Common options (language,
    initial_prompt, temperature) can be overridden per binding.
    """
    if not local_model:
        local_model = create_local_model()
    model_options = ConfigManager.get_config_section('model_options')
    common_options = ConfigManager.resolve_binding_common_options(binding)

    if ConfigManager.is_verbose():
        local_options = model_options['local']
        ConfigManager.verbose_print(f"[verbose] binding '{_binding_name(binding)}' -> local faster-whisper "
                                    f"model={local_options['model']!r} device={local_options['device']} "
                                    f"compute_type={local_options['compute_type']}")
        ConfigManager.verbose_print(f"[verbose] prompt sent: {common_options['initial_prompt']!r}")
        ConfigManager.verbose_print(f"[verbose] language={common_options['language']!r} "
                                    f"temperature={common_options['temperature']}")

    # Convert int16 to float32
    audio_data_float = audio_data.astype(np.float32) / 32768.0
    sample_rate = ConfigManager.get_config_section('recording_options').get('sample_rate') or 16000

    start_time = time.perf_counter()
    response = local_model.transcribe(audio=audio_data_float,
                                      language=common_options['language'],
                                      initial_prompt=common_options['initial_prompt'],
                                      condition_on_previous_text=model_options['local']['condition_on_previous_text'],
                                      temperature=common_options['temperature'],
                                      vad_filter=model_options['local']['vad_filter'],)
    elapsed = time.perf_counter() - start_time

    if ConfigManager.is_verbose():
        audio_seconds = len(audio_data) / sample_rate
        ConfigManager.verbose_print(f"[verbose] local transcription took {elapsed:.2f}s "
                                    f"(audio {audio_seconds:.2f}s)")
        info = response[1] if isinstance(response, tuple) and len(response) > 1 else None
        if info is not None:
            try:
                ConfigManager.verbose_print(f"[verbose] response info: {dict(info._asdict())}")
            except Exception:
                ConfigManager.verbose_print(f"[verbose] response info: {info}")

    return ''.join([segment.text for segment in list(response[0])])

def _audio_to_wav(audio_data):
    """Convert a numpy int16 audio array to an in-memory WAV BytesIO buffer."""
    byte_io = io.BytesIO()
    sample_rate = ConfigManager.get_config_section('recording_options').get('sample_rate') or 16000
    sf.write(byte_io, audio_data, sample_rate, format='wav')
    byte_io.seek(0)
    return byte_io

def _transcribe_xai(audio_file, api_config, audio_seconds):
    """
    Transcribe an audio file using xAI's speech-to-text API (POST {base_url}/stt).

    xAI's transcription endpoint is not OpenAI-compatible: it has no `prompt`
    or `temperature` parameters, so initial_prompt/temperature are ignored for
    this provider (a notice is printed when initial_prompt is set). Extra form
    fields such as `format` or `keyterm` can be supplied via a binding's
    `stt_options` map.
    """
    import requests

    if api_config['initial_prompt'] or api_config['temperature']:
        ConfigManager.console_print("Note: the xAI provider does not support initial_prompt/temperature; they will be ignored.")

    base_url = (api_config['base_url'] or 'https://api.x.ai/v1').rstrip('/')

    # Option fields must precede `file` in the multipart form.
    fields = [('model', api_config['model'] or 'grok-voice-transcribe-2.0')]
    if api_config['language']:
        fields.append(('language', api_config['language']))
    for key, value in (api_config.get('stt_options') or {}).items():
        if value is None or value is False:
            continue
        if isinstance(value, (list, tuple)):
            # Repeatable fields (e.g. keyterm)
            fields.extend((key, str(item)) for item in value)
        elif value is True:
            fields.append((key, 'true'))
        else:
            fields.append((key, str(value)))

    start_time = time.perf_counter()
    response = requests.post(
        f'{base_url}/stt',
        headers={'Authorization': f"Bearer {api_config['api_key']}"},
        data=fields,
        files={'file': ('audio.wav', audio_file, 'audio/wav')},
        timeout=120,
    )
    elapsed = time.perf_counter() - start_time
    response.raise_for_status()
    data = response.json()
    if ConfigManager.is_verbose():
        _verbose_request_summary(api_config, audio_seconds, elapsed, data)
    return data.get('text', '')

def transcribe_api(audio_data, binding=None):
    """
    Transcribe an audio file using an OpenAI-compatible API or the xAI API,
    depending on the binding's provider.
    """
    api_config = ConfigManager.resolve_binding_api_config(binding)
    sample_rate = ConfigManager.get_config_section('recording_options').get('sample_rate') or 16000
    audio_seconds = len(audio_data) / sample_rate
    _verbose_request_header(binding, api_config)

    if api_config['provider'] == 'xai':
        return _transcribe_xai(_audio_to_wav(audio_data), api_config, audio_seconds)

    client = OpenAI(
        api_key=api_config['api_key'] or None,
        base_url=api_config['base_url']
    )

    file = ('audio.wav', _audio_to_wav(audio_data), 'audio/wav')

    def create_request(include_optional_params: bool):
        kwargs = {
            'model': api_config['model'],
            'file': file,
        }
        if include_optional_params:
            if api_config['language']:
                kwargs['language'] = api_config['language']
            if api_config['initial_prompt']:
                kwargs['prompt'] = api_config['initial_prompt']
            if api_config['temperature'] is not None:
                kwargs['temperature'] = api_config['temperature']
        return client.audio.transcriptions.create(**kwargs)

    start_time = time.perf_counter()
    try:
        response = create_request(include_optional_params=True)
    except BadRequestError:
        # Some providers reject optional parameters; retry with file and model only.
        ConfigManager.console_print(f"Provider '{api_config['provider']}' rejected optional parameters. "
                                    "Retrying with file and model only.")
        file[1].seek(0)
        response = create_request(include_optional_params=False)
    elapsed = time.perf_counter() - start_time

    if ConfigManager.is_verbose():
        try:
            response_data = response.model_dump(exclude_none=True)
        except Exception:
            response_data = {'text': response.text}
        _verbose_request_summary(api_config, audio_seconds, elapsed, response_data)
    return response.text

def post_process_transcription(transcription):
    """
    Apply post-processing to the transcription.
    """
    transcription = transcription.strip()
    post_processing = ConfigManager.get_config_section('post_processing')
    if post_processing['remove_trailing_period'] and transcription.endswith('.'):
        transcription = transcription[:-1]
    if post_processing['add_trailing_space']:
        transcription += ' '
    if post_processing['remove_capitalization']:
        transcription = transcription.lower()

    return transcription

def transcribe(audio_data, local_model=None, binding_name=None):
    """
    Transcribe audio data using the API or a local model, depending on the
    configuration of the binding with the given name.
    """
    if audio_data is None:
        return ''

    binding = ConfigManager.get_binding_by_name(binding_name)
    if ConfigManager.binding_uses_api(binding):
        transcription = transcribe_api(audio_data, binding)
    else:
        transcription = transcribe_local(audio_data, local_model, binding)

    return post_process_transcription(transcription)
