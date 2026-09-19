import os
import yaml
from dotenv import dotenv_values

class ConfigManager:
    _instance = None
    _verbose = False
    _ENV_MISSING = object()
    _api_env_prefixes = {
        'openai': 'OPENAI',
        'groq': 'GROQ',
        'openrouter': 'OPENROUTER',
        'xai': 'XAI',
    }
    _api_default_base_urls = {
        'openai': 'https://api.openai.com/v1',
        'groq': 'https://api.groq.com/openai/v1',
        'openrouter': 'https://openrouter.ai/api/v1',
        'xai': 'https://api.x.ai/v1',
    }
    _api_default_models = {
        'openai': 'whisper-1',
        'xai': 'grok-voice-transcribe-2.0',
    }

    def __init__(self):
        """Initialize the ConfigManager instance."""
        self.config = None
        self.schema = None

    @classmethod
    def initialize(cls, schema_path=None):
        """Initialize the ConfigManager with the given schema path."""
        if cls._instance is None:
            cls._instance = cls()
            cls._instance.schema = cls._instance.load_config_schema(schema_path)
            cls._instance.config = cls._instance.load_default_config()
            cls._instance.load_user_config()

    @classmethod
    def get_schema(cls):
        """Get the configuration schema."""
        if cls._instance is None:
            raise RuntimeError("ConfigManager not initialized")
        return cls._instance.schema

    @classmethod
    def get_schema_default(cls, *keys):
        """Get the default value for a schema entry."""
        schema = cls.get_schema()
        for key in keys:
            if not isinstance(schema, dict):
                return None
            schema = schema.get(key)
            if schema is None:
                return None
        if isinstance(schema, dict) and 'value' in schema:
            return schema['value']
        return None

    @classmethod
    def get_api_provider(cls, provider=None):
        """Return the API provider name. Falls back to the global config provider
        when none is given (e.g. for a binding without an explicit provider)."""
        if provider is None:
            provider = cls.get_config_value('model_options', 'api', 'provider') or 'openai'
        return str(provider).lower()

    @classmethod
    def get_api_env_prefix(cls, provider=None):
        provider = cls.get_api_provider(provider)
        return cls._api_env_prefixes.get(provider, provider.upper())

    @classmethod
    def get_api_env_var_name(cls, suffix, provider=None):
        return f"{cls.get_api_env_prefix(provider)}_{suffix}"

    @classmethod
    def get_api_env_value(cls, suffix, provider=None):
        env_var = cls.get_api_env_var_name(suffix, provider)
        env_values = cls._load_env_file_values()
        if env_var in env_values:
            return env_values[env_var]
        if env_var in os.environ:
            return os.environ.get(env_var)
        return cls._ENV_MISSING

    @classmethod
    def _get_env_or_none(cls, suffix, provider):
        """Return a provider env value or None when it is not set."""
        value = cls.get_api_env_value(suffix, provider)
        return None if value is cls._ENV_MISSING else value

    @classmethod
    def _load_env_file_values(cls):
        env_path = cls.get_env_file_path()
        if os.path.isfile(env_path):
            return dotenv_values(env_path)
        return {}

    @classmethod
    def get_env_file_path(cls):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.abspath(os.path.join(base_dir, '..', '.env'))

    @classmethod
    def get_api_base_url(cls, provider=None):
        """Return the API base URL. When an explicit provider is given, the global
        config value is deliberately skipped so it cannot leak across providers."""
        env_value = cls.get_api_env_value('BASE_URL', provider)
        if env_value is not cls._ENV_MISSING:
            return env_value
        if provider is None:
            config_value = cls.get_config_value('model_options', 'api', 'base_url')
            if config_value:
                return config_value
        resolved_provider = cls.get_api_provider(provider)
        return cls._api_default_base_urls.get(resolved_provider, cls._api_default_base_urls['openai'])

    @classmethod
    def get_api_model(cls, provider=None):
        env_value = cls.get_api_env_value('MODEL', provider)
        if env_value is not cls._ENV_MISSING:
            return env_value
        if provider is None:
            config_value = cls.get_config_value('model_options', 'api', 'model')
            if config_value:
                return config_value
            return cls.get_schema_default('model_options', 'api', 'model')
        return cls._api_default_models.get(cls.get_api_provider(provider))

    @classmethod
    def get_api_key(cls, provider=None):
        env_value = cls.get_api_env_value('API_KEY', provider)
        if env_value is not cls._ENV_MISSING:
            return env_value
        if provider is None:
            return cls.get_config_value('model_options', 'api', 'api_key')
        return None

    @classmethod
    def get_bindings(cls):
        """Return the configured key bindings.

        Normalizes the user's `bindings` list (fills in unique names). When no
        `bindings` are configured, a single default binding is synthesized from
        the legacy single-key configuration (recording_options.activation_key +
        model_options), preserving pre-bindings behavior.
        """
        raw = cls.get_config_value('bindings')
        bindings = []
        if isinstance(raw, list):
            for index, entry in enumerate(raw):
                if not isinstance(entry, dict):
                    continue
                binding = dict(entry)
                if not binding.get('name'):
                    if binding.get('provider'):
                        binding['name'] = str(binding['provider'])
                    elif binding.get('use_api') is False:
                        binding['name'] = 'local'
                    else:
                        binding['name'] = f'binding {index + 1}'
                bindings.append(binding)

        if bindings:
            seen = set()
            for binding in bindings:
                base_name = binding['name']
                counter = 1
                while binding['name'] in seen:
                    counter += 1
                    binding['name'] = f'{base_name} {counter}'
                seen.add(binding['name'])
            return bindings

        return [{
            'name': 'default',
            'activation_key': cls.get_config_value('recording_options', 'activation_key'),
            'use_api': cls.get_config_value('model_options', 'use_api'),
        }]

    @classmethod
    def get_binding_by_name(cls, name):
        """Return the binding with the given name, or an empty dict."""
        if not name:
            return {}
        for binding in cls.get_bindings():
            if binding.get('name') == name:
                return binding
        return {}

    @classmethod
    def binding_uses_api(cls, binding):
        """Return True when a binding should transcribe via an API.

        Falls back to the global model_options.use_api when the binding does
        not specify it."""
        if not isinstance(binding, dict) or binding.get('use_api') is None:
            return bool(cls.get_config_value('model_options', 'use_api'))
        return bool(binding['use_api'])

    @classmethod
    def resolve_binding_common_options(cls, binding):
        """Resolve the common transcription options (language, initial_prompt,
        temperature) for a binding.

        Precedence: binding override -> model_options.common. An explicit
        `null`/empty-string binding value means "use the global value" except
        for initial_prompt, where an empty string disables the prompt.
        """
        binding = binding if isinstance(binding, dict) else {}
        common = cls.get_config_section('model_options', 'common') or {}
        options = {}
        for key in ('language', 'initial_prompt', 'temperature'):
            value = binding.get(key)
            options[key] = common.get(key) if value is None else value
        if options.get('temperature') is None:
            options['temperature'] = 0.0
        return options

    @classmethod
    def resolve_binding_api_config(cls, binding):
        """Resolve the API settings for a binding.

        Precedence: binding override -> provider env vars (values from .env are
        preferred over the shell environment) -> provider defaults. Bindings
        without an explicit `provider` use the legacy global resolution
        (model_options.api), which keeps old configurations working unchanged.
        """
        binding = binding or {}
        provider = binding.get('provider')
        provider = str(provider).lower() if provider else None

        if provider is None:
            resolved_provider = cls.get_api_provider()
            model = binding.get('model') or cls.get_api_model()
            base_url = binding.get('base_url') or cls.get_api_base_url()
            api_key = binding.get('api_key') or cls.get_api_key()
        else:
            resolved_provider = provider
            model = binding.get('model')
            if not model:
                model = cls._get_env_or_none('MODEL', provider)
            if not model:
                model = cls._api_default_models.get(provider)
            base_url = binding.get('base_url')
            if not base_url:
                base_url = cls._get_env_or_none('BASE_URL', provider)
            if not base_url:
                base_url = cls._api_default_base_urls.get(provider, cls._api_default_base_urls['openai'])
            api_key = binding.get('api_key')
            if not api_key:
                api_key = cls._get_env_or_none('API_KEY', provider)

        if not model:
            cls.console_print(f"Warning: no model could be resolved for binding '{binding.get('name')}'; "
                              f"set 'model' in the binding or {cls.get_api_env_var_name('MODEL', resolved_provider)} in .env.")

        common_options = cls.resolve_binding_common_options(binding)

        return {
            'provider': resolved_provider,
            'model': model,
            'base_url': base_url,
            'api_key': api_key,
            'initial_prompt': common_options['initial_prompt'],
            'language': common_options['language'],
            'temperature': common_options['temperature'],
            'stt_options': binding.get('stt_options') or {},
            'price_per_minute': binding.get('price_per_minute'),
        }

    @classmethod
    def get_config_section(cls, *keys):
        """Get a specific section of the configuration."""
        if cls._instance is None:
            raise RuntimeError("ConfigManager not initialized")

        section = cls._instance.config
        for key in keys:
            if isinstance(section, dict) and key in section:
                section = section[key]
            else:
                return {}
        return section

    @classmethod
    def get_config_value(cls, *keys):
        """Get a specific configuration value using nested keys."""
        if cls._instance is None:
            raise RuntimeError("ConfigManager not initialized")

        value = cls._instance.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None
        return value

    @classmethod
    def set_config_value(cls, value, *keys):
        """Set a specific configuration value using nested keys."""
        if cls._instance is None:
            raise RuntimeError("ConfigManager not initialized")

        config = cls._instance.config
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            elif not isinstance(config[key], dict):
                config[key] = {}
            config = config[key]
        config[keys[-1]] = value

    @staticmethod
    def load_config_schema(schema_path=None):
        """Load the configuration schema from a YAML file."""
        if schema_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            schema_path = os.path.join(base_dir, 'config_schema.yaml')

        with open(schema_path, 'r') as file:
            schema = yaml.safe_load(file)
        return schema

    def load_default_config(self):
        """Load default configuration values from the schema."""
        def extract_value(item):
            if isinstance(item, dict):
                if 'value' in item:
                    return item['value']
                else:
                    return {k: extract_value(v) for k, v in item.items()}
            return item

        config = {}
        for category, settings in self.schema.items():
            config[category] = extract_value(settings)
        return config

    def load_user_config(self, config_path=os.path.join('src', 'config.yaml')):
        """Load user configuration and merge with default config."""
        def deep_update(source, overrides):
            for key, value in overrides.items():
                if isinstance(value, dict) and key in source:
                    deep_update(source[key], value)
                else:
                    source[key] = value

        if config_path and os.path.isfile(config_path):
            try:
                with open(config_path, 'r') as file:
                    user_config = yaml.safe_load(file)
                    deep_update(self.config, user_config)
            except yaml.YAMLError:
                print("Error in configuration file. Using default configuration.")

    @classmethod
    def save_config(cls, config_path=os.path.join('src', 'config.yaml')):
        """Save the current configuration to a YAML file."""
        if cls._instance is None:
            raise RuntimeError("ConfigManager not initialized")
        with open(config_path, 'w') as file:
            yaml.dump(cls._instance.config, file, default_flow_style=False)

    @classmethod
    def reload_config(cls):
        """
        Reload the configuration from the file.
        """
        if cls._instance is None:
            raise RuntimeError("ConfigManager not initialized")
        cls._instance.config = cls._instance.load_default_config()
        cls._instance.load_user_config()

    @classmethod
    def config_file_exists(cls):
        """Check if a valid config file exists."""
        config_path = os.path.join('src', 'config.yaml')
        return os.path.isfile(config_path)

    @classmethod
    def console_print(cls, message):
        """Print a message to the console if enabled in the configuration."""
        if cls._instance and cls._instance.config['misc']['print_to_terminal']:
            print(message)

    @classmethod
    def set_verbose(cls, enabled=True):
        """Enable or disable verbose (-v) console output."""
        cls._verbose = bool(enabled)

    @classmethod
    def is_verbose(cls):
        """Return True when verbose (-v) output is enabled."""
        return cls._verbose

    @classmethod
    def verbose_print(cls, message):
        """Print a message to the console when verbose mode (-v) is enabled.

        Still respects misc.print_to_terminal, so silent mode stays silent."""
        if cls._verbose and cls._instance and cls._instance.config['misc']['print_to_terminal']:
            print(message)
