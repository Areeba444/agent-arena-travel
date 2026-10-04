from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
ROOT = Path(__file__).resolve().parent.parent
class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / '.env', extra='ignore')
    host: str = '127.0.0.1'
    port: int = 8000
    model_provider: str = 'unconfigured'
    model_name: str = 'openai/gpt-oss-120b'
    allowed_models: str = 'openai/gpt-oss-120b,openai/gpt-oss-20b'
    groq_api_key: str = ''
    max_steps: int = Field(default=6, ge=1, le=6)
    max_tool_retries: int = Field(default=2, ge=0, le=2)
    max_output_tokens: int = Field(default=1500, ge=1)
    max_input_chars: int = Field(default=20000, ge=1000)
    model_call_timeout_seconds: float = Field(default=15.0, gt=0)
    run_timeout_seconds: float = Field(default=40, gt=0, le=40)
    rate_limit_per_minute: int = Field(default=30, ge=1)
    max_concurrency: int = Field(default=3, ge=1)

    @property
    def allowed_models_list(self) -> list[str]:
        if not self.allowed_models or not self.allowed_models.strip():
            return [self.model_name] if self.model_name else []
        return [m.strip() for m in self.allowed_models.split(',') if m.strip()]

settings = Settings()
