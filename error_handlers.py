import json
import logging
import uuid
import os
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.exception_handlers import RequestValidationError
from starlette.status import HTTP_400_BAD_REQUEST, HTTP_413_REQUEST_ENTITY_TOO_LARGE, HTTP_500_INTERNAL_SERVER_ERROR, HTTP_502_BAD_GATEWAY

# Chargement de la config d'erreur
with open('error_config.json', 'r', encoding='utf-8') as f:
    ERROR_CONFIG = json.load(f)

LANG = ERROR_CONFIG.get('language', 'fr')
LOG_ERRORS = ERROR_CONFIG.get('log_errors', True)
LOG_DEST = ERROR_CONFIG.get('log_destination', 'console')
LOG_FILE = ERROR_CONFIG.get('log_file_path', 'logs/app.log')
LOG_LEVEL = ERROR_CONFIG.get('log_level', 'INFO').upper()
SHOW_TRACE = ERROR_CONFIG.get('show_trace', False)
MESSAGES = ERROR_CONFIG.get('messages', {})

# Initialisation du logging avancée
if LOG_ERRORS:
    log_handlers = []
    if LOG_DEST in ('file', 'both'):
        log_dir = os.path.dirname(LOG_FILE)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(LOG_FILE, encoding='utf-8')
        file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
        log_handlers.append(file_handler)
    if LOG_DEST in ('console', 'both'):
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
        log_handlers.append(console_handler)
    logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO), handlers=log_handlers)

# Exceptions personnalisées
class ValidationError(Exception):
    def __init__(self, code="ValidationError", details=None):
        self.code = code
        self.details = details

class ImageTooLarge(ValidationError):
    def __init__(self, details=None):
        super().__init__(code="ImageTooLarge", details=details)

class UnsupportedImageFormat(ValidationError):
    def __init__(self, details=None):
        super().__init__(code="UnsupportedImageFormat", details=details)

class OllamaError(Exception):
    def __init__(self, details=None):
        self.code = "OllamaError"
        self.details = details

class ConfigError(Exception):
    def __init__(self, details=None):
        self.code = "ConfigError"
        self.details = details

class InternalError(Exception):
    def __init__(self, details=None):
        self.code = "InternalError"
        self.details = details

# Utilitaire pour formater la réponse d'erreur

def error_response(exc, status_code, trace_id=None):
    code = getattr(exc, 'code', 'InternalError')
    msg_dict = MESSAGES.get(code, {})
    message = msg_dict.get(LANG) or msg_dict.get('fr') or str(exc)
    resp = {
        "error": message,
        "code": status_code,
        "type": code,
        "details": getattr(exc, 'details', None)
    }
    if trace_id:
        resp["trace_id"] = trace_id
    return JSONResponse(status_code=status_code, content=resp)

# Handlers FastAPI

def register_error_handlers(app):
    @app.exception_handler(ValidationError)
    async def validation_error_handler(request: Request, exc: ValidationError):
        trace_id = str(uuid.uuid4())
        if LOG_ERRORS:
            logging.warning(f"[ValidationError] {exc.code} {exc.details} trace_id={trace_id}")
        return error_response(exc, HTTP_400_BAD_REQUEST, trace_id)

    @app.exception_handler(ImageTooLarge)
    async def image_too_large_handler(request: Request, exc: ImageTooLarge):
        trace_id = str(uuid.uuid4())
        if LOG_ERRORS:
            logging.warning(f"[ImageTooLarge] {exc.details} trace_id={trace_id}")
        return error_response(exc, HTTP_413_REQUEST_ENTITY_TOO_LARGE, trace_id)

    @app.exception_handler(UnsupportedImageFormat)
    async def unsupported_image_format_handler(request: Request, exc: UnsupportedImageFormat):
        trace_id = str(uuid.uuid4())
        if LOG_ERRORS:
            logging.warning(f"[UnsupportedImageFormat] {exc.details} trace_id={trace_id}")
        return error_response(exc, HTTP_400_BAD_REQUEST, trace_id)

    @app.exception_handler(OllamaError)
    async def ollama_error_handler(request: Request, exc: OllamaError):
        trace_id = str(uuid.uuid4())
        if LOG_ERRORS:
            logging.error(f"[OllamaError] {exc.details} trace_id={trace_id}")
        return error_response(exc, HTTP_502_BAD_GATEWAY, trace_id)

    @app.exception_handler(ConfigError)
    async def config_error_handler(request: Request, exc: ConfigError):
        trace_id = str(uuid.uuid4())
        if LOG_ERRORS:
            logging.error(f"[ConfigError] {exc.details} trace_id={trace_id}")
        return error_response(exc, HTTP_500_INTERNAL_SERVER_ERROR, trace_id)

    @app.exception_handler(RequestValidationError)
    async def fastapi_validation_error_handler(request: Request, exc: RequestValidationError):
        trace_id = str(uuid.uuid4())
        if LOG_ERRORS:
            logging.warning(f"[FastAPIValidationError] {exc.errors()} trace_id={trace_id}")
        return error_response(ValidationError(details=exc.errors()), HTTP_400_BAD_REQUEST, trace_id)

    @app.exception_handler(Exception)
    async def generic_error_handler(request: Request, exc: Exception):
        trace_id = str(uuid.uuid4())
        if LOG_ERRORS:
            logging.error(f"[InternalError] {str(exc)} trace_id={trace_id}")
        return error_response(InternalError(details=str(exc)), HTTP_500_INTERNAL_SERVER_ERROR, trace_id) 