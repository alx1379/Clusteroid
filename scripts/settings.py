# Copyright 2025 Alex Erofeev / AIGENTTO
# Created by Alex Erofeev at AIGENTTO (http://aigentto.com/)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
from dotenv import load_dotenv

# Путь до .env от текущего файла
ENV_PATH = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=ENV_PATH)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "text-embedding-3-small")

USE_PROXY = os.getenv("USE_PROXY", "false").lower() == "true"
PROXY_URL = os.getenv("PROXY_URL", "")

def debug_settings():
    print(f"✅ Loaded settings from .env:")
    print(f"OPENAI_MODEL = {OPENAI_MODEL}")
    print(f"USE_PROXY = {USE_PROXY}")
    print(f"PROXY_URL = {PROXY_URL}")
