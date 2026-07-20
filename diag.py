#!/usr/bin/env python3
"""Diagnose API connection issues. Run this from ZTF_prompt/ directory."""
import sys, os, socket, ssl
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import config

print(f"=== Env ===")
print(f"Python: {sys.version}")
for var in ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'no_proxy', 'NO_PROXY',
            'REQUESTS_CA_BUNDLE', 'SSL_CERT_FILE', 'CURL_CA_BUNDLE']:
    val = os.environ.get(var)
    if val:
        print(f"  {var}={val}")

print(f"\n=== Package Versions ===")
for pkg in ['httpx', 'openai', 'httpcore']:
    try:
        m = __import__(pkg)
        v = getattr(m, '__version__', '?')
        print(f"  {pkg}=={v}")
    except ImportError:
        print(f"  {pkg}: NOT INSTALLED")

# Check httpx HTTP/2 support
try:
    import httpx
    print(f"  httpx http2_kwarg: {'http2' in httpx.Client.__init__.__code__.co_varnames}")
except:
    pass

print(f"\n=== urllib3 direct ===")
try:
    import urllib3
    urllib3.disable_warnings()
    pm = urllib3.PoolManager(cert_reqs='CERT_NONE')
    resp = pm.request('GET', f"{config.API_BASE_URL}/models", timeout=15.0)
    print(f"  HTTP {resp.status}")
except Exception as e:
    print(f"  FAIL: {type(e).__name__}: {e}")

print(f"\n=== requests ===")
try:
    import requests
    resp = requests.get(f"{config.API_BASE_URL}/models", timeout=15, verify=True)
    print(f"  HTTP {resp.status_code}")
except Exception as e:
    print(f"  FAIL: {type(e).__name__}: {e}")

print(f"\n=== requests (verify=False) ===")
try:
    import requests
    resp = requests.get(f"{config.API_BASE_URL}/models", timeout=15, verify=False)
    print(f"  HTTP {resp.status_code}")
except Exception as e:
    print(f"  FAIL: {type(e).__name__}: {e}")

print(f"\n=== DONE ===")