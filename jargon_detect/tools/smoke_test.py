"""Test an already running service over HTTP. Never starts/stops a process."""
import argparse
import json
import sys
import urllib.error
import urllib.request

from app.config import Settings
from app.schemas import DetectResponse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', help='Default: http://127.0.0.1:<API_PORT>')
    args = parser.parse_args()
    base = (args.base_url or f'http://127.0.0.1:{Settings().api_port}').rstrip('/')
    # Local verification must not accidentally use HTTP_PROXY for loopback traffic.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def request(path, payload=None):
        data = json.dumps(payload, ensure_ascii=False).encode('utf-8') if payload is not None else None
        req = urllib.request.Request(base + path, data=data, headers={'Content-Type': 'application/json'})
        try:
            with opener.open(req, timeout=120) as response:
                return response.status, json.load(response)
        except urllib.error.HTTPError as exc:
            return exc.code, json.load(exc)

    try:
        status, ready = request('/ready')
        assert status == 200 and ready['ready'], ready
        assert request('/health') == (200, {'status': 'ok'})
        status, result = request('/api/v1/detect', {'dialogue': '我今天在图书馆学习。'})
        assert status == 200, result
        DetectResponse.model_validate(result)
        assert request('/api/v1/detect', {'dialogue': ' '})[0] == 422
        assert request('/api/v1/detect', {'text': '旧字段'})[0] == 422
        print(json.dumps({'status': 'ok', 'ready': ready, 'response': result}, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print(f'Smoke test failed: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
