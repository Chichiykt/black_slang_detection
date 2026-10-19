from unittest.mock import MagicMock, Mock

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.detector import ChoiceFormatError, InferenceError, ModelNotReadyError, parse_choice
from app.main import create_app


@pytest.fixture
def detector():
    result = Mock()
    result.ready = True
    result.device = 'cpu'
    result.load_error = None
    result.detect.return_value = parse_choice({'answers': {'jargon_detection': {
        'type': 'choice', 'choice': 'A', 'probabilities': {'A': 0.8, 'B': 0.2}}}})
    return result


@pytest.fixture
def client(detector):
    with TestClient(create_app(Settings(max_text_length=20), detector)) as client:
        yield client


def test_success_jargon(client, detector):
    response = client.post('/api/v1/detect', json={'dialogue': '这个菠菜平台还能上分吗？'})
    assert response.status_code == 200
    assert response.json()['data']['is_jargon'] is True
    assert response.json()['data']['probabilities'] == {'jargon': 0.8, 'normal': 0.2}
    detector.detect.assert_called_once_with('这个菠菜平台还能上分吗？')


def test_success_normal(client, detector):
    detector.detect.return_value = parse_choice({'answers': {'jargon_detection': {
        'type': 'choice', 'choice': 'B', 'probabilities': {'A': 0.1, 'B': 0.9}}}})
    response = client.post('/api/v1/detect', json={'dialogue': '我今天吃了菠菜。'})
    assert response.status_code == 200
    assert response.json()['data']['label'] == '非黑话'


@pytest.mark.parametrize('dialogue', ['', '   ', 1, True, None, [], {}, '字' * 21])
def test_invalid_dialogue(client, detector, dialogue):
    response = client.post('/api/v1/detect', json={'dialogue': dialogue})
    assert response.status_code == 422 and response.json()['code'] == 422
    detector.detect.assert_not_called()


def test_invalid_json(client):
    assert client.post('/api/v1/detect', content='{', headers={'Content-Type': 'application/json'}).status_code == 422


def test_user_cannot_submit_questions(client, detector):
    assert client.post('/api/v1/detect', json={'dialogue': '你好', 'questions': {'type': 'score'}}).status_code == 422
    detector.detect.assert_not_called()


@pytest.mark.parametrize('error,status', [(ModelNotReadyError('未就绪'), 503),
                                        (InferenceError('failed'), 500), (ChoiceFormatError('bad'), 500)])
def test_model_errors(client, detector, error, status):
    detector.detect.side_effect = error
    response = client.post('/api/v1/detect', json={'dialogue': '正常中文'})
    assert response.status_code == status and response.json()['data'] is None


def test_health_ready_docs(client):
    assert client.get('/health').json() == {'status': 'ok'}
    assert client.get('/ready').json()['ready'] is True
    assert client.get('/docs').status_code == 200


def test_load_failure_is_observable(detector):
    detector.load.side_effect = RuntimeError('missing weights')
    detector.load_error = 'missing weights'
    detector.ready = False
    detector.detect.side_effect = ModelNotReadyError('未就绪')
    with TestClient(create_app(Settings(), detector)) as client:
        assert client.get('/health').status_code == 200
        assert client.get('/ready').status_code == 503
        assert client.get('/ready').json()['error'] == 'missing weights'
        assert client.post('/api/v1/detect', json={'dialogue': '你好'}).status_code == 503
    detector.close.assert_called_once()



@pytest.mark.parametrize('option,business,label,probabilities', [
    ('A', 'jargon', '黑话', {'A': 0.8, 'B': 0.2}),
    ('B', 'normal', '非黑话', {'A': 0.1, 'B': 0.9}),
])
def test_native_sdk_to_http_contract(option, business, label, probabilities):
    from app.detector import LayaDetector
    detector = LayaDetector(Settings())
    detector.agent = MagicMock()
    detector.load = Mock()
    detector.agent.predict.return_value = {'answers': {'jargon_detection': {
        'type': 'choice', 'choice': option, 'probabilities': probabilities,
        'confidence': 0.35, 'answer_confidence': 0.67,
    }}}
    agent = detector.agent
    with TestClient(create_app(Settings(), detector)) as client:
        response = client.post('/api/v1/detect', json={'dialogue': '原样待检测对话'})
    assert agent.predict.call_args.args[0] == {'text': '原样待检测对话'}
    assert response.status_code == 200
    assert response.json() == {'code': 200, 'message': 'success', 'data': {
        'is_jargon': option == 'A', 'label': label, 'choice': business,
        'probabilities': {'jargon': probabilities['A'], 'normal': probabilities['B']},
        'answer_confidence': 0.67,
    }}
    assert detector.agent is None  # lifespan closes the reused agent


@pytest.mark.parametrize('raw', [None, {}, {'answers': {'jargon_detection': {
    'type': 'choice', 'choice': 'C', 'probabilities': {'A': 0.8, 'B': 0.2}}}}])
def test_malformed_sdk_output_returns_http_error(raw):
    from app.detector import LayaDetector
    detector = LayaDetector(Settings())
    detector.agent = MagicMock()
    detector.load = Mock()
    detector.agent.predict.return_value = raw
    with TestClient(create_app(Settings(), detector)) as client:
        response = client.post('/api/v1/detect', json={'dialogue': '你好'})
    assert response.status_code == 500
    assert response.json()['code'] == 500
    assert response.json()['data'] is None
    assert '模型' in response.json()['message']


@pytest.mark.parametrize('payload', [
    {'text': '旧字段'},
    {'dialogue': '新字段', 'text': '旧字段'},
    {},
])
def test_request_requires_dialogue_only(client, detector, payload):
    assert client.post('/api/v1/detect', json=payload).status_code == 422
    detector.detect.assert_not_called()


def test_openapi_uses_dialogue(client):
    schema = client.get('/openapi.json').json()['components']['schemas']['DetectRequest']
    assert schema['required'] == ['dialogue']
    assert set(schema['properties']) == {'dialogue'}
