"""Server-controlled choice schema; user text is always passed as state data."""
JARGON_QUESTIONS = {
    'jargon_detection': {
        'type': 'choice',
        'instructions': (
            '判断对话中的暗语是否用于违法或灰产活动。合法引用、反诈提醒、新闻报道、学术研究和合规咨询中的提及不算黑话。'
        ),
        'criteria': {
            'A': '暗语用于招揽、实施或掩饰违法、灰产活动。',
            'B': '普通合法交流，或仅在合法引用、反诈、新闻、研究、合规咨询中提及暗语。',
        },
    }
}

# Native options are mapped only at the service boundary; keep the public API stable.
CHOICE_LABEL_MAP = {'A': 'jargon', 'B': 'normal'}
