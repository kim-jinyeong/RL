import os

if os.environ.get('RLTRADER_BACKEND', 'pytorch') == 'pytorch':
    from app.networks.networks_pytorch import Network, LSTMNetwork, CNN


__all__ = [
    'Network', 'LSTMNetwork', 'CNN'
]
