from typing import Optional
from fastapi import FastAPI
import datetime
from app.services import agent
import os
import sys
import logging
from fastapi.responses import FileResponse
from app.services import settings
from app.services import data_manager

app = FastAPI()

@app.get("/")
def now():
    pass
@app.get("/train")
def now():
    mode = 'train' #'train', 'test', 'update', 'predict'
    ver = 'v1' # 'v1', 'v2', 'v3', 'v4'
    name = '005930' # 종목코드
    stock_code = ['005930']
    rl_method = 'a2c' # 'dqn', 'pg', 'ac', 'a2c', 'a3c', 'monkey'
    net = 'cnn' # 'dnn', 'lstm', 'cnn', 'monkey'
    backend = 'pytorch'
    start_date = '2021-08-30'
    end_date = '2022-08-30'
    lr = 0.0001
    discount_factor = 0.7
    balance = 100000000

    # 학습기 파라미터 설정
    output_name = f'{mode}_{name}_{rl_method}_{net}'
    learning = mode in ['train', 'update']
    reuse_models = mode in ['test', 'update', 'predict']
    value_network_name = f'{name}_{rl_method}_{net}_value.mdl'
    policy_network_name = f'{name}_{rl_method}_{net}_policy.mdl'
    start_epsilon = 1 if mode in ['train', 'update'] else 0
    num_epoches = 1000 if mode in ['train', 'update'] else 1
    num_steps = 5 if net in ['lstm', 'cnn'] else 1

    # Backend 설정
    os.environ['RLTRADER_BACKEND'] = backend

    # 출력 경로 생성
    output_path = os.path.join(settings.BASE_DIR, 'output', output_name)
    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    # 파라미터 기록


    # 모델 경로 준비
    # 모델 포멧 PyTorch는 pickle
    value_network_path = os.path.join(settings.BASE_DIR, 'models', value_network_name)
    policy_network_path = os.path.join(settings.BASE_DIR, 'models', policy_network_name)

    # 로그 기록 설정
    log_path = os.path.join(output_path, f'{output_name}.log')
    if os.path.exists(log_path):
        os.remove(log_path)
    logging.basicConfig(format='%(message)s')
    logger = logging.getLogger(settings.LOGGER_NAME)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    file_handler = logging.FileHandler(filename=log_path, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    
    # Backend 설정, 로그 설정을 먼저하고 RLTrader 모듈들을 이후에 임포트해야 함
    from app.services.learners import A2CLearner

    common_params = {}
    list_stock_code = []
    list_chart_data = []
    list_training_data = []
    list_min_trading_price = []
    list_max_trading_price = []

    for stock_code in stock_code:
        # 차트 데이터, 학습 데이터 준비
        chart_data, training_data = data_manager.load_data(
            stock_code, start_date, end_date, ver=ver)

        assert len(chart_data) >= num_steps
        
        # 최소/최대 단일 매매 금액 설정
        min_trading_price = 100000
        max_trading_price = 10000000

        # 공통 파라미터 설정
        common_params = {'rl_method': rl_method,
            'net': net, 'num_steps': num_steps, 'lr': lr,
            'balance': balance, 'num_epoches': num_epoches,
            'discount_factor': discount_factor, 'start_epsilon': start_epsilon,
            'output_path': output_path, 'reuse_models': reuse_models}

        # 강화학습 시작
        common_params.update({'stock_code': stock_code,
            'chart_data': chart_data,
            'training_data': training_data,
            'min_trading_price': min_trading_price,
            'max_trading_price': max_trading_price})

        learner = A2CLearner(**{**common_params,
            'value_network_path': value_network_path,
            'policy_network_path': policy_network_path})

    
    assert learner is not None

    if mode in ['train', 'test', 'update']:
        learner.run(learning=learning)
        if mode in ['train', 'update']:
            learner.save_models()
    elif mode == 'predict':
        learner.predict()
    return 