# src/intelligence/core/base_predictor.py

class BasePredictiveArchitecture(BaseIntelligenceModel):
    """
    JEPA와 같이 '예측' 기반 아키텍처를 위한 베이스 클래스.
    """
    def __init__(self):
        super().__init__()
        self.target_encoder = None  # EMA로 업데이트될 타겟 엔진
        self.context_encoder = None # 실제 학습될 컨텍스트 엔진
        self.predictor = None       # 잠재 공간에서의 변화를 시뮬레이션

    @abstractmethod
    def update_target_encoder(self, momentum: float):
        """
        EMA(Exponential Moving Average)를 통해 타겟 인코더를 업데이트합니다.
        """
        pass