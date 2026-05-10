import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class BaseIntelligenceModel(nn.Module, ABC):
    """
    모든 지능 모델(JEPA, GPT, BERT)의 최상위 추상 클래스.
    복원(Reconstruction)과 예측(Prediction) 로직을 공통으로 관리합니다.
    """
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x, **kwargs):
        """
        기본적인 데이터 흐름을 정의합니다.
        """
        pass

    @abstractmethod
    def compute_loss(self, outputs, targets):
        """
        모델별 학습 목적 함수(Loss Function)를 정의합니다.
        JEPA: L2 in Latent Space
        GPT/BERT: CrossEntropy in Pixel/Token Space
        """
        pass

    def get_representation(self, x):
        """
        학습된 인코더로부터 특징(Representation)을 추출합니다.
        다운스트림 태스크(Evaluation)에서 공통으로 사용됩니다.
        """
        raise NotImplementedError