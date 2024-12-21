from abc import ABC, abstractmethod


class BaseChain(ABC):
    """
    체인의 기본 클래스입니다.
    모든 체인 클래스는 이 클래스를 상속받아야 합니다.

    Attributes:
        model (str): 사용할 LLM 모델명
        temperature (float): 모델의 temperature 값
    """

    def __init__(self, model: str = "exaone", temperature: float = 0, **kwargs):
        self.model = model
        self.temperature = temperature
        self.kwargs = kwargs

    @abstractmethod
    def setup(self):
        """체인 설정을 위한 추상 메서드"""
        pass

    def create(self):
        """체인을 생성하고 반환합니다."""
        return self.setup()
